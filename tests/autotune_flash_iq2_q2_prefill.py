#!/usr/bin/env python3
"""Offline, fail-closed Flash IQ2/Q2 SSD-prefill autotune artifact tool.

This tool deliberately cannot certify or enable a runtime path.  Until the
P5 long-context oracle is complete it emits only DIAGNOSTIC reports/caches;
the default lookup rejects those caches and returns the legacy variant.

Benchmark samples are separate ``ds4-bench`` processes in ABBA order.  The
tool never starts, stops, or probes a server, never drops the page cache, and
never tunes inside an inference process.  ``--dry-run`` is side-effect free
unless ``--output`` is supplied, in which case only the JSON plan is written.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable
import zlib


CACHE_SCHEMA = "ds4.flash_iq2_q2_prefill.autotune_cache.v1"
REPORT_SCHEMA = "ds4.flash_iq2_q2_prefill.autotune_report.v1"
SELECTOR_ABI = "flash-iq2-q2-prefill-selector-v1"
CACHE_STATUS = "DIAGNOSTIC"
CACHE_EOF = "DS4_FLASH_IQ2_Q2_PREFILL_CACHE_V1_EOF"
CACHE_MAX_BYTES = 1024 * 1024

LEGACY_VARIANT = "legacy_sorted_pairs"
MMQ_VARIANT = "mmq_raw_compact"
VARIANTS = (LEGACY_VARIANT, MMQ_VARIANT)

DEFAULT_SHAPES = (
    "2:12,3:18,4:24,5:30,8:48,12:72,16:96,20:120,32:192,"
    "64:256,128:256,256:256,512:256,1024:256"
)

EXACT_BATCH_ENV = {
    "DS4_CUDA_UNSAFE_IQ2_LAYER_MAJOR_PREFILL": "1",
    "DS4_METAL_DISABLE_STREAMING_DECODE_PREFILL": "1",
    "DS4_CUDA_Q8_BATCH_DECODE_EXACT": "1",
    "DS4_CUDA_ATTENTION_BATCH_DECODE_EXACT": "1",
    "DS4_CUDA_HC_BATCH_DECODE_EXACT": "1",
    "DS4_CUDA_ROUTER_BATCH_DECODE_EXACT": "1",
    "DS4_CUDA_COMPRESSOR_BATCH_DECODE_EXACT": "1",
    "DS4_CUDA_NO_Q8_F16_CACHE": "1",
}

SCRUB_ENV = {
    *EXACT_BATCH_ENV,
    "DS4_CUDA_MMQ",
    "DS4_CUDA_FLASH_IQ2_Q2_PREFILL_VARIANT",
    "DS4_CUDA_MOE_NO_SORTED_PAIRS",
    "DS4_CUDA_MOE_NO_EXPERT_TILES",
    "DS4_CUDA_MOE_NO_P2",
    "DS4_CUDA_MOE_ATOMIC_DOWN",
    "DS4_CUDA_MOE_NO_ATOMIC_DOWN",
    "DS4_CUDA_MOE_TILE4",
    "DS4_CUDA_MOE_TILE8",
    "DS4_MMQ_NO_YIND",
    "DS4_MMID_LARGE",
    "DS4_METAL_GRAPH_DUMP_PREFIX",
    "DS4_METAL_GRAPH_DUMP_NAME",
    "DS4_METAL_GRAPH_DUMP_LAYER",
    "DS4_METAL_GRAPH_DUMP_POS",
    "DS4_METAL_INDEXER_STAGE_PROFILE",
    "DS4_METAL_GRAPH_LAYER_PROFILE",
    "DS4_METAL_GRAPH_TOKEN_PROFILE",
    "DS4_CUDA_ATTN_OUTPUT_PROFILE",
}

BUILD_INPUTS = (
    "ds4.c",
    "ds4_cuda.cu",
    "ds4_gpu.h",
    "cuda/mmq/VENDOR.md",
    "cuda/mmq/ds4_mmq.h",
    "cuda/mmq/ds4_mmq.cu",
    "cuda/mmq/ds4_mmq_d2r.cu",
    "cuda/mmq/mmid.cu",
    "cuda/mmq/mmq.cuh",
    "cuda/mmq/quantize.cu",
    "cuda/mmq/vecdotq.cuh",
)


class AutotuneError(RuntimeError):
    pass


class CacheError(AutotuneError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json_dump(path: Path, value: Any) -> None:
    data = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, data)


def strict_json_loads(data: bytes) -> Any:
    def no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CacheError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=no_duplicate_object)
    except CacheError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheError(f"invalid cache JSON: {exc}") from exc


def require_dict(value: Any, label: str, keys: Iterable[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CacheError(f"{label} must be an object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CacheError(f"{label} fields mismatch: missing={missing} extra={extra}")
    return value


def require_string(value: Any, label: str, max_len: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > max_len:
        raise CacheError(f"{label} must be a non-empty string <= {max_len} bytes")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise CacheError(f"{label} contains a control character")
    return value


def require_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CacheError(f"{label} must be an integer >= {minimum}")
    return value


def require_float(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CacheError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise CacheError(f"{label} must be finite and >= {minimum}")
    return result


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise CacheError(f"{label} must be boolean")
    return value


@dataclasses.dataclass(frozen=True)
class HardwareKey:
    device_uuid: str
    compute_capability: str
    sm_count: int
    l2_bytes: int
    memory_bus_bits: int
    global_mem_bytes: int
    cuda_driver: str
    cuda_runtime: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> "HardwareKey":
        obj = require_dict(value, "hardware", (
            "device_uuid", "compute_capability", "sm_count", "l2_bytes",
            "memory_bus_bits", "global_mem_bytes", "cuda_driver",
            "cuda_runtime",
        ))
        return cls(
            require_string(obj["device_uuid"], "hardware.device_uuid"),
            require_string(obj["compute_capability"], "hardware.compute_capability"),
            require_int(obj["sm_count"], "hardware.sm_count"),
            require_int(obj["l2_bytes"], "hardware.l2_bytes"),
            require_int(obj["memory_bus_bits"], "hardware.memory_bus_bits"),
            require_int(obj["global_mem_bytes"], "hardware.global_mem_bytes"),
            require_string(obj["cuda_driver"], "hardware.cuda_driver"),
            require_string(obj["cuda_runtime"], "hardware.cuda_runtime"),
        )


@dataclasses.dataclass(frozen=True)
class BuildKey:
    source_sha256: str
    cuda_arch: str
    nvcc_flags_sha256: str
    mmq_vendor_commit: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> "BuildKey":
        obj = require_dict(value, "build", (
            "source_sha256", "cuda_arch", "nvcc_flags_sha256",
            "mmq_vendor_commit",
        ))
        result = cls(
            require_string(obj["source_sha256"], "build.source_sha256"),
            require_string(obj["cuda_arch"], "build.cuda_arch"),
            require_string(obj["nvcc_flags_sha256"], "build.nvcc_flags_sha256"),
            require_string(obj["mmq_vendor_commit"], "build.mmq_vendor_commit"),
        )
        if not re.fullmatch(r"[0-9a-f]{64}", result.source_sha256):
            raise CacheError("build.source_sha256 is not a lowercase SHA-256")
        if not re.fullmatch(r"[0-9a-f]{64}", result.nvcc_flags_sha256):
            raise CacheError("build.nvcc_flags_sha256 is not a lowercase SHA-256")
        if not re.fullmatch(r"[0-9a-f]{40}", result.mmq_vendor_commit):
            raise CacheError("build.mmq_vendor_commit is not a 40-digit commit id")
        return result


@dataclasses.dataclass(frozen=True)
class ModelKey:
    identity: str
    variant: str
    gate_type: int
    up_type: int
    down_type: int
    expert_in_dim: int
    expert_mid_dim: int
    out_dim: int
    total_experts: int
    top_k: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> "ModelKey":
        obj = require_dict(value, "model", (
            "identity", "variant", "gate_type", "up_type", "down_type",
            "expert_in_dim", "expert_mid_dim", "out_dim", "total_experts",
            "top_k",
        ))
        result = cls(
            require_string(obj["identity"], "model.identity"),
            require_string(obj["variant"], "model.variant"),
            require_int(obj["gate_type"], "model.gate_type", 1),
            require_int(obj["up_type"], "model.up_type", 1),
            require_int(obj["down_type"], "model.down_type", 1),
            require_int(obj["expert_in_dim"], "model.expert_in_dim", 1),
            require_int(obj["expert_mid_dim"], "model.expert_mid_dim", 1),
            require_int(obj["out_dim"], "model.out_dim", 1),
            require_int(obj["total_experts"], "model.total_experts", 1),
            require_int(obj["top_k"], "model.top_k", 1),
        )
        if result.top_k > result.total_experts:
            raise CacheError("model.top_k exceeds model.total_experts")
        if result.variant != "flash" or (
            result.gate_type, result.up_type, result.down_type
        ) != (16, 16, 10):
            raise CacheError("cache model is not Flash IQ2_XXS/IQ2_XXS/Q2_K")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", result.identity):
            raise CacheError("cache model identity is not sha256:<64 lowercase hex>")
        return result


@dataclasses.dataclass(frozen=True, order=True)
class ShapeKey:
    n_tokens: int
    compact_experts: int
    layout: str = "ssd_raw_compact_v1"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> "ShapeKey":
        obj = require_dict(value, "shape", ("n_tokens", "compact_experts", "layout"))
        result = cls(
            require_int(obj["n_tokens"], "shape.n_tokens", 2),
            require_int(obj["compact_experts"], "shape.compact_experts", 1),
            require_string(obj["layout"], "shape.layout"),
        )
        if result.layout != "ssd_raw_compact_v1":
            raise CacheError(f"unsupported shape layout: {result.layout}")
        return result


@dataclasses.dataclass(frozen=True)
class CacheEntry:
    shape: ShapeKey
    variant: str
    exact_pairwise: bool
    legacy_samples: int
    mmq_samples: int
    legacy_median_us: float
    mmq_median_us: float
    winner_p95_us: float
    speedup_ppm: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape.to_dict(),
            "variant": self.variant,
            "exact_pairwise": self.exact_pairwise,
            "legacy_samples": self.legacy_samples,
            "mmq_samples": self.mmq_samples,
            "legacy_median_us": self.legacy_median_us,
            "mmq_median_us": self.mmq_median_us,
            "winner_p95_us": self.winner_p95_us,
            "speedup_ppm": self.speedup_ppm,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "CacheEntry":
        obj = require_dict(value, "entry", (
            "shape", "variant", "exact_pairwise", "legacy_samples",
            "mmq_samples", "legacy_median_us", "mmq_median_us",
            "winner_p95_us", "speedup_ppm",
        ))
        variant = require_string(obj["variant"], "entry.variant")
        if variant not in VARIANTS:
            raise CacheError(f"unsupported entry variant: {variant}")
        return cls(
            ShapeKey.from_dict(obj["shape"]),
            variant,
            require_bool(obj["exact_pairwise"], "entry.exact_pairwise"),
            require_int(obj["legacy_samples"], "entry.legacy_samples"),
            require_int(obj["mmq_samples"], "entry.mmq_samples"),
            require_float(obj["legacy_median_us"], "entry.legacy_median_us"),
            require_float(obj["mmq_median_us"], "entry.mmq_median_us"),
            require_float(obj["winner_p95_us"], "entry.winner_p95_us"),
            require_int(obj["speedup_ppm"], "entry.speedup_ppm"),
        )


@dataclasses.dataclass(frozen=True)
class TuningCache:
    hardware: HardwareKey
    build: BuildKey
    model: ModelKey
    oracle: str
    oracle_digest: str
    entries: tuple[CacheEntry, ...]
    created_at: str
    status: str = CACHE_STATUS

    def payload(self) -> dict[str, Any]:
        return {
            "schema": CACHE_SCHEMA,
            "selector_abi": SELECTOR_ABI,
            "status": self.status,
            "created_at": self.created_at,
            "hardware": self.hardware.to_dict(),
            "build": self.build.to_dict(),
            "model": self.model.to_dict(),
            "oracle": self.oracle,
            "oracle_digest": self.oracle_digest,
            "entry_count": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclasses.dataclass(frozen=True)
class LookupKey:
    hardware: HardwareKey
    build: BuildKey
    model: ModelKey
    shape: ShapeKey


@dataclasses.dataclass(frozen=True)
class LookupResult:
    variant: str
    hit: bool
    reason: str


def encode_cache(cache: TuningCache) -> bytes:
    if cache.status != CACHE_STATUS:
        raise CacheError("P5 is not certified: only DIAGNOSTIC caches may be emitted")
    ordered = tuple(sorted(cache.entries, key=lambda entry: entry.shape))
    if ordered != cache.entries:
        raise CacheError("cache entries must be sorted by exact shape key")
    if len({entry.shape for entry in cache.entries}) != len(cache.entries):
        raise CacheError("cache contains duplicate shape keys")
    incomplete = cache_key_completeness_errors(cache.hardware, cache.model)
    if incomplete:
        raise CacheError(
            "cache has an incomplete exact key: " + "; ".join(incomplete)
        )
    payload = cache.payload()
    payload_bytes = canonical_json_bytes(payload)
    envelope = {
        "payload": payload,
        "crc32": f"{zlib.crc32(payload_bytes) & 0xFFFFFFFF:08x}",
        "eof": CACHE_EOF,
    }
    return json.dumps(envelope, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def write_cache(path: Path, cache: TuningCache) -> None:
    atomic_write_bytes(path, encode_cache(cache))


def parse_cache_bytes(data: bytes) -> TuningCache:
    if len(data) > CACHE_MAX_BYTES:
        raise CacheError("cache exceeds the 1 MiB safety limit")
    envelope = require_dict(strict_json_loads(data), "cache envelope", (
        "payload", "crc32", "eof",
    ))
    if envelope["eof"] != CACHE_EOF:
        raise CacheError("cache EOF marker is missing or invalid")
    crc = require_string(envelope["crc32"], "cache.crc32", 8)
    if not re.fullmatch(r"[0-9a-f]{8}", crc):
        raise CacheError("cache CRC32 must be eight lowercase hex digits")
    payload_obj = envelope["payload"]
    actual_crc = f"{zlib.crc32(canonical_json_bytes(payload_obj)) & 0xFFFFFFFF:08x}"
    if actual_crc != crc:
        raise CacheError(f"cache CRC32 mismatch: expected {crc}, computed {actual_crc}")

    payload = require_dict(payload_obj, "cache payload", (
        "schema", "selector_abi", "status", "created_at", "hardware",
        "build", "model", "oracle", "oracle_digest", "entry_count",
        "entries",
    ))
    if payload["schema"] != CACHE_SCHEMA:
        raise CacheError(f"unsupported cache schema: {payload['schema']!r}")
    if payload["selector_abi"] != SELECTOR_ABI:
        raise CacheError(f"stale selector ABI: {payload['selector_abi']!r}")
    if payload["status"] != CACHE_STATUS:
        raise CacheError("P5 is not certified: cache status must be DIAGNOSTIC")
    if not isinstance(payload["entries"], list):
        raise CacheError("cache entries must be an array")
    count = require_int(payload["entry_count"], "cache.entry_count")
    if count != len(payload["entries"]):
        raise CacheError("cache entry_count does not match entries length")
    entries = tuple(CacheEntry.from_dict(item) for item in payload["entries"])
    if tuple(sorted(entries, key=lambda entry: entry.shape)) != entries:
        raise CacheError("cache entries are not sorted by exact shape key")
    if len({entry.shape for entry in entries}) != len(entries):
        raise CacheError("cache contains duplicate shape keys")
    cache = TuningCache(
        hardware=HardwareKey.from_dict(payload["hardware"]),
        build=BuildKey.from_dict(payload["build"]),
        model=ModelKey.from_dict(payload["model"]),
        oracle=require_string(payload["oracle"], "cache.oracle"),
        oracle_digest=require_string(payload["oracle_digest"], "cache.oracle_digest"),
        entries=entries,
        created_at=require_string(payload["created_at"], "cache.created_at"),
        status=payload["status"],
    )
    incomplete = cache_key_completeness_errors(cache.hardware, cache.model)
    if incomplete:
        raise CacheError(
            "cache has an incomplete exact key: " + "; ".join(incomplete)
        )
    return cache


def read_cache(path: Path) -> TuningCache:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CacheError(f"cannot read cache: {exc}") from exc
    return parse_cache_bytes(data)


def lookup_cache(
    cache: TuningCache,
    key: LookupKey,
    *,
    allow_diagnostic: bool = False,
) -> LookupResult:
    if cache.status != "EXACT" and not allow_diagnostic:
        return LookupResult(LEGACY_VARIANT, False, "diagnostic_cache_rejected")
    if cache.hardware != key.hardware:
        return LookupResult(LEGACY_VARIANT, False, "stale_hardware_key")
    if cache.build != key.build:
        return LookupResult(LEGACY_VARIANT, False, "stale_build_key")
    if cache.model != key.model:
        return LookupResult(LEGACY_VARIANT, False, "stale_model_key")
    for entry in cache.entries:
        if entry.shape == key.shape:
            if not entry.exact_pairwise:
                return LookupResult(LEGACY_VARIANT, False, "entry_not_pairwise_exact")
            return LookupResult(entry.variant, True, "exact_key_match")
    return LookupResult(LEGACY_VARIANT, False, "shape_key_miss")


def parse_shapes(spec: str, total_experts: int) -> list[ShapeKey]:
    result: list[ShapeKey] = []
    seen: set[ShapeKey] = set()
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        left, sep, right = item.partition(":")
        if not sep:
            raise AutotuneError(f"shape {item!r} must be TOKENS:COMPACT_EXPERTS")
        try:
            n_tokens = int(left, 10)
            compact_experts = int(right, 10)
        except ValueError as exc:
            raise AutotuneError(f"invalid shape {item!r}: {exc}") from exc
        if n_tokens < 2 or compact_experts < 1:
            raise AutotuneError(
                f"shape {item!r} requires tokens >= 2 and compact experts >= 1"
            )
        shape = ShapeKey(n_tokens, compact_experts)
        if shape.compact_experts > total_experts:
            raise AutotuneError(
                f"shape {item!r} exceeds total_experts={total_experts}"
            )
        if shape in seen:
            raise AutotuneError(f"duplicate shape {item!r}")
        seen.add(shape)
        result.append(shape)
    if not result:
        raise AutotuneError("shape list is empty")
    return sorted(result)


def variant_schedule(repeats: int, order: str) -> list[str]:
    if repeats <= 0:
        raise AutotuneError("repeats must be positive")
    if order == "legacy-first":
        return [LEGACY_VARIANT] * repeats + [MMQ_VARIANT] * repeats
    if order == "mmq-first":
        return [MMQ_VARIANT] * repeats + [LEGACY_VARIANT] * repeats
    if order != "abba":
        raise AutotuneError(f"unsupported order: {order}")
    pattern = (LEGACY_VARIANT, MMQ_VARIANT, MMQ_VARIANT, LEGACY_VARIANT)
    counts = {variant: 0 for variant in VARIANTS}
    result: list[str] = []
    while any(counts[variant] < repeats for variant in VARIANTS):
        for variant in pattern:
            if counts[variant] >= repeats:
                continue
            result.append(variant)
            counts[variant] += 1
    return result


def parse_key_value(items: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key or "\x00" in key or "\x00" in value:
            raise AutotuneError(f"expected KEY=VALUE, got {item!r}")
        result[key] = value
    return result


def build_environment(args: argparse.Namespace, variant: str) -> tuple[dict[str, str], dict[str, str]]:
    env = os.environ.copy()
    explicit = parse_key_value(args.env)
    for key in SCRUB_ENV:
        if key not in explicit:
            env.pop(key, None)
    env.update(EXACT_BATCH_ENV)
    env["DS4_CUDA_MMQ"] = "0" if variant == LEGACY_VARIANT else "1"
    # P7 will consume this exact spelling.  It is inert in the current
    # runtime, so all artifacts remain explicitly DIAGNOSTIC.
    env["DS4_CUDA_FLASH_IQ2_Q2_PREFILL_VARIANT"] = variant
    env.update(explicit)
    if args.cuda_home:
        cuda_home = str(Path(args.cuda_home).resolve())
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = cuda_home + "/bin:" + env.get("PATH", "")
        old_library = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = cuda_home + "/lib64" + (
            ":" + old_library if old_library else ""
        )
    recorded_keys = {
        *EXACT_BATCH_ENV,
        "DS4_CUDA_MMQ",
        "DS4_CUDA_FLASH_IQ2_Q2_PREFILL_VARIANT",
        *explicit,
    }
    if args.cuda_home:
        recorded_keys.add("CUDA_HOME")
    return env, {key: env[key] for key in sorted(recorded_keys) if key in env}


def benchmark_command(
    args: argparse.Namespace,
    shape: ShapeKey,
    csv_path: Path,
    artifact_dir: Path,
) -> list[str]:
    command = [
        str(args.binary_path),
        "--cuda",
        "-m",
        str(args.model_path),
        "--ssd-streaming",
        "--ssd-streaming-cache-experts",
        args.expert_cache,
        "--prefill-chunk",
        str(shape.n_tokens),
        "--prompt-file",
        str(args.prompt_path),
        "--ctx-start",
        str(shape.n_tokens),
        "--ctx-max",
        str(shape.n_tokens),
        "--ctx-alloc",
        str(shape.n_tokens + args.gen_tokens + 1),
        "--step-incr",
        "1",
        "--gen-tokens",
        str(args.gen_tokens),
        "--csv",
        str(csv_path),
        "--dump-frontier-logits-dir",
        str(artifact_dir),
    ]
    if args.gpu_vram:
        command += ["--gpu-vram", args.gpu_vram]
    if args.gpu_devices:
        command += ["--gpu-devices", args.gpu_devices]
    if args.warm_weights:
        command.append("--warm-weights")
    command.extend(args.bench_arg)
    return command


def build_plan(args: argparse.Namespace, shapes: list[ShapeKey], artifact_root: Path) -> list[dict[str, Any]]:
    schedule = variant_schedule(args.repeats, args.order)
    plan: list[dict[str, Any]] = []
    sequence = 0
    for shape in shapes:
        ordinals = {variant: 0 for variant in VARIANTS}
        for variant in schedule:
            ordinal = ordinals[variant]
            ordinals[variant] += 1
            run_id = (
                f"t{shape.n_tokens:05d}-c{shape.compact_experts:03d}-"
                f"{variant}-r{ordinal:02d}-s{sequence:04d}"
            )
            directory = artifact_root / run_id
            csv_path = directory / "metrics.csv"
            _, recorded_env = build_environment(args, variant)
            command = benchmark_command(args, shape, csv_path, directory)
            plan.append({
                "sequence": sequence,
                "run_id": run_id,
                "shape": shape.to_dict(),
                "variant": variant,
                "variant_ordinal": ordinal,
                "artifact_dir": str(directory),
                "csv_path": str(csv_path),
                "command": command,
                "command_display": shlex.join(command),
                "environment": recorded_env,
            })
            sequence += 1
    return plan


def source_fingerprint(root: Path, cuda_arch: str, nvcc_flags: str) -> BuildKey:
    digest = hashlib.sha256()
    for relative in BUILD_INPUTS:
        path = root / relative
        digest.update(relative.encode("utf-8") + b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
        digest.update(b"\0")
    digest.update(cuda_arch.encode("utf-8") + b"\0")
    digest.update(nvcc_flags.encode("utf-8") + b"\0")
    vendor_commit = "unresolved"
    try:
        vendor_text = (root / "cuda/mmq/VENDOR.md").read_text(encoding="utf-8")
        match = re.search(r"Commit\s*\|\s*`([0-9a-f]{40})`", vendor_text)
        if match:
            vendor_commit = match.group(1)
    except OSError:
        pass
    return BuildKey(
        source_sha256=digest.hexdigest(),
        cuda_arch=cuda_arch or "unresolved",
        nvcc_flags_sha256=hashlib.sha256(nvcc_flags.encode("utf-8")).hexdigest(),
        mmq_vendor_commit=vendor_commit,
    )


def diagnostic_model_identity(path: Path) -> str:
    try:
        stat = path.stat()
        material = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}"
        return "diagnostic-stat-sha256:" + hashlib.sha256(material.encode()).hexdigest()
    except OSError:
        return "diagnostic-missing-sha256:" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()


def keys_from_args(args: argparse.Namespace) -> tuple[HardwareKey, BuildKey, ModelKey]:
    hardware = HardwareKey(
        device_uuid=args.device_uuid,
        compute_capability=args.compute_capability,
        sm_count=args.sm_count,
        l2_bytes=args.l2_bytes,
        memory_bus_bits=args.memory_bus_bits,
        global_mem_bytes=args.global_mem_bytes,
        cuda_driver=args.cuda_driver,
        cuda_runtime=args.cuda_runtime,
    )
    build = source_fingerprint(args.repo_root_path, args.cuda_arch, args.nvcc_flags)
    model = ModelKey(
        identity=args.model_identity or diagnostic_model_identity(args.model_path),
        variant="flash",
        gate_type=16,
        up_type=16,
        down_type=10,
        expert_in_dim=args.expert_in_dim,
        expert_mid_dim=args.expert_mid_dim,
        out_dim=args.out_dim,
        total_experts=args.total_experts,
        top_k=args.top_k,
    )
    return hardware, build, model


def cache_key_completeness_errors(
    hardware: HardwareKey, model: ModelKey
) -> list[str]:
    errors: list[str] = []
    for label, value in (
        ("device_uuid", hardware.device_uuid),
        ("compute_capability", hardware.compute_capability),
        ("cuda_driver", hardware.cuda_driver),
        ("cuda_runtime", hardware.cuda_runtime),
    ):
        if "unresolved" in value:
            errors.append(f"hardware.{label} is unresolved")
    for label, value in (
        ("sm_count", hardware.sm_count),
        ("l2_bytes", hardware.l2_bytes),
        ("memory_bus_bits", hardware.memory_bus_bits),
        ("global_mem_bytes", hardware.global_mem_bytes),
    ):
        if value <= 0:
            errors.append(f"hardware.{label} must be positive")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", model.identity):
        errors.append("model.identity must be an explicit sha256:<64 lowercase hex> value")
    return errors


def read_single_csv(path: Path, expected_tokens: int) -> tuple[dict[str, float], list[str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        return {}, [f"cannot read CSV: {exc}"]
    if len(rows) != 1:
        return {}, [f"expected one CSV row, found {len(rows)}"]
    result: dict[str, float] = {}
    errors: list[str] = []
    for key, raw in rows[0].items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"CSV field {key} is not numeric: {raw!r}")
            continue
        if not math.isfinite(value):
            errors.append(f"CSV field {key} is non-finite")
            continue
        result[key] = value
    if int(result.get("prefill_tokens", -1)) != expected_tokens:
        errors.append("CSV prefill_tokens does not match the exact shape")
    return result, errors


def load_json(path: Path, label: str) -> tuple[dict[str, Any], list[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {}, [f"cannot read {label}: {exc}"]
    if not isinstance(value, dict):
        return {}, [f"{label} is not an object"]
    return value, []


def f32_bits(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return struct.unpack("<I", struct.pack("<f", number))[0]


def inspect_run(run: dict[str, Any], gen_tokens: int) -> dict[str, Any]:
    shape = ShapeKey.from_dict(run["shape"])
    directory = Path(run["artifact_dir"])
    frontier = shape.n_tokens
    csv_row, errors = read_single_csv(Path(run["csv_path"]), frontier)
    logits_path = directory / f"frontier_{frontier:06d}.logits.json"
    tokens_path = directory / f"frontier_{frontier:06d}.tokens.json"
    logits, logits_errors = load_json(logits_path, "frontier logits")
    tokens, token_errors = load_json(tokens_path, "greedy token sidecar")
    errors.extend(logits_errors)
    errors.extend(token_errors)

    values = logits.get("logits")
    if not isinstance(values, list) or not values:
        errors.append("frontier logits has no complete logits array")
        values = []
    bits = [f32_bits(value) for value in values]
    if any(value is None for value in bits):
        errors.append("frontier logits contains invalid or non-finite values")
    token_ids = tokens.get("token_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in token_ids
    ):
        errors.append("greedy token sidecar has no valid token_ids")
        token_ids = []
    if len(token_ids) != gen_tokens:
        errors.append(f"expected {gen_tokens} greedy tokens, found {len(token_ids)}")

    prefill_tps = csv_row.get("prefill_tps", 0.0)
    prefill_us = frontier / prefill_tps * 1.0e6 if prefill_tps > 0 else 0.0
    return {
        **run,
        "errors": errors,
        "metrics": {"prefill_tps": prefill_tps, "prefill_us": prefill_us},
        "artifacts": {
            "logits_path": str(logits_path),
            "tokens_path": str(tokens_path),
            "logits_sha256_f32_bits": hashlib.sha256(
                b"".join(struct.pack("<I", value or 0) for value in bits)
            ).hexdigest(),
            "_logits_f32_bits": bits,
            "token_ids": token_ids,
            "argmax_id": logits.get("argmax_id"),
        },
    }


def compare_runs(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if left.get("errors"):
        errors.append("legacy run invalid")
    if right.get("errors"):
        errors.append("MMQ run invalid")
    left_art = left.get("artifacts", {})
    right_art = right.get("artifacts", {})
    logits_exact = left_art.get("_logits_f32_bits") == right_art.get("_logits_f32_bits")
    tokens_exact = left_art.get("token_ids") == right_art.get("token_ids")
    argmax_exact = left_art.get("argmax_id") == right_art.get("argmax_id")
    if not logits_exact:
        errors.append("complete frontier logits differ")
    if not tokens_exact:
        errors.append("greedy token ids differ")
    if not argmax_exact:
        errors.append("argmax ids differ")
    return {
        "shape": left["shape"],
        "legacy_run_id": left["run_id"],
        "mmq_run_id": right["run_id"],
        "exact": not errors,
        "logits_exact": logits_exact,
        "tokens_exact": tokens_exact,
        "argmax_exact": argmax_exact,
        "errors": errors,
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(
    shapes: list[ShapeKey],
    runs: list[dict[str, Any]],
    min_speedup: float,
) -> tuple[list[CacheEntry], list[dict[str, Any]], str]:
    entries: list[CacheEntry] = []
    comparisons: list[dict[str, Any]] = []
    for shape in shapes:
        matching = [run for run in runs if ShapeKey.from_dict(run["shape"]) == shape]
        by_variant = {
            variant: sorted(
                (run for run in matching if run["variant"] == variant),
                key=lambda run: run["variant_ordinal"],
            )
            for variant in VARIANTS
        }
        pair_count = min(len(by_variant[LEGACY_VARIANT]), len(by_variant[MMQ_VARIANT]))
        shape_comparisons = [
            compare_runs(by_variant[LEGACY_VARIANT][index], by_variant[MMQ_VARIANT][index])
            for index in range(pair_count)
        ]
        comparisons.extend(shape_comparisons)
        pairwise_exact = pair_count > 0 and all(item["exact"] for item in shape_comparisons)
        legacy_times = [
            run["metrics"]["prefill_us"] for run in by_variant[LEGACY_VARIANT]
            if not run.get("errors") and run["metrics"]["prefill_us"] > 0
        ]
        mmq_times = [
            run["metrics"]["prefill_us"] for run in by_variant[MMQ_VARIANT]
            if not run.get("errors") and run["metrics"]["prefill_us"] > 0
        ]
        legacy_median = statistics.median(legacy_times) if legacy_times else 0.0
        mmq_median = statistics.median(mmq_times) if mmq_times else 0.0
        speedup = legacy_median / mmq_median if legacy_median > 0 and mmq_median > 0 else 0.0
        winner = MMQ_VARIANT if pairwise_exact and speedup >= min_speedup else LEGACY_VARIANT
        winner_times = mmq_times if winner == MMQ_VARIANT else legacy_times
        entries.append(CacheEntry(
            shape=shape,
            variant=winner,
            exact_pairwise=pairwise_exact,
            legacy_samples=len(legacy_times),
            mmq_samples=len(mmq_times),
            legacy_median_us=legacy_median,
            mmq_median_us=mmq_median,
            winner_p95_us=percentile(winner_times, 0.95),
            speedup_ppm=max(0, int(round(speedup * 1_000_000.0))),
        ))
    oracle_digest = hashlib.sha256(canonical_json_bytes(comparisons)).hexdigest()
    return entries, comparisons, oracle_digest


def run_plan(args: argparse.Namespace, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed_runs: list[dict[str, Any]] = []
    for planned in plan:
        directory = Path(planned["artifact_dir"])
        directory.mkdir(parents=True, exist_ok=False)
        stdout_path = directory / "stdout.log"
        stderr_path = directory / "stderr.log"
        env, _ = build_environment(args, planned["variant"])
        started = time.monotonic()
        errors: list[str] = []
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                result = subprocess.run(
                    planned["command"], cwd=args.repo_root_path, env=env,
                    stdout=stdout, stderr=stderr, timeout=args.timeout,
                    check=False,
                )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
            errors.append(f"benchmark exceeded timeout {args.timeout}s")
        except OSError as exc:
            returncode = 127
            errors.append(f"cannot launch benchmark: {exc}")
        if returncode != 0:
            errors.append(f"benchmark exited with status {returncode}")
        raw = {
            **planned,
            "returncode": returncode,
            "wall_s": time.monotonic() - started,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        inspected = inspect_run(raw, args.gen_tokens)
        inspected["errors"] = errors + inspected["errors"]
        completed_runs.append(inspected)
    return completed_runs


def make_report(
    args: argparse.Namespace,
    shapes: list[ShapeKey],
    plan: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    entries: list[CacheEntry],
    comparisons: list[dict[str, Any]],
    hardware: HardwareKey,
    build: BuildKey,
    model: ModelKey,
    oracle_digest: str,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    public_runs: list[dict[str, Any]] = []
    for run in runs:
        public = dict(run)
        artifacts = dict(public.get("artifacts", {}))
        artifacts.pop("_logits_f32_bits", None)
        if artifacts:
            public["artifacts"] = artifacts
        public_runs.append(public)
    return {
        "schema": REPORT_SCHEMA,
        "status": "DRY_RUN" if dry_run else CACHE_STATUS,
        "created_at": utc_now(),
        "hardware": hardware.to_dict(),
        "build": build.to_dict(),
        "model": model.to_dict(),
        "shapes": [shape.to_dict() for shape in shapes],
        "schedule_per_shape": variant_schedule(args.repeats, args.order),
        "plan": plan,
        "runs": public_runs,
        "comparisons": comparisons,
        "entries": [entry.to_dict() for entry in entries],
        "oracle": "pairwise-frontier-logits-f32-bits+argmax+greedy-v1",
        "oracle_digest": oracle_digest,
        "limitations": [
            "DIAGNOSTIC only: P5 long-context safe-vs-layer-major certification is absent.",
            "The current runtime does not consume the compact-MMQ selector variant; P7 wiring is out of scope.",
            "A performance cache is never evidence that a candidate is logits-exact.",
        ],
        "side_effect_contract": {
            "strictly_sequential": True,
            "process_per_sample": True,
            "controls_server": False,
            "restarts_server": False,
            "flushes_page_cache": False,
            "runs_gpu_in_dry_run": False,
            "runtime_selector_enabled": False,
        },
    }


def self_test() -> None:
    hardware = HardwareKey("GPU-test", "sm_120", 16, 1024, 256, 4096, "1", "2")
    build = BuildKey("a" * 64, "sm_120", "b" * 64, "c" * 40)
    model = ModelKey("sha256:" + "d" * 64, "flash", 16, 16, 10, 4096, 2048, 4096, 256, 6)
    shape = ShapeKey(20, 96)
    entry = CacheEntry(shape, MMQ_VARIANT, True, 3, 3, 100.0, 80.0, 82.0, 1_250_000)
    cache = TuningCache(
        hardware, build, model,
        "pairwise-frontier-logits-f32-bits+argmax+greedy-v1",
        "e" * 64, (entry,), "2026-08-14T00:00:00+00:00",
    )
    encoded = encode_cache(cache)
    parsed = parse_cache_bytes(encoded)
    assert parsed == cache
    key = LookupKey(hardware, build, model, shape)

    rejected = lookup_cache(parsed, key)
    assert not rejected.hit and rejected.variant == LEGACY_VARIANT
    assert rejected.reason == "diagnostic_cache_rejected"
    accepted = lookup_cache(parsed, key, allow_diagnostic=True)
    assert accepted.hit and accepted.variant == MMQ_VARIANT

    stale_hardware = dataclasses.replace(hardware, cuda_driver="stale")
    stale = lookup_cache(
        parsed, LookupKey(stale_hardware, build, model, shape),
        allow_diagnostic=True,
    )
    assert not stale.hit and stale.reason == "stale_hardware_key"
    stale_build = lookup_cache(
        parsed,
        LookupKey(hardware, dataclasses.replace(build, cuda_arch="sm_121"), model, shape),
        allow_diagnostic=True,
    )
    assert not stale_build.hit and stale_build.reason == "stale_build_key"
    stale_model = lookup_cache(
        parsed,
        LookupKey(hardware, build, dataclasses.replace(model, top_k=8), shape),
        allow_diagnostic=True,
    )
    assert not stale_model.hit and stale_model.reason == "stale_model_key"
    miss = lookup_cache(
        parsed, LookupKey(hardware, build, model, ShapeKey(21, 96)),
        allow_diagnostic=True,
    )
    assert not miss.hit and miss.reason == "shape_key_miss"

    corrupt_obj = strict_json_loads(encoded)
    corrupt_obj["payload"]["hardware"]["device_uuid"] = "GPU-corrupt"
    corrupt = json.dumps(corrupt_obj, sort_keys=True).encode("utf-8")
    try:
        parse_cache_bytes(corrupt)
    except CacheError as exc:
        assert "CRC32 mismatch" in str(exc)
    else:
        raise AssertionError("corrupt cache was accepted")

    bad_eof_obj = strict_json_loads(encoded)
    bad_eof_obj["eof"] = "truncated"
    try:
        parse_cache_bytes(json.dumps(bad_eof_obj).encode("utf-8"))
    except CacheError as exc:
        assert "EOF marker" in str(exc)
    else:
        raise AssertionError("cache without EOF marker was accepted")

    assert variant_schedule(2, "abba") == [
        LEGACY_VARIANT, MMQ_VARIANT, MMQ_VARIANT, LEGACY_VARIANT,
    ]
    assert parse_shapes("2:12,20:96", 256) == [ShapeKey(2, 12), shape]

    with tempfile.TemporaryDirectory(prefix="ds4-autotune-selftest-") as temporary:
        path = Path(temporary) / "nested" / "cache.json"
        write_cache(path, cache)
        assert read_cache(path) == cache
        output = Path(temporary) / "report.json"
        atomic_json_dump(output, {"status": "PASS"})
        assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"
    print("autotune_flash_iq2_q2_prefill self-test: PASS")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Offline ABBA planner/runner and fail-closed V1 cache tool for "
            "Flash IQ2/Q2 SSD batch prefill. Outputs remain DIAGNOSTIC until P5."
        )
    )
    parser.add_argument("--repo-root", default=str(root))
    parser.add_argument("--binary", default="ds4-bench")
    parser.add_argument("--model", default="ds4flash.gguf")
    parser.add_argument("--prompt-file", default="tests/long_context_story_prompt.txt")
    parser.add_argument("--output", help="atomic JSON report path")
    parser.add_argument(
        "--diagnostic-cache",
        help="optional atomic V1 cache artifact; default lookup rejects it",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--shapes", default=DEFAULT_SHAPES, metavar="TOKENS:COMPACT,...")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--order", choices=("abba", "legacy-first", "mmq-first"), default="abba"
    )
    parser.add_argument("--gen-tokens", type=int, default=8)
    parser.add_argument("--expert-cache", default="10GB")
    parser.add_argument("--gpu-vram")
    parser.add_argument("--gpu-devices")
    parser.add_argument("--warm-weights", action="store_true")
    parser.add_argument("--cuda-home", default=os.environ.get("CUDA_HOME", ""))
    parser.add_argument("--cuda-arch", default=os.environ.get("CUDA_ARCH", "unresolved"))
    parser.add_argument("--nvcc-flags", default=os.environ.get("NVCCFLAGS", ""))
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--bench-arg", action="append", default=[])

    parser.add_argument("--device-uuid", default="diagnostic-unresolved")
    parser.add_argument("--compute-capability", default="diagnostic-unresolved")
    parser.add_argument("--sm-count", type=int, default=0)
    parser.add_argument("--l2-bytes", type=int, default=0)
    parser.add_argument("--memory-bus-bits", type=int, default=0)
    parser.add_argument("--global-mem-bytes", type=int, default=0)
    parser.add_argument("--cuda-driver", default="diagnostic-unresolved")
    parser.add_argument("--cuda-runtime", default="diagnostic-unresolved")
    parser.add_argument("--model-identity")
    parser.add_argument("--expert-in-dim", type=int, default=4096)
    parser.add_argument("--expert-mid-dim", type=int, default=2048)
    parser.add_argument("--out-dim", type=int, default=4096)
    parser.add_argument("--total-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    if not args.dry_run and not args.output:
        parser.error("--output is required for a benchmark run")
    if args.repeats <= 0 or args.gen_tokens < 0 or args.timeout <= 0:
        parser.error("repeats/timeout must be positive and gen-tokens non-negative")
    if args.min_speedup < 1.0:
        parser.error("--min-speedup must be >= 1.0")
    for name in (
        "sm_count", "l2_bytes", "memory_bus_bits", "global_mem_bytes",
        "expert_in_dim", "expert_mid_dim", "out_dim", "total_experts", "top_k",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    return args


def resolve_from(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    try:
        args.repo_root_path = Path(args.repo_root).resolve()
        args.binary_path = resolve_from(args.repo_root_path, args.binary)
        args.model_path = resolve_from(args.repo_root_path, args.model)
        args.prompt_path = resolve_from(args.repo_root_path, args.prompt_file)
        shapes = parse_shapes(args.shapes, args.total_experts)
        hardware, build, model = keys_from_args(args)
        if args.diagnostic_cache:
            incomplete = cache_key_completeness_errors(hardware, model)
            if incomplete:
                raise AutotuneError(
                    "refusing diagnostic cache with incomplete exact key: "
                    + "; ".join(incomplete)
                )

        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.output:
            output_path = Path(args.output).resolve()
            artifact_base = output_path.parent / f"{output_path.stem}.artifacts"
        else:
            output_path = None
            artifact_base = args.repo_root_path / ".cache" / "flash-iq2-q2-autotune"
        artifact_root = artifact_base / f"run-{stamp}-{os.getpid()}"
        plan = build_plan(args, shapes, artifact_root)

        if args.dry_run:
            report = make_report(
                args, shapes, plan, [], [], [], hardware, build, model,
                hashlib.sha256(b"dry-run").hexdigest(), dry_run=True,
            )
            if output_path:
                atomic_json_dump(output_path, report)
            else:
                json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
                sys.stdout.write("\n")
            return 0

        runs = run_plan(args, plan)
        entries, comparisons, oracle_digest = summarize(
            shapes, runs, args.min_speedup
        )
        report = make_report(
            args, shapes, plan, runs, entries, comparisons,
            hardware, build, model, oracle_digest, dry_run=False,
        )
        assert output_path is not None
        atomic_json_dump(output_path, report)
        if args.diagnostic_cache:
            cache = TuningCache(
                hardware=hardware,
                build=build,
                model=model,
                oracle="pairwise-frontier-logits-f32-bits+argmax+greedy-v1",
                oracle_digest=oracle_digest,
                entries=tuple(entries),
                created_at=report["created_at"],
            )
            write_cache(Path(args.diagnostic_cache), cache)
        failed_runs = sum(bool(run.get("errors")) for run in runs)
        if failed_runs:
            print(
                f"autotune diagnostic completed with {failed_runs} invalid run(s); "
                f"report: {output_path}", file=sys.stderr,
            )
            return 1
        print(f"autotune diagnostic report: {output_path}")
        return 0
    except (AutotuneError, CacheError, OSError, ValueError) as exc:
        print(f"autotune error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
