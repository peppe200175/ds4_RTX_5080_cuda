#!/usr/bin/env python3
"""Sequential Flash IQ2/Q2 prefill correctness and performance matrix.

The harness launches ``ds4-bench`` directly; it never controls, restarts, or
flushes an existing server and never drops the operating-system page cache.
Each sample is a declared fresh benchmark process.  Safe and candidate runs
are interleaved (ABBA by default) to reduce ordering bias while preserving the
engine's single-instance lock.  Frontier 1 is an authoritative tokenwise
control, not evidence for the candidate batch path.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shlex
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable


SHORT_CASES = (1, 2, 4, 8, 18, 19, 27, 64)
DIAGNOSTIC_CASES = (127, 128, 129, 256, 257, 258, 270, 1024, 1026, 1027, 1028, 1280, 1281, 1540, 2044)
LONG_CASES = (2048, 2051, 2052, 4099, 4100, 4160, 8192)
ALL_CASES = SHORT_CASES + DIAGNOSTIC_CASES + LONG_CASES

LAYER_MAJOR_FORCE_ENV = {
    "DS4_CUDA_UNSAFE_IQ2_LAYER_MAJOR_PREFILL": "1",
    "DS4_METAL_DISABLE_STREAMING_DECODE_PREFILL": "1",
}

CANDIDATE_ORACLE_ENV = {
    "DS4_CUDA_Q8_BATCH_DECODE_EXACT": "1",
    "DS4_CUDA_ATTENTION_BATCH_DECODE_EXACT": "1",
    "DS4_CUDA_HC_BATCH_DECODE_EXACT": "1",
    "DS4_CUDA_ROUTER_BATCH_DECODE_EXACT": "1",
    "DS4_CUDA_COMPRESSOR_BATCH_DECODE_EXACT": "1",
}

CANDIDATE_BATCH_EXACT_ENV = {
    "DS4_CUDA_MOE_BATCH_DECODE_EXACT": "1",
}

CANDIDATE_ENV = {
    **LAYER_MAJOR_FORCE_ENV,
    **CANDIDATE_ORACLE_ENV,
    **CANDIDATE_BATCH_EXACT_ENV,
}

# These probes force synchronizations or write large tensors.  An inherited
# shell setting must not silently turn a performance run into a diagnostic.
DIAGNOSTIC_ENV = {
    "DS4_METAL_GRAPH_DUMP_PREFIX",
    "DS4_METAL_GRAPH_DUMP_NAME",
    "DS4_METAL_GRAPH_DUMP_LAYER",
    "DS4_METAL_GRAPH_DUMP_POS",
    "DS4_METAL_INDEXER_STAGE_PROFILE",
    "DS4_METAL_GRAPH_LAYER_PROFILE",
    "DS4_METAL_GRAPH_TOKEN_PROFILE",
    "DS4_CUDA_ATTN_OUTPUT_PROFILE",
}


class HarnessError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json_dump(path: Path, value: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_cases(spec: str) -> list[int]:
    selected: set[int] = set()
    for raw in spec.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"all", "*"}:
            selected.update(ALL_CASES)
        elif item == "short":
            selected.update(SHORT_CASES)
        elif item == "long":
            selected.update(LONG_CASES)
        else:
            try:
                value = int(item, 10)
            except ValueError as exc:
                raise HarnessError(f"invalid case selector: {raw!r}") from exc
            if value not in ALL_CASES:
                raise HarnessError(
                    f"unsupported frontier {value}; choose from "
                    + ",".join(str(v) for v in ALL_CASES)
                )
            selected.add(value)
    if not selected:
        raise HarnessError("the case selection is empty")
    return [value for value in ALL_CASES if value in selected]


def mode_schedule(repeats: int, order: str) -> list[str]:
    if repeats <= 0:
        raise HarnessError("repeats must be positive")
    if order == "safe-first":
        return ["safe"] * repeats + ["candidate"] * repeats
    if order == "candidate-first":
        return ["candidate"] * repeats + ["safe"] * repeats
    if order != "abba":
        raise HarnessError(f"unsupported order: {order}")

    result: list[str] = []
    counts = {"safe": 0, "candidate": 0}
    pattern = ("safe", "candidate", "candidate", "safe")
    while counts["safe"] < repeats or counts["candidate"] < repeats:
        for mode in pattern:
            if counts[mode] >= repeats:
                continue
            result.append(mode)
            counts[mode] += 1
    return result


def parse_key_value(items: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key or "\x00" in key or "\x00" in value:
            raise HarnessError(f"expected KEY=VALUE, got {item!r}")
        result[key] = value
    return result


def resolve_from(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def candidate_degenerate_tokenwise(frontier: int, mode: str) -> bool:
    return mode == "candidate" and frontier == 1


def has_chunked_single_token_tail(frontier: int, prefill_chunk: int) -> bool:
    return (
        frontier > prefill_chunk
        and prefill_chunk > 0
        and frontier % prefill_chunk == 1
    )


def build_environment(
    args: argparse.Namespace,
    mode: str,
    frontier: int,
) -> tuple[dict[str, str], dict[str, str]]:
    env = os.environ.copy()
    explicit = parse_key_value(args.env)
    for key in DIAGNOSTIC_ENV:
        if key not in explicit:
            env.pop(key, None)

    cuda_home = str(Path(args.cuda_home).resolve())
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = cuda_home + "/bin:" + env.get("PATH", "")
    old_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = cuda_home + "/lib64" + (
        ":" + old_library_path if old_library_path else ""
    )
    env["DS4_CUDA_MMQ"] = "0"
    env["DS4_CUDA_NO_Q8_F16_CACHE"] = "1"
    env.update(explicit)

    # The harness owns these switches.  Safe cannot inherit an unsafe flag.
    # Frontier 1 has no supported layer-major selected batch, so its candidate
    # deliberately retains the harmless row-wise exact-value oracles while
    # removing switches that only apply to multi-token layer-major execution.
    for key in CANDIDATE_ENV:
        env.pop(key, None)
    if mode == "candidate":
        env.update(CANDIDATE_ENV)
        if candidate_degenerate_tokenwise(frontier, mode):
            for key in (*LAYER_MAJOR_FORCE_ENV, *CANDIDATE_BATCH_EXACT_ENV):
                env.pop(key, None)
    elif mode != "safe":
        raise HarnessError(f"unknown mode: {mode}")

    recorded_keys = {
        "CUDA_HOME",
        "DS4_CUDA_MMQ",
        "DS4_CUDA_NO_Q8_F16_CACHE",
        *CANDIDATE_ENV.keys(),
        *explicit.keys(),
    }
    recorded = {key: env[key] for key in sorted(recorded_keys) if key in env}
    return env, recorded


def command_for(
    args: argparse.Namespace,
    frontier: int,
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
        str(args.prefill_chunk),
        "--prompt-file",
        str(args.prompt_path),
        "--ctx-start",
        str(frontier),
        "--ctx-max",
        str(frontier),
        "--ctx-alloc",
        str(frontier + args.gen_tokens + 1),
        "--step-incr",
        "1",
        "--gen-tokens",
        str(args.gen_tokens),
        "--csv",
        str(csv_path),
        "--dump-frontier-logits-dir",
        str(artifact_dir),
    ]
    if args.threads is not None:
        command += ["--threads", str(args.threads)]
    if args.gpu_vram:
        command += ["--gpu-vram", args.gpu_vram]
    if args.gpu_devices:
        command += ["--gpu-devices", args.gpu_devices]
    if args.warm_weights:
        command.append("--warm-weights")
    command.extend(args.bench_arg)
    return command


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def read_single_csv(path: Path, frontier: int) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        return {}, [f"cannot read benchmark CSV: {exc}"]
    if len(rows) != 1:
        return {}, [f"expected one CSV row, found {len(rows)}"]
    row = rows[0]
    converted: dict[str, Any] = {}
    integer_fields = {"ctx_tokens", "prefill_tokens", "gen_tokens", "gen_steady_tokens", "kvcache_bytes"}
    for key, raw in row.items():
        try:
            converted[key] = int(raw) if key in integer_fields else float(raw)
        except (TypeError, ValueError):
            errors.append(f"CSV field {key} is not numeric: {raw!r}")
    if converted.get("ctx_tokens") != frontier:
        errors.append(
            f"CSV ctx_tokens={converted.get('ctx_tokens')!r}, expected {frontier}"
        )
    if converted.get("prefill_tokens") != frontier:
        errors.append(
            f"CSV prefill_tokens={converted.get('prefill_tokens')!r}, expected {frontier}"
        )
    return converted, errors


def float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def float32_ordered(bits: int) -> int:
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000


def logits_digest(values: list[Any]) -> tuple[str, int]:
    digest = hashlib.sha256()
    nonfinite = 0
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            digest.update(b"N")
            nonfinite += 1
        else:
            digest.update(struct.pack("<f", float(value)))
    return digest.hexdigest(), nonfinite


def inspect_artifacts(
    directory: Path,
    csv_path: Path,
    frontier: int,
    gen_tokens: int,
) -> tuple[dict[str, Any], dict[str, float], list[str]]:
    errors: list[str] = []
    csv_row, csv_errors = read_single_csv(csv_path, frontier)
    errors.extend(csv_errors)

    logits_path = directory / f"frontier_{frontier:06d}.logits.json"
    tokens_path = directory / f"frontier_{frontier:06d}.tokens.json"
    metadata: dict[str, Any] = {
        "csv_path": str(csv_path),
        "logits_path": str(logits_path),
        "tokens_path": str(tokens_path),
    }

    try:
        with logits_path.open("r", encoding="utf-8") as handle:
            logits_doc = json.load(handle)
    except (OSError, ValueError) as exc:
        logits_doc = {}
        errors.append(f"cannot read frontier logits: {exc}")
    values = logits_doc.get("logits")
    if isinstance(values, list):
        digest, nonfinite = logits_digest(values)
        metadata.update(
            {
                "prompt_tokens": logits_doc.get("prompt_tokens"),
                "frontier_tokens": logits_doc.get("frontier_tokens"),
                "backend": logits_doc.get("backend"),
                "quant_bits": logits_doc.get("quant_bits"),
                "vocab": logits_doc.get("vocab"),
                "argmax_id": logits_doc.get("argmax_id"),
                "argmax_logit": logits_doc.get("argmax_logit"),
                "logits_count": len(values),
                "logits_sha256_f32": digest,
                "nonfinite_logits": nonfinite,
            }
        )
        if logits_doc.get("prompt_tokens") != frontier:
            errors.append(
                f"prompt_tokens={logits_doc.get('prompt_tokens')!r}, expected {frontier}"
            )
        if logits_doc.get("frontier_tokens") != frontier:
            errors.append("frontier logits metadata does not match the requested case")
        if logits_doc.get("vocab") != len(values):
            errors.append("vocab size does not match the complete logits array")
        if nonfinite:
            errors.append(f"frontier logits contain {nonfinite} non-finite value(s)")
        if values and nonfinite == 0:
            actual_argmax = max(range(len(values)), key=lambda i: float(values[i]))
            metadata["computed_argmax_id"] = actual_argmax
            if logits_doc.get("argmax_id") != actual_argmax:
                errors.append(
                    f"argmax_id={logits_doc.get('argmax_id')!r}, computed {actual_argmax}"
                )
    else:
        errors.append("frontier logits JSON has no complete logits array")

    try:
        with tokens_path.open("r", encoding="utf-8") as handle:
            tokens_doc = json.load(handle)
    except (OSError, ValueError) as exc:
        tokens_doc = {}
        errors.append(
            "cannot read greedy token sidecar (rebuild ds4-bench with the "
            f"machine-readable hook): {exc}"
        )
    token_ids = tokens_doc.get("token_ids")
    if isinstance(token_ids, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in token_ids):
        metadata["token_ids"] = token_ids
        metadata["generated_tokens"] = tokens_doc.get("generated_tokens")
        metadata["requested_tokens"] = tokens_doc.get("requested_tokens")
        if tokens_doc.get("frontier_tokens") != frontier:
            errors.append("greedy token sidecar frontier does not match the requested case")
        if tokens_doc.get("requested_tokens") != gen_tokens:
            errors.append("greedy token sidecar requested_tokens is inconsistent")
        if tokens_doc.get("generated_tokens") != len(token_ids):
            errors.append("greedy token sidecar length is inconsistent")
        if len(token_ids) != gen_tokens:
            errors.append(
                f"generated {len(token_ids)} greedy tokens, expected {gen_tokens}"
            )
    else:
        errors.append("greedy token sidecar has no valid token_ids array")

    prefill_tps = number(csv_row.get("prefill_tps"))
    generated = number(csv_row.get("gen_tokens"))
    gen_tps = number(csv_row.get("gen_tps"))
    metrics = {
        "prefill_tokens": float(csv_row.get("prefill_tokens", 0)),
        "prefill_tps": prefill_tps or 0.0,
        "prefill_s": frontier / prefill_tps if prefill_tps and prefill_tps > 0 else 0.0,
        "gen_tokens": generated or 0.0,
        "gen_tps": gen_tps or 0.0,
        "gen_s": generated / gen_tps if generated and gen_tps and gen_tps > 0 else 0.0,
        "gen_first_ms": number(csv_row.get("gen_first_ms")) or 0.0,
        "gen_steady_tokens": number(csv_row.get("gen_steady_tokens")) or 0.0,
        "gen_steady_tps": number(csv_row.get("gen_steady_tps")) or 0.0,
        "kvcache_bytes": number(csv_row.get("kvcache_bytes")) or 0.0,
    }
    return metadata, metrics, errors


def compare_logits_files(safe_path: Path, candidate_path: Path) -> dict[str, Any]:
    try:
        with safe_path.open("r", encoding="utf-8") as handle:
            safe_doc = json.load(handle)
        with candidate_path.open("r", encoding="utf-8") as handle:
            candidate_doc = json.load(handle)
    except (OSError, ValueError) as exc:
        return {"exact": False, "error": str(exc)}
    safe = safe_doc.get("logits")
    candidate = candidate_doc.get("logits")
    if not isinstance(safe, list) or not isinstance(candidate, list):
        return {"exact": False, "error": "one side has no complete logits array"}
    if len(safe) != len(candidate):
        return {
            "exact": False,
            "error": "logits lengths differ",
            "safe_count": len(safe),
            "candidate_count": len(candidate),
        }

    different = 0
    nonfinite = 0
    first: dict[str, Any] | None = None
    max_abs = 0.0
    max_ulp = 0
    for index, (left, right) in enumerate(zip(safe, candidate)):
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, (int, float))
            or not isinstance(right, (int, float))
            or not math.isfinite(float(left))
            or not math.isfinite(float(right))
        ):
            nonfinite += 1
            if first is None:
                first = {"index": index, "safe": left, "candidate": right}
            continue
        left_bits = float32_bits(float(left))
        right_bits = float32_bits(float(right))
        if left_bits == right_bits:
            continue
        different += 1
        absolute = abs(float(left) - float(right))
        ulp = abs(float32_ordered(left_bits) - float32_ordered(right_bits))
        if absolute > max_abs:
            max_abs = absolute
        if ulp > max_ulp:
            max_ulp = ulp
        if first is None:
            first = {
                "index": index,
                "safe": left,
                "candidate": right,
                "safe_bits": f"0x{left_bits:08x}",
                "candidate_bits": f"0x{right_bits:08x}",
                "abs": absolute,
                "ulp": ulp,
            }
    return {
        "exact": different == 0 and nonfinite == 0,
        "count": len(safe),
        "different": different,
        "nonfinite_pairs": nonfinite,
        "first_difference": first,
        "max_abs": max_abs,
        "max_ulp": max_ulp,
    }


def compare_pair(frontier: int, pair_index: int, safe: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if safe.get("errors"):
        failures.append("safe run is invalid")
    if candidate.get("errors"):
        failures.append("candidate run is invalid")
    safe_art = safe.get("artifacts", {})
    candidate_art = candidate.get("artifacts", {})

    for label, artifact in (("safe", safe_art), ("candidate", candidate_art)):
        if artifact.get("prompt_tokens") != frontier:
            failures.append(f"{label} prompt_tokens does not equal {frontier}")
        if artifact.get("quant_bits") != 2:
            failures.append(f"{label} quant_bits is not 2")
        if artifact.get("backend") != "cuda":
            failures.append(f"{label} backend is not cuda")
        if artifact.get("nonfinite_logits") != 0:
            failures.append(f"{label} logits are not all finite")

    logits = compare_logits_files(
        Path(safe_art.get("logits_path", "")),
        Path(candidate_art.get("logits_path", "")),
    )
    if not logits.get("exact"):
        failures.append("complete logits are not bit-identical")
    safe_argmax = safe_art.get("argmax_id")
    candidate_argmax = candidate_art.get("argmax_id")
    argmax_match = (
        isinstance(safe_argmax, int)
        and not isinstance(safe_argmax, bool)
        and isinstance(candidate_argmax, int)
        and not isinstance(candidate_argmax, bool)
        and safe_argmax == candidate_argmax
    )
    if not argmax_match:
        failures.append("argmax ids differ")
    safe_tokens = safe_art.get("token_ids")
    candidate_tokens = candidate_art.get("token_ids")
    tokens_match = (
        isinstance(safe_tokens, list)
        and isinstance(candidate_tokens, list)
        and safe_tokens == candidate_tokens
    )
    if not tokens_match:
        failures.append("greedy token ids differ")

    return {
        "frontier": frontier,
        "pair_index": pair_index,
        "safe_run_id": safe.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "status": "PASS" if not failures else "FAIL",
        "candidate_degenerate_tokenwise": frontier == 1,
        "candidate_batch_path_exercised": frontier >= 2,
        "prompt_tokens_match": (
            safe_art.get("prompt_tokens")
            == candidate_art.get("prompt_tokens")
            == frontier
        ),
        "argmax_match": argmax_match,
        "token_ids_match": tokens_match,
        "logits": logits,
        "failures": failures,
    }


def percentile(values: list[float], quantile: float) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def metric_summary(values: list[float]) -> dict[str, float | int | None]:
    values = [value for value in values if math.isfinite(value)]
    return {
        "samples": len(values),
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def performance_for_case(
    frontier: int, runs: list[dict[str, Any]]
) -> dict[str, Any]:
    by_mode = {
        mode: [run for run in runs if run.get("mode") == mode and not run.get("errors")]
        for mode in ("safe", "candidate")
    }
    fields = (
        "prefill_tps",
        "prefill_s",
        "gen_tps",
        "gen_first_ms",
        "gen_steady_tps",
        "disk_read_bytes_estimate",
        "wall_s",
    )
    metrics: dict[str, Any] = {}
    for field in fields:
        safe_values = [
            float(run["metrics"].get(field, 0.0)) for run in by_mode["safe"]
        ]
        candidate_values = [
            float(run["metrics"].get(field, 0.0))
            for run in by_mode["candidate"]
        ]
        safe_summary = metric_summary(safe_values)
        candidate_summary = metric_summary(candidate_values)
        safe_median = safe_summary["median"]
        candidate_median = candidate_summary["median"]
        higher_is_better = field in {"prefill_tps", "gen_tps", "gen_steady_tps"}
        regression = None
        ratio = None
        if (
            isinstance(safe_median, (int, float))
            and isinstance(candidate_median, (int, float))
            and safe_median > 0
        ):
            ratio = candidate_median / safe_median
            regression = (
                (safe_median - candidate_median) / safe_median * 100.0
                if higher_is_better
                else (candidate_median - safe_median) / safe_median * 100.0
            )
        metrics[field] = {
            "safe": safe_summary,
            "candidate": candidate_summary,
            "candidate_over_safe": ratio,
            "regression_pct": regression,
            "higher_is_better": higher_is_better,
        }
    return {
        "frontier": frontier,
        "candidate_degenerate_tokenwise": frontier == 1,
        "candidate_batch_path_exercised": frontier >= 2,
        "metrics": metrics,
    }


def run_sample(
    args: argparse.Namespace,
    frontier: int,
    mode: str,
    mode_ordinal: int,
    sequence: int,
    root_artifacts: Path,
) -> dict[str, Any]:
    run_id = f"n{frontier:05d}-{mode}-r{mode_ordinal:02d}-s{sequence:04d}"
    directory = root_artifacts / run_id
    directory.mkdir(parents=True, exist_ok=False)
    csv_path = directory / "metrics.csv"
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    command = command_for(args, frontier, csv_path, directory)
    env, recorded_env = build_environment(args, mode, frontier)
    record: dict[str, Any] = {
        "run_id": run_id,
        "sequence": sequence,
        "frontier": frontier,
        "mode": mode,
        "candidate_degenerate_tokenwise": candidate_degenerate_tokenwise(
            frontier, mode
        ),
        "candidate_chunked_single_token_tail": (
            mode == "candidate"
            and has_chunked_single_token_tail(frontier, args.prefill_chunk)
        ),
        "mode_ordinal": mode_ordinal,
        "command": command,
        "command_display": shlex.join(command),
        "environment": recorded_env,
        "artifact_dir": str(directory),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "started_at": utc_now(),
        "errors": [],
    }

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                command,
                cwd=args.repo_root_path,
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=args.timeout,
                check=False,
            )
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        returncode = 124
        record["errors"].append(f"benchmark exceeded timeout {args.timeout}s")
    except OSError as exc:
        returncode = 127
        record["errors"].append(f"cannot launch ds4-bench: {exc}")
    elapsed = time.monotonic() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    record["returncode"] = returncode
    record["finished_at"] = utc_now()
    if returncode != 0:
        record["errors"].append(f"ds4-bench exited with status {returncode}")

    artifacts, metrics, artifact_errors = inspect_artifacts(
        directory, csv_path, frontier, args.gen_tokens
    )
    record["artifacts"] = artifacts
    record["errors"].extend(artifact_errors)
    metrics.update(
        {
            "wall_s": elapsed,
            # Linux ru_inblock is counted in 512-byte blocks.  This is physical
            # process I/O, not the engine's logical expert-read byte counter.
            "disk_read_blocks": max(0.0, after.ru_inblock - before.ru_inblock),
            "disk_read_bytes_estimate": max(
                0.0, (after.ru_inblock - before.ru_inblock) * 512.0
            ),
            "disk_write_blocks": max(0.0, after.ru_oublock - before.ru_oublock),
            "major_faults": max(0.0, after.ru_majflt - before.ru_majflt),
            "minor_faults": max(0.0, after.ru_minflt - before.ru_minflt),
        }
    )
    record["metrics"] = metrics
    return record


def dry_run_plan(args: argparse.Namespace, cases: list[int], artifact_root: Path) -> dict[str, Any]:
    schedule = mode_schedule(args.repeats, args.order)
    commands: list[dict[str, Any]] = []
    sequence = 0
    for frontier in cases:
        ordinals = {"safe": 0, "candidate": 0}
        for mode in schedule:
            ordinal = ordinals[mode]
            ordinals[mode] += 1
            run_id = f"n{frontier:05d}-{mode}-r{ordinal:02d}-s{sequence:04d}"
            directory = artifact_root / run_id
            csv_path = directory / "metrics.csv"
            _, recorded_env = build_environment(args, mode, frontier)
            commands.append(
                {
                    "sequence": sequence,
                    "frontier": frontier,
                    "mode": mode,
                    "candidate_degenerate_tokenwise": candidate_degenerate_tokenwise(
                        frontier, mode
                    ),
                    "candidate_chunked_single_token_tail": (
                        mode == "candidate"
                        and has_chunked_single_token_tail(
                            frontier, args.prefill_chunk
                        )
                    ),
                    "run_id": run_id,
                    "command": command_for(args, frontier, csv_path, directory),
                    "environment": recorded_env,
                }
            )
            sequence += 1
    return {
        "schema_version": 1,
        "status": "DRY_RUN",
        "created_at": utc_now(),
        "cases": cases,
        "candidate_degenerate_tokenwise": 1 in cases,
        "candidate_degenerate_tokenwise_frontiers": [
            frontier for frontier in cases if frontier == 1
        ],
        "candidate_chunked_single_token_tail_frontiers": [
            frontier
            for frontier in cases
            if has_chunked_single_token_tail(frontier, args.prefill_chunk)
        ],
        "candidate_batch_evidence_frontiers": [
            frontier for frontier in cases if frontier >= 2
        ],
        "schedule_per_case": schedule,
        "commands": commands,
        "side_effect_contract": {
            "strictly_sequential": True,
            "process_per_sample": True,
            "controls_server": False,
            "restarts_server": False,
            "flushes_page_cache": False,
            "runs_gpu_in_dry_run": False,
        },
    }


def self_test() -> None:
    assert parse_cases("short") == list(SHORT_CASES)
    assert parse_cases("2052,1,long") == [1, *LONG_CASES]
    assert mode_schedule(2, "abba") == ["safe", "candidate", "candidate", "safe"]
    assert mode_schedule(3, "abba") == [
        "safe",
        "candidate",
        "candidate",
        "safe",
        "safe",
        "candidate",
    ]
    assert has_chunked_single_token_tail(257, 32)
    assert has_chunked_single_token_tail(129, 32)
    assert not has_chunked_single_token_tail(256, 32)
    assert not has_chunked_single_token_tail(1, 32)

    harness_args = argparse.Namespace(
        env=["DS4_CUDA_UNSAFE_IQ2_LAYER_MAJOR_PREFILL=1"],
        cuda_home="/tmp/cuda",
        repeats=1,
        order="safe-first",
        binary_path=Path("/tmp/ds4-bench"),
        model_path=Path("/tmp/model.gguf"),
        expert_cache="10GB",
        prefill_chunk=1024,
        prompt_path=Path("/tmp/prompt.txt"),
        gen_tokens=1,
        threads=None,
        gpu_vram=None,
        gpu_devices=None,
        warm_weights=False,
        bench_arg=[],
    )
    safe_env, _ = build_environment(harness_args, "safe", 1)
    candidate_n1_env, _ = build_environment(harness_args, "candidate", 1)
    candidate_n2_env, _ = build_environment(harness_args, "candidate", 2)
    assert all(key not in safe_env for key in CANDIDATE_ENV)
    assert all(key not in candidate_n1_env for key in LAYER_MAJOR_FORCE_ENV)
    assert all(
        key not in candidate_n1_env for key in CANDIDATE_BATCH_EXACT_ENV
    )
    assert all(
        candidate_n1_env.get(key) == value
        for key, value in CANDIDATE_ORACLE_ENV.items()
    )
    assert all(
        candidate_n2_env.get(key) == value
        for key, value in CANDIDATE_ENV.items()
    )

    dry_plan = dry_run_plan(harness_args, [1, 2], Path("/tmp/artifacts"))
    dry_candidates = {
        command["frontier"]: command
        for command in dry_plan["commands"]
        if command["mode"] == "candidate"
    }
    assert dry_candidates[1]["candidate_degenerate_tokenwise"] is True
    assert dry_candidates[2]["candidate_degenerate_tokenwise"] is False
    assert all(
        key not in dry_candidates[1]["environment"]
        for key in (*LAYER_MAJOR_FORCE_ENV, *CANDIDATE_BATCH_EXACT_ENV)
    )
    assert all(
        dry_candidates[2]["environment"].get(key) == value
        for key, value in {
            **LAYER_MAJOR_FORCE_ENV,
            **CANDIDATE_BATCH_EXACT_ENV,
        }.items()
    )

    tail_args = argparse.Namespace(**vars(harness_args))
    tail_args.prefill_chunk = 32
    tail_plan = dry_run_plan(tail_args, [256, 257], Path("/tmp/tail-artifacts"))
    tail_candidates = {
        command["frontier"]: command
        for command in tail_plan["commands"]
        if command["mode"] == "candidate"
    }
    assert tail_candidates[256]["candidate_chunked_single_token_tail"] is False
    assert tail_candidates[257]["candidate_chunked_single_token_tail"] is True
    assert tail_plan["candidate_chunked_single_token_tail_frontiers"] == [257]

    with tempfile.TemporaryDirectory(prefix="ds4-prefill-selftest-") as temporary:
        root = Path(temporary)
        safe_path = root / "safe.json"
        candidate_path = root / "candidate.json"
        document = {
            "prompt_tokens": 4,
            "frontier_tokens": 4,
            "vocab": 4,
            "argmax_id": 3,
            "logits": [-1.0, 0.0, 1.0, 2.0],
        }
        atomic_json_dump(safe_path, document)
        atomic_json_dump(candidate_path, document)
        exact = compare_logits_files(safe_path, candidate_path)
        assert exact["exact"] and exact["different"] == 0

        n1_run = {
            "errors": [],
            "artifacts": {
                "prompt_tokens": 1,
                "quant_bits": 2,
                "backend": "cuda",
                "nonfinite_logits": 0,
                "logits_path": str(safe_path),
                "argmax_id": 3,
                "token_ids": [3],
            },
        }
        n1_candidate = {
            **n1_run,
            "artifacts": {
                **n1_run["artifacts"],
                "logits_path": str(candidate_path),
            },
        }
        n1_comparison = compare_pair(1, 0, n1_run, n1_candidate)
        assert n1_comparison["status"] == "PASS"
        assert n1_comparison["candidate_degenerate_tokenwise"] is True
        assert n1_comparison["candidate_batch_path_exercised"] is False

        changed = dict(document)
        changed["logits"] = list(document["logits"])
        changed["logits"][2] = struct.unpack(
            "<f", struct.pack("<I", float32_bits(1.0) + 1)
        )[0]
        atomic_json_dump(candidate_path, changed)
        mismatch = compare_logits_files(safe_path, candidate_path)
        assert not mismatch["exact"] and mismatch["different"] == 1
        assert mismatch["first_difference"]["index"] == 2

        output = root / "nested" / "result.json"
        atomic_json_dump(output, {"status": "PASS"})
        assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"
    print("bench_flash_prefill_exact self-test: PASS")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run a sequential safe/candidate Flash IQ2/Q2 prefill matrix via ds4-bench. "
            "The harness does not stop or restart a server and does not flush caches."
        )
    )
    parser.add_argument("--repo-root", default=str(default_root))
    parser.add_argument("--binary", default="ds4-bench")
    parser.add_argument("--model", default="ds4flash.gguf")
    parser.add_argument(
        "--prompt-file", default="tests/long_context_story_prompt.txt"
    )
    parser.add_argument(
        "--cases",
        default="all",
        help=(
            "comma-separated frontiers, or short,long,all; frontier 1 is a "
            "tokenwise control and is not batch-path evidence"
        ),
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--order",
        choices=("abba", "safe-first", "candidate-first"),
        default="abba",
    )
    parser.add_argument("--gen-tokens", type=int, default=8)
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--expert-cache", default="10GB")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--gpu-vram")
    parser.add_argument("--gpu-devices")
    parser.add_argument("--warm-weights", action="store_true")
    parser.add_argument(
        "--cuda-home",
        default=os.environ.get(
            "CUDA_HOME", "/home/peppe200175/.local/cuda-13.3.1"
        ),
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--bench-arg",
        action="append",
        default=[],
        help="extra ds4-bench argument; repeat for multiple arguments",
    )
    parser.add_argument("--output")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--prefill-regression-limit-pct", type=float, default=5.0)
    parser.add_argument("--gen-regression-limit-pct", type=float, default=3.0)
    parser.add_argument("--disk-regression-limit-pct", type=float)
    parser.add_argument("--min-prefill-geomean-speedup", type=float, default=1.0)
    parser.add_argument("--report-only-performance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    if not args.output and not args.dry_run:
        parser.error("--output is required unless --dry-run or --self-test is used")
    if args.repeats <= 0 or args.gen_tokens < 0 or args.prefill_chunk <= 0:
        parser.error("repeats and prefill-chunk must be positive; gen-tokens cannot be negative")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    for name in (
        "prefill_regression_limit_pct",
        "gen_regression_limit_pct",
        "min_prefill_geomean_speedup",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    if args.disk_regression_limit_pct is not None and args.disk_regression_limit_pct < 0:
        parser.error("--disk-regression-limit-pct cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    try:
        cases = parse_cases(args.cases)
        args.repo_root_path = Path(args.repo_root).resolve()
        args.binary_path = resolve_from(args.repo_root_path, args.binary)
        args.model_path = resolve_from(args.repo_root_path, args.model)
        args.prompt_path = resolve_from(args.repo_root_path, args.prompt_file)
        output_path = Path(args.output).resolve() if args.output else None
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.artifact_dir:
            artifact_base = Path(args.artifact_dir).resolve()
        elif output_path:
            artifact_base = output_path.parent / f"{output_path.stem}.artifacts"
        else:
            artifact_base = args.repo_root_path / ".cache" / "flash-prefill-exact"
        artifact_root = artifact_base / f"run-{stamp}-{os.getpid()}"

        if args.dry_run:
            plan = dry_run_plan(args, cases, artifact_root)
            if output_path:
                atomic_json_dump(output_path, plan)
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0

        for path, label in (
            (args.repo_root_path, "repository root"),
            (args.binary_path, "ds4-bench binary"),
            (args.model_path, "model"),
            (args.prompt_path, "prompt fixture"),
        ):
            if not path.exists():
                raise HarnessError(f"{label} does not exist: {path}")
        if not os.access(args.binary_path, os.X_OK):
            raise HarnessError(f"ds4-bench is not executable: {args.binary_path}")

        artifact_root.mkdir(parents=True, exist_ok=False)
        schedule = mode_schedule(args.repeats, args.order)
        started_at = utc_now()
        runs: list[dict[str, Any]] = []
        sequence = 0
        for frontier in cases:
            ordinals = {"safe": 0, "candidate": 0}
            for mode in schedule:
                ordinal = ordinals[mode]
                ordinals[mode] += 1
                print(
                    f"[{sequence + 1}/{len(cases) * len(schedule)}] "
                    f"frontier={frontier} mode={mode} repeat={ordinal}",
                    flush=True,
                )
                runs.append(
                    run_sample(
                        args,
                        frontier,
                        mode,
                        ordinal,
                        sequence,
                        artifact_root,
                    )
                )
                sequence += 1

        comparisons: list[dict[str, Any]] = []
        performance: list[dict[str, Any]] = []
        failures: list[str] = []
        warnings: list[str] = []
        prefill_batch_speedups: list[float] = []
        for frontier in cases:
            case_runs = [run for run in runs if run["frontier"] == frontier]
            safe_runs = sorted(
                (run for run in case_runs if run["mode"] == "safe"),
                key=lambda run: run["mode_ordinal"],
            )
            candidate_runs = sorted(
                (run for run in case_runs if run["mode"] == "candidate"),
                key=lambda run: run["mode_ordinal"],
            )
            for pair_index, (safe, candidate) in enumerate(zip(safe_runs, candidate_runs)):
                comparison = compare_pair(frontier, pair_index, safe, candidate)
                comparisons.append(comparison)
                if comparison["status"] != "PASS":
                    failures.append(
                        f"frontier {frontier} pair {pair_index}: correctness mismatch"
                    )

            perf = performance_for_case(frontier, case_runs)
            performance.append(perf)
            prefill = perf["metrics"]["prefill_tps"]
            speedup = prefill.get("candidate_over_safe")
            regression = prefill.get("regression_pct")
            if (
                frontier >= 2
                and isinstance(speedup, (int, float))
                and speedup > 0
            ):
                prefill_batch_speedups.append(float(speedup))
            if (
                isinstance(regression, (int, float))
                and regression > args.prefill_regression_limit_pct
            ):
                message = (
                    f"frontier {frontier}: median prefill throughput regressed "
                    f"by {regression:.2f}%"
                )
                (warnings if args.report_only_performance else failures).append(message)

            for field in ("gen_tps", "gen_steady_tps", "gen_first_ms"):
                gen_regression = perf["metrics"][field].get("regression_pct")
                if (
                    isinstance(gen_regression, (int, float))
                    and gen_regression > args.gen_regression_limit_pct
                ):
                    message = (
                        f"frontier {frontier}: {field} regressed by "
                        f"{gen_regression:.2f}%"
                    )
                    (warnings if args.report_only_performance else failures).append(message)

            disk_regression = perf["metrics"]["disk_read_bytes_estimate"].get(
                "regression_pct"
            )
            if (
                args.disk_regression_limit_pct is not None
                and isinstance(disk_regression, (int, float))
                and disk_regression > args.disk_regression_limit_pct
            ):
                message = (
                    f"frontier {frontier}: OS read-block estimate regressed by "
                    f"{disk_regression:.2f}%"
                )
                (warnings if args.report_only_performance else failures).append(message)

        geomean_speedup = None
        if prefill_batch_speedups:
            geomean_speedup = math.exp(
                sum(math.log(value) for value in prefill_batch_speedups)
                / len(prefill_batch_speedups)
            )
            if geomean_speedup < args.min_prefill_geomean_speedup:
                message = (
                    f"prefill geomean speedup {geomean_speedup:.4f} is below "
                    f"{args.min_prefill_geomean_speedup:.4f}"
                )
                (warnings if args.report_only_performance else failures).append(message)

        for run in runs:
            if run.get("errors"):
                failures.append(f"{run['run_id']}: invalid benchmark sample")
            disk = run.get("metrics", {}).get("disk_read_bytes_estimate")
            if disk == 0:
                warnings.append(
                    f"{run['run_id']}: OS read-block estimate is zero; data may have been page-cached"
                )

        result = {
            "schema_version": 1,
            "status": "PASS" if not failures else "FAIL",
            "candidate_degenerate_tokenwise": 1 in cases,
            "started_at": started_at,
            "finished_at": utc_now(),
            "configuration": {
                "repo_root": str(args.repo_root_path),
                "binary": str(args.binary_path),
                "model": str(args.model_path),
                "prompt_file": str(args.prompt_path),
                "cases": cases,
                "repeats_per_mode": args.repeats,
                "order": args.order,
                "schedule_per_case": schedule,
                "gen_tokens": args.gen_tokens,
                "prefill_chunk": args.prefill_chunk,
                "expert_cache": args.expert_cache,
                "candidate_environment": CANDIDATE_ENV,
                "candidate_environment_frontier_1": CANDIDATE_ORACLE_ENV,
                "candidate_degenerate_tokenwise": 1 in cases,
                "candidate_degenerate_tokenwise_frontiers": [
                    frontier for frontier in cases if frontier == 1
                ],
                "candidate_batch_evidence_frontiers": [
                    frontier for frontier in cases if frontier >= 2
                ],
                "candidate_chunked_single_token_tail_frontiers": [
                    frontier
                    for frontier in cases
                    if has_chunked_single_token_tail(frontier,
                                                     args.prefill_chunk)
                ],
                "performance_limits": {
                    "prefill_regression_pct": args.prefill_regression_limit_pct,
                    "generation_regression_pct": args.gen_regression_limit_pct,
                    "disk_regression_pct": args.disk_regression_limit_pct,
                    "min_prefill_geomean_speedup": args.min_prefill_geomean_speedup,
                    "report_only": args.report_only_performance,
                },
                "side_effect_contract": {
                    "strictly_sequential": True,
                    "process_per_sample": True,
                    "controls_server": False,
                    "restarts_server": False,
                    "flushes_page_cache": False,
                },
                "disk_metric": (
                    "Linux child ru_inblock * 512; physical-I/O estimate, not logical expert bytes"
                ),
            },
            "artifact_root": str(artifact_root),
            "runs": runs,
            "comparisons": comparisons,
            "performance": performance,
            "summary": {
                "samples": len(runs),
                "comparison_pairs": len(comparisons),
                "correctness_pairs_passed": sum(
                    comparison["status"] == "PASS" for comparison in comparisons
                ),
                "prefill_geomean_speedup": geomean_speedup,
                "prefill_geomean_batch_frontiers": [
                    frontier for frontier in cases if frontier >= 2
                ],
                "failures": len(failures),
                "warnings": len(warnings),
            },
            "failures": sorted(set(failures)),
            "warnings": sorted(set(warnings)),
        }
        assert output_path is not None
        atomic_json_dump(output_path, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": str(output_path),
                    "artifact_root": str(artifact_root),
                    "samples": len(runs),
                    "failures": len(result["failures"]),
                    "warnings": len(result["warnings"]),
                },
                sort_keys=True,
            )
        )
        return 0 if result["status"] == "PASS" else 1
    except HarnessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
