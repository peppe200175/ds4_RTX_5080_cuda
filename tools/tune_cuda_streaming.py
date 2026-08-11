#!/usr/bin/env python3
"""Explain and optionally benchmark a safe CUDA SSD-streaming profile."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "run_cuda_experiment_wsl.sh"


@dataclass(frozen=True)
class Hardware:
    gpu_name: str
    vram_mib: int
    cpu_threads: int
    ram_gib: float
    model_path: str
    model_gib: float
    numa_nodes: int
    gpu_numa_node: int


@dataclass(frozen=True)
class Profile:
    name: str
    cache_gib: int
    context: int
    prefill_chunk: int
    arena_mb: int
    read_threads: int
    lazy_kv: bool
    rationale: tuple[str, ...]

    def environment(self) -> dict[str, str]:
        return {
            "DS4_CUDA_MMQ": "0",
            "DS4_CUDA_NO_Q8_F16_CACHE": "1",
            "DS4_CUDA_STREAMING_READ_THREADS": str(self.read_threads),
            "DS4_CUDA_STREAMING_SMALL_MISS_PARALLEL": "1",
            "DS4_CUDA_STREAMING_PERSISTENT_READERS": "1",
            "DS4_CUDA_STREAMING_NUMA_AFFINITY": "1",
            "DS4_CUDA_DECODE_CACHE_LRU": "1",
            "DS4_CUDA_DYNAMIC_TIER_PROMOTION": "1",
            "DS4_CUDA_STREAMING_PREFILL_SHARED_OVERLAP": "0",
            "DS4_CUDA_PROMPT_EXPERT_CACHE": "0",
            "DS4_CUDA_PREFIX_EXPERT_CACHE": "0",
            "DS4_CUDA_WEIGHT_ARENA_CHUNK_MB": str(self.arena_mb),
            "DS4_CUDA_LAZY_KV_CACHE": "1" if self.lazy_kv else "0",
            "DS4_CUDA_LAZY_KV_INITIAL_TOKENS": "4096",
        }


def command_output(argv: list[str]) -> str:
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def detect_hardware(model: Path) -> Hardware:
    gpu_rows = command_output([
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]).splitlines()
    if not gpu_rows or "," not in gpu_rows[0]:
        raise RuntimeError("nvidia-smi did not report a CUDA GPU")
    gpu_name, vram = (part.strip() for part in gpu_rows[0].rsplit(",", 1))
    pci_bus = command_output([
        "nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader",
    ]).splitlines()[0].strip().lower()
    pci_short = ("0000:" + pci_bus.split(":", 1)[1]
                 if pci_bus.startswith("00000000:") else pci_bus)
    numa_file = Path("/sys/bus/pci/devices") / pci_short / "numa_node"
    try:
        gpu_numa = int(numa_file.read_text().strip())
    except (OSError, ValueError):
        gpu_numa = -1
    numa_nodes = len(list(Path("/sys/devices/system/node").glob("node[0-9]*")))
    mem_kib = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            mem_kib = int(line.split()[1])
            break
    return Hardware(
        gpu_name=gpu_name,
        vram_mib=int(vram),
        cpu_threads=os.cpu_count() or 1,
        ram_gib=mem_kib / 1024**2,
        model_path=str(model),
        model_gib=model.stat().st_size / 1024**3,
        numa_nodes=max(1, numa_nodes),
        gpu_numa_node=gpu_numa,
    )


def initial_profile(hw: Hardware, context: int) -> Profile:
    if hw.vram_mib <= 18_000:
        cache, arena, chunk = (8 if context >= 65_536 else 10), 256, 1024
        rationale = (
            "16-18 GiB GPUs need lazy KV to leave room for the expert cache.",
            "The measured RTX 5080 profile uses four persistent NVMe readers.",
            "A 256 MiB arena avoids the observed late 1792 MiB reservation failure.",
            "Chunk 1024 won the controlled long-prefill sweep.",
        )
    elif hw.vram_mib <= 32_000:
        cache, arena, chunk = 16, 512, 2048
        rationale = (
            "The larger VRAM tier can retain more routed experts.",
            "A 512 MiB arena reduces allocation bookkeeping while preserving headroom.",
            "Chunk 2048 is the first candidate for a wider GPU and must be A/B tested.",
        )
    else:
        cache, arena, chunk = 24, 1024, 2048
        rationale = (
            "High-memory GPUs should spend spare capacity on routed-expert residency.",
            "The plan reserves capacity for KV, graph scratch, and CUDA workspaces.",
            "The tuner will reject this candidate if allocation or correctness fails.",
        )
    return Profile(
        name="baseline",
        cache_gib=cache,
        context=context,
        prefill_chunk=chunk,
        arena_mb=arena,
        read_threads=min(4, max(1, hw.cpu_threads)),
        lazy_kv=context >= 32_768,
        rationale=rationale,
    )


def candidates(base: Profile) -> list[Profile]:
    variants = [base]
    changes = (
        ("arena512", {"arena_mb": 512}, "Tests fewer/larger arena allocations."),
        ("chunk512", {"prefill_chunk": 512}, "Checks short-prompt launch/occupancy tradeoff."),
        ("cache_minus1", {"cache_gib": max(1, base.cache_gib - 1)}, "Restores VRAM headroom to detect pressure."),
        ("readers2", {"read_threads": 2}, "Checks whether four readers oversaturate this storage path."),
    )
    raw = asdict(base)
    for name, update, reason in changes:
        item = dict(raw)
        item.update(update)
        item["name"] = name
        item["rationale"] = tuple(base.rationale) + (reason,)
        variants.append(Profile(**item))
    return variants


def shell_command(profile: Profile, model: Path, hw: Hardware) -> str:
    continuation = " \\" + "\n"
    env = continuation.join(
        f"{key}={value}" for key, value in profile.environment().items()
    )
    return (
        f"{env}{continuation}"
        f"./ds4 --cuda -m {model} --ssd-streaming "
        f"--ssd-streaming-cache-experts {profile.cache_gib}GB "
        f"--prefill-chunk {profile.prefill_chunk} --ctx {profile.context} --nothink"
    )


def compare_answers(reference: dict, candidate: dict) -> bool:
    fields = ("status", "answer", "prompt_tokens", "completion_tokens", "finish_reason")
    by_id = {row["id"]: row for row in reference.get("results", [])}
    return all(
        row.get("id") in by_id and
        all(row.get(field) == by_id[row["id"]].get(field) for field in fields)
        for row in candidate.get("results", [])
    )


def execute_profile(
    profile: Profile, model: Path, limit: int, port: int, max_temp: int
) -> tuple[Path, dict]:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"autotune_{profile.name}"
    env = os.environ.copy()
    env.update(profile.environment())
    env.update({
        "DS4_MODEL": str(model),
        "DS4_BENCH_LIMIT": str(limit),
        "DS4_BENCH_PORT": str(port),
        "DS4_EXPERIMENT_RUN_ID": run_id,
        "DS4_EXPERIMENT_CACHE_BUDGET": f"{profile.cache_gib}GB",
        "DS4_EXPERIMENT_CONTEXT": str(profile.context),
        "DS4_EXPERIMENT_PREFILL_CHUNK": str(profile.prefill_chunk),
        "DS4_BENCH_MAX_GPU_TEMP": str(max_temp),
    })
    subprocess.run(["bash", str(RUNNER), name], cwd=ROOT, env=env, check=True)
    run_dir = ROOT / "logs" / "cuda_experiments" / f"{run_id}_{name}"
    return run_dir, json.loads((run_dir / "summary.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "tune"), nargs="?", default="plan")
    parser.add_argument("--model", type=Path, default=ROOT / "ds4flash.gguf")
    parser.add_argument("--ctx", type=int, default=131072)
    parser.add_argument("--sentences", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--max-gpu-temp", type=int, default=65)
    parser.add_argument("--output", type=Path, default=ROOT / "ds4_cuda_tuning.json")
    args = parser.parse_args()
    model = args.model.resolve()
    if not model.is_file():
        parser.error(f"model is not readable: {model}")
    try:
        hw = detect_hardware(model)
    except RuntimeError as exc:
        parser.error(str(exc))
    base = initial_profile(hw, args.ctx)
    print(json.dumps({"hardware": asdict(hw), "profile": asdict(base)}, indent=2))
    print("\nRecommended command:\n" + shell_command(base, model, hw))
    if args.action == "plan":
        return 0

    if not RUNNER.is_file() or not command_output(["nvidia-smi", "--version"]):
        parser.error("benchmark runner or nvidia-smi is unavailable")
    results = []
    reference = None
    for index, profile in enumerate(candidates(base)):
        run_dir, summary = execute_profile(
            profile, model, args.sentences, args.port, args.max_gpu_temp
        )
        exact = reference is None or compare_answers(reference, summary)
        if reference is None:
            reference = summary
        row = {
            "profile": asdict(profile),
            "run_dir": str(run_dir),
            "exact": exact,
            "summary": summary.get("summary", {}),
        }
        results.append(row)
        if not exact:
            print(f"Rejected {profile.name}: output differs from baseline", file=sys.stderr)
    eligible = [row for row in results if row["exact"]]
    winner = min(
        eligible,
        key=lambda row: float(row["summary"].get("total_elapsed_s", float("inf"))),
    )
    report = {
        "schema": "ds4-cuda-autotune-v1",
        "hardware": asdict(hw),
        "results": results,
        "winner": winner,
        "command": shell_command(Profile(**winner["profile"]), model, hw),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWinner: {winner['profile']['name']}\n{report['command']}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
