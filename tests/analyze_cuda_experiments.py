#!/usr/bin/env python3
"""Aggregate DS4 CUDA A/B runs into reusable JSON, CSV, and Markdown reports."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from summarize_cuda_experiment import correctness, load_config, safe_div, summarize


IGNORED_CONFIG_KEYS = {"experiment", "run_id", "DS4_LOCK_FILE", "start_gpu_temp_c"}


def thermal_summary(path: Path) -> dict[str, float | int | None]:
    if not path.exists():
        return {"start_c": None, "average_c": None, "maximum_c": None,
                "average_power_w": None, "maximum_vram_mib": None}
    text = path.read_text(encoding="utf-8", errors="replace")
    starts = [int(x) for x in re.findall(r"temp_c=(\d+)", text)]
    samples = [
        (int(t), float(p), int(vram))
        for t, p, vram in re.findall(
            r"sample=(\d+),\s*([0-9.]+),\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*(\d+)",
            text,
        )
    ]
    return {
        "start_c": starts[-1] if starts else None,
        "average_c": safe_div(sum(x[0] for x in samples), len(samples)) if samples else None,
        "maximum_c": max((x[0] for x in samples), default=None),
        "average_power_w": safe_div(sum(x[1] for x in samples), len(samples)) if samples else None,
        "maximum_vram_mib": max((x[2] for x in samples), default=None),
    }


def changed_config(config: dict[str, str], reference: dict[str, str]) -> str:
    keys = sorted((set(config) | set(reference)) - IGNORED_CONFIG_KEYS)
    changes = []
    for key in keys:
        if key.startswith("DS4_") and config.get(key) != reference.get(key):
            changes.append(f"{key}={config.get(key, '<unset>')}")
    if config.get("cache_budget") != reference.get("cache_budget"):
        changes.append(f"cache_budget={config.get('cache_budget', '<unset>')}")
    return "; ".join(changes) or "baseline profile"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("logs/cuda_experiments"))
    parser.add_argument("--correctness-reference", type=Path, required=True)
    parser.add_argument("--performance-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    correctness_reference = summarize(
        args.correctness_reference,
        args.correctness_reference.with_name("server.log"),
    )
    performance_reference = summarize(
        args.performance_reference,
        args.performance_reference.with_name("server.log"),
    )
    reference_config = load_config(args.performance_reference.with_name("config.env"))
    reference_elapsed = performance_reference["summary"]["total_elapsed_s"]

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        results_path = run_dir / "results.json"
        if not results_path.exists():
            continue
        candidate = summarize(results_path, run_dir / "server.log")
        config = load_config(run_dir / "config.env")
        check = correctness(correctness_reference, candidate)
        summary = candidate["summary"]
        thermal = thermal_summary(run_dir / "thermal.log")
        rows.append({
            "run": run_dir.name,
            "experiment": config.get("experiment", run_dir.name),
            "change": changed_config(config, reference_config),
            "exact_output": check["exact_match"],
            "changed_prompts": [item["id"] for item in check["differences"]],
            "total_elapsed_s": summary.get("total_elapsed_s"),
            "change_vs_control": safe_div(
                summary.get("total_elapsed_s", 0) - reference_elapsed,
                reference_elapsed,
            ),
            "prefill_tps": summary.get("prefill_tokens_per_second"),
            "decode_tps": summary.get("decode_tokens_per_second"),
            "completion_tps": summary.get("weighted_end_to_end_completion_tps"),
            "cache_hit_rate": summary.get("cache_hit_rate"),
            "cache_ssd_gib": summary.get("cache_ssd_gib"),
            **thermal,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "experiment_report.json").write_text(
        json.dumps({
            "schema": "ds4-cuda-experiment-report-v1",
            "correctness_reference": str(args.correctness_reference),
            "performance_reference": str(args.performance_reference),
            "runs": rows,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_fields = list(rows[0]) if rows else []
    with (args.output_dir / "experiment_report.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "changed_prompts": ",".join(row["changed_prompts"])})

    exact_rows = [row for row in rows if row["exact_output"]]
    fastest = min(exact_rows, key=lambda row: row["total_elapsed_s"]) if exact_rows else None
    lines = [
        "# DS4 CUDA experiment report",
        "",
        f"Correctness reference: `{args.correctness_reference}`",
        f"Performance control: `{args.performance_reference}`",
        "",
        "| Experiment | Change | Exact | Total s | vs control | Prefill t/s | Decode t/s | Hit rate | SSD GiB | Start/avg/max °C |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        temp = "/".join(
            "-" if row[key] is None else f"{row[key]:.1f}"
            for key in ("start_c", "average_c", "maximum_c")
        )
        lines.append(
            f"| {row['experiment']} | {row['change']} | "
            f"{'yes' if row['exact_output'] else 'no'} | {row['total_elapsed_s']:.2f} | "
            f"{row['change_vs_control']:+.1%} | {row['prefill_tps']:.3f} | "
            f"{row['decode_tps']:.3f} | "
            f"{row['cache_hit_rate']:.1%} | {row['cache_ssd_gib']:.2f} | {temp} |"
        )
    if fastest:
        lines.extend([
            "",
            f"Fastest exact-output run: **{fastest['experiment']}**, "
            f"{fastest['total_elapsed_s']:.2f} s.",
        ])
    (args.output_dir / "experiment_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"runs": len(rows), "fastest_exact": fastest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
