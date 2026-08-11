#!/usr/bin/env python3
"""Summarize and compare deterministic DS4 CUDA benchmark experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def load_config(path: Path | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path:
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def summarize(results_path: Path, server_log: Path | None) -> dict[str, Any]:
    report = json.loads(results_path.read_text(encoding="utf-8"))
    summary = dict(report.get("summary", {}))
    log_text = server_log.read_text(encoding="utf-8") if server_log else ""
    prefill_seconds = sum(
        float(value) for value in re.findall(r"prompt done ([0-9.]+)s", log_text)
    )
    decode_seconds = sum(
        float(value)
        for value in re.findall(
            r"decoding chunk=[0-9.]+ t/s avg=[0-9.]+ t/s ([0-9.]+)s", log_text
        )
    )
    prompt_tokens = int(summary.get("total_prompt_tokens", 0))
    completion_tokens = int(summary.get("total_completion_tokens", 0))
    cache_rows = [
        {
            "hits": int(hits),
            "misses": int(misses),
            "evictions": int(evictions),
            "ssd_gib": float(ssd_gib),
            "capacity": int(capacity),
            "resident": int(resident),
        }
        for hits, misses, evictions, ssd_gib, capacity, resident in re.findall(
            r"CUDA prompt expert cache summary hits=(\d+) misses=(\d+) "
            r"evictions=(\d+) SSD=([0-9.]+) GiB capacity=(\d+) resident=(\d+)",
            log_text,
        )
    ]
    phase_rows = [
        {
            "prefill_hits": int(prefill_hits),
            "prefill_misses": int(prefill_misses),
            "prefill_evictions": int(prefill_evictions),
            "prefill_ssd_gib": float(prefill_ssd_gib),
            "decode_hits": int(decode_hits),
            "decode_misses": int(decode_misses),
            "decode_evictions": int(decode_evictions),
            "decode_ssd_gib": float(decode_ssd_gib),
        }
        for (
            prefill_hits,
            prefill_misses,
            prefill_evictions,
            prefill_ssd_gib,
            decode_hits,
            decode_misses,
            decode_evictions,
            decode_ssd_gib,
        ) in re.findall(
            r"prefill_hits=(\d+) prefill_misses=(\d+) prefill_evictions=(\d+) "
            r"prefill_SSD=([0-9.]+) GiB decode_hits=(\d+) decode_misses=(\d+) "
            r"decode_evictions=(\d+) decode_SSD=([0-9.]+) GiB",
            log_text,
        )
    ]
    for row, phase_row in zip(cache_rows, phase_rows):
        row.update(phase_row)
    cache_hits = sum(row["hits"] for row in cache_rows)
    cache_misses = sum(row["misses"] for row in cache_rows)
    cache_requests = cache_hits + cache_misses
    summary.update(
        {
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "prefill_tokens_per_second": safe_div(prompt_tokens, prefill_seconds),
            "decode_tokens_per_second": safe_div(completion_tokens, decode_seconds),
            "cache_prompt_summaries": len(cache_rows),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": safe_div(cache_hits, cache_requests),
            "cache_evictions": sum(row["evictions"] for row in cache_rows),
            "cache_ssd_gib": sum(row["ssd_gib"] for row in cache_rows),
            "cache_per_prompt": cache_rows,
            "cache_prefill_hits": sum(row["prefill_hits"] for row in phase_rows),
            "cache_prefill_misses": sum(row["prefill_misses"] for row in phase_rows),
            "cache_prefill_ssd_gib": sum(row["prefill_ssd_gib"] for row in phase_rows),
            "cache_decode_hits": sum(row["decode_hits"] for row in phase_rows),
            "cache_decode_misses": sum(row["decode_misses"] for row in phase_rows),
            "cache_decode_ssd_gib": sum(row["decode_ssd_gib"] for row in phase_rows),
        }
    )
    return {"summary": summary, "results": report.get("results", [])}


def correctness(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_by_id = {row["id"]: row for row in reference.get("results", [])}
    differences = []
    for row in candidate.get("results", []):
        base = reference_by_id.get(row["id"])
        if not base:
            differences.append({"id": row["id"], "reason": "missing_reference"})
            continue
        fields = ("status", "answer", "prompt_tokens", "completion_tokens", "finish_reason")
        changed = {
            field: {"reference": base.get(field), "candidate": row.get(field)}
            for field in fields
            if base.get(field) != row.get(field)
        }
        if changed:
            differences.append({"id": row["id"], "changed": changed})
    return {"exact_match": not differences, "differences": differences}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--reference-server-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = summarize(args.results, args.server_log)
    output: dict[str, Any] = {
        "schema": "ds4-cuda-experiment-summary-v1",
        "config": load_config(args.config),
        **candidate,
    }
    if args.reference_results:
        reference = summarize(args.reference_results, args.reference_server_log)
        output["correctness"] = correctness(reference, candidate)
        comparisons = {}
        for metric in (
            "total_elapsed_s",
            "prefill_seconds",
            "decode_seconds",
            "weighted_end_to_end_completion_tps",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
        ):
            base = float(reference["summary"].get(metric, 0.0))
            value = float(candidate["summary"].get(metric, 0.0))
            comparisons[metric] = {
                "reference": base,
                "candidate": value,
                "change_rate": safe_div(value - base, base),
            }
        output["comparison"] = comparisons

    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("summary", "correctness", "comparison") if key in output}, indent=2))
    return 0 if output.get("correctness", {}).get("exact_match", True) else 3


if __name__ == "__main__":
    raise SystemExit(main())
