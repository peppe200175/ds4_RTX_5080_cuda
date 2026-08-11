#!/usr/bin/env python3
"""Analyze DS4_EXPERT_TRACE JSONL and build reusable layer/expert hitmaps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


NUMERIC_METRICS = (
    "route_events",
    "cache_stat_transactions",
    "observed_transactions",
    "failed_transactions",
    "layer_token_evaluations",
    "expert_selections",
    "selection_cache_hits",
    "selection_cache_misses",
    "unique_requests",
    "unique_cache_hits",
    "unique_cache_misses",
    "evictions",
    "insertions",
    "model_bytes_read",
    "cache_hit_bytes",
    "bytes_copied",
    "gpu_sync_ms",
    "readback_ms",
    "cache_load_ms",
    "resume_ms",
)


def empty_metrics() -> dict[str, float]:
    return {name: 0 for name in NUMERIC_METRICS}


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def finalized(metrics: dict[str, float]) -> dict[str, float]:
    result = dict(metrics)
    result["selection_hit_rate"] = safe_div(
        metrics["selection_cache_hits"], metrics["expert_selections"]
    )
    result["transaction_hit_rate"] = safe_div(
        metrics["unique_cache_hits"], metrics["unique_requests"]
    )
    result["disk_read_gib"] = metrics["model_bytes_read"] / (1024**3)
    result["cache_hit_gib"] = metrics["cache_hit_bytes"] / (1024**3)
    result["bytes_copied_gib"] = metrics["bytes_copied"] / (1024**3)
    result["mean_cache_load_ms"] = safe_div(
        metrics["cache_load_ms"], metrics["observed_transactions"]
    )
    return result


def update_metrics(
    target: dict[str, float], event: dict[str, Any], hit_selections: int, miss_selections: int
) -> None:
    after = event.get("cache_after") or {}
    persistent = after.get("persistent") or {}
    timing = event.get("timing_ms") or {}
    observed = bool(event.get("cache_transaction_observed"))
    has_cache_stats = bool(after.get("valid")) and int(after.get("unique_experts", 0)) == (
        int(persistent.get("last_hits", 0)) + int(persistent.get("last_misses", 0))
    )

    target["route_events"] += 1
    target["cache_stat_transactions"] += int(has_cache_stats)
    target["observed_transactions"] += int(observed)
    target["failed_transactions"] += int(observed and event.get("load_ok") is False)
    target["layer_token_evaluations"] += int(event.get("token_count", 0))
    target["expert_selections"] += hit_selections + miss_selections
    target["selection_cache_hits"] += hit_selections
    target["selection_cache_misses"] += miss_selections
    target["unique_requests"] += int(after.get("unique_experts", 0))
    target["unique_cache_hits"] += int(persistent.get("last_hits", 0))
    target["unique_cache_misses"] += int(persistent.get("last_misses", 0))
    target["evictions"] += int(persistent.get("last_evictions", 0))
    target["insertions"] += int(persistent.get("last_insertions", 0))
    target["model_bytes_read"] += int(persistent.get("last_model_bytes_read", 0))
    target["cache_hit_bytes"] += int(persistent.get("last_cache_hit_bytes", 0))
    target["bytes_copied"] += int(after.get("last_bytes_copied", 0))
    target["gpu_sync_ms"] += float(timing.get("gpu_sync", 0.0))
    target["readback_ms"] += float(timing.get("readback", 0.0))
    target["cache_load_ms"] += float(timing.get("cache_load", 0.0))
    target["resume_ms"] += float(timing.get("resume", 0.0))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simulate_layer_cache(
    events: list[tuple[str, list[int], list[int]]], capacity: int
) -> int:
    """Replay the traced LFU-prefill/LRU-decode policy and return SSD misses."""
    frequency = [0] * 256
    residents: dict[int, int] = {}
    clock = 0
    misses = 0
    for phase, occurrences, unique_experts in events:
        for expert in occurrences:
            frequency[expert] += 1
        missing: list[int] = []
        for expert in unique_experts:
            if expert in residents:
                clock += 1
                residents[expert] = clock
            else:
                misses += 1
                missing.append(expert)
        for expert in missing:
            if len(residents) < capacity:
                clock += 1
                residents[expert] = clock
                continue
            if not residents:
                continue
            if phase == "decode":
                victim = min(residents, key=residents.get)
            else:
                victim = min(
                    residents, key=lambda resident: (frequency[resident], residents[resident])
                )
                if frequency[expert] <= frequency[victim]:
                    continue
            del residents[victim]
            clock += 1
            residents[expert] = clock
    return misses


def simulate_phase_resized_layer_cache(
    events: list[tuple[str, list[int], list[int]]],
    prefill_capacity: int,
    decode_capacity: int,
) -> int:
    """Replay a cache that reclaims prefill-only VRAM during decode."""
    frequency = [0] * 256
    residents: dict[int, int] = {}
    clock = 0
    misses = 0
    for phase, occurrences, unique_experts in events:
        capacity = decode_capacity if phase == "decode" else prefill_capacity
        if len(residents) > capacity:
            keep = sorted(
                residents,
                key=lambda expert: (frequency[expert], residents[expert]),
                reverse=True,
            )[:capacity]
            residents = {expert: residents[expert] for expert in keep}
        for expert in occurrences:
            frequency[expert] += 1
        missing: list[int] = []
        for expert in unique_experts:
            if expert in residents:
                clock += 1
                residents[expert] = clock
            else:
                misses += 1
                missing.append(expert)
        for expert in missing:
            if len(residents) < capacity:
                clock += 1
                residents[expert] = clock
                continue
            if not residents:
                continue
            if phase == "decode":
                victim = min(residents, key=residents.get)
            else:
                victim = min(
                    residents, key=lambda resident: (frequency[resident], residents[resident])
                )
                if frequency[expert] <= frequency[victim]:
                    continue
            del residents[victim]
            clock += 1
            residents[expert] = clock
    return misses


def optimize_layer_capacities(
    replay_events: dict[int, list[tuple[str, list[int], list[int]]]],
    current_capacities: dict[int, int],
    layers: int,
    minimum: int = 6,
    maximum: int = 128,
) -> dict[str, Any]:
    """Find the trace-optimal per-layer split for the same total slot budget."""
    budget = sum(current_capacities.get(layer, 0) for layer in range(layers))
    maximum = min(maximum, 256)
    costs: list[list[int]] = [[10**12] * (maximum + 1) for _ in range(layers)]
    for layer in range(layers):
        for capacity in range(minimum, maximum + 1):
            costs[layer][capacity] = simulate_layer_cache(
                replay_events.get(layer, []), capacity
            )

    infinity = 10**15
    previous = [infinity] * (budget + 1)
    previous[0] = 0
    parents: list[list[int]] = []
    for layer in range(layers):
        current = [infinity] * (budget + 1)
        parent = [-1] * (budget + 1)
        for used, prior_cost in enumerate(previous):
            if prior_cost >= infinity:
                continue
            cap_max = min(maximum, budget - used)
            for capacity in range(minimum, cap_max + 1):
                candidate = prior_cost + costs[layer][capacity]
                if candidate < current[used + capacity]:
                    current[used + capacity] = candidate
                    parent[used + capacity] = capacity
        previous = current
        parents.append(parent)

    if previous[budget] >= infinity:
        raise ValueError("unable to allocate the layer-cache slot budget")
    recommended = [0] * layers
    used = budget
    for layer in range(layers - 1, -1, -1):
        capacity = parents[layer][used]
        if capacity < 0:
            raise ValueError("invalid layer-cache capacity backtrack")
        recommended[layer] = capacity
        used -= capacity

    rows = []
    baseline_misses = 0
    recommended_misses = 0
    for layer in range(layers):
        old_capacity = current_capacities.get(layer, 0)
        old_misses = simulate_layer_cache(replay_events.get(layer, []), old_capacity)
        new_misses = costs[layer][recommended[layer]]
        baseline_misses += old_misses
        recommended_misses += new_misses
        rows.append(
            {
                "layer": layer,
                "current_capacity": old_capacity,
                "recommended_capacity": recommended[layer],
                "current_misses": old_misses,
                "recommended_misses": new_misses,
                "predicted_misses_saved": old_misses - new_misses,
            }
        )
    return {
        "assumptions": "Exact LFU prefill + LRU decode replay; prompt-aware and pinned policies disabled.",
        "slot_budget": budget,
        "minimum_slots_per_layer": minimum,
        "maximum_slots_per_layer": maximum,
        "baseline_replayed_misses": baseline_misses,
        "recommended_replayed_misses": recommended_misses,
        "predicted_misses_saved": baseline_misses - recommended_misses,
        "rows": rows,
    }


def write_matrix(
    path: Path,
    cells: dict[tuple[int, int], dict[str, Any]],
    metric: str,
    layers: int,
    experts: int,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["expert_id", *[f"layer_{layer}" for layer in range(layers)]])
        for expert in range(experts):
            writer.writerow(
                [expert]
                + [cells.get((layer, expert), {}).get(metric, 0) for layer in range(layers)]
            )


def write_html_hitmap(
    path: Path,
    cells: list[dict[str, Any]],
    layers: int,
    experts: int,
    source_name: str,
) -> None:
    payload = json.dumps(cells, separators=(",", ":"))
    title = html.escape(f"DwarfStar expert hitmap — {source_name}")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font:14px system-ui;background:#101318;color:#e7edf5;margin:20px}}
h1{{font-size:20px}} .controls{{display:flex;gap:12px;align-items:center;margin:12px 0}}
canvas{{background:#181d25;border:1px solid #374151;image-rendering:pixelated;max-width:100%}}
#tip{{position:fixed;display:none;background:#05070a;border:1px solid #64748b;padding:8px;pointer-events:none;white-space:pre}}
.note{{color:#9aa7b8;max-width:1000px}}
</style></head><body><h1>{title}</h1>
<div class="controls"><label>Metric <select id="metric">
<option value="selected_count">Expert selections</option>
<option value="disk_loads">Unique SSD loads</option>
<option value="transaction_cache_hits">Unique cache hits</option>
<option value="prefill_disk_loads">Prefill SSD loads</option>
<option value="decode_disk_loads">Decode SSD loads</option>
<option value="prompt_presence">Prompt presence</option>
<option value="selection_hit_rate">Selection hit rate</option>
</select></label><span id="scale"></span></div>
<p class="note">Rows are layers 0–{layers - 1}; columns are experts 0–{experts - 1}. Hover a cell for exact counters.</p>
<canvas id="map" width="{experts * 5}" height="{layers * 8}"></canvas><div id="tip"></div>
<script>
const rows={payload}, layers={layers}, experts={experts}, by=new Map(rows.map(x=>[x.layer+','+x.expert_id,x]));
const canvas=document.querySelector('#map'), ctx=canvas.getContext('2d'), select=document.querySelector('#metric'), tip=document.querySelector('#tip');
function value(cell,metric){{return metric==='selection_hit_rate' ? (cell.selection_hit_rate||0) : (cell[metric]||0)}}
function draw(){{const metric=select.value, vals=rows.map(x=>value(x,metric)), max=metric==='selection_hit_rate'?1:Math.max(1,...vals);ctx.clearRect(0,0,canvas.width,canvas.height);for(let l=0;l<layers;l++)for(let e=0;e<experts;e++){{const c=by.get(l+','+e)||{{}},v=value(c,metric),q=Math.sqrt(v/max);ctx.fillStyle=`rgb(${{Math.round(20+235*q)}},${{Math.round(28+120*q)}},${{Math.round(45+30*(1-q))}})`;ctx.fillRect(e*5,l*8,5,8)}}document.querySelector('#scale').textContent=`max ${{max.toFixed(3)}}`;}}
select.onchange=draw;canvas.onmousemove=ev=>{{const r=canvas.getBoundingClientRect(),e=Math.floor((ev.clientX-r.left)*canvas.width/r.width/5),l=Math.floor((ev.clientY-r.top)*canvas.height/r.height/8),c=by.get(l+','+e)||{{layer:l,expert_id:e}};tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';tip.textContent=`layer ${{l}}, expert ${{e}}\nselections ${{c.selected_count||0}}\nSSD loads ${{c.disk_loads||0}}\ncache hits ${{c.transaction_cache_hits||0}}\nprompts ${{c.prompt_presence||0}}\nselection hit rate ${{((c.selection_hit_rate||0)*100).toFixed(1)}}%`;}};canvas.onmouseleave=()=>tip.style.display='none';draw();
</script></body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--benchmark-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--decode-reclaim-gib",
        type=float,
        default=0.0,
        help="prefill-only VRAM that a proposed phase-resized cache may reclaim",
    )
    args = parser.parse_args()

    benchmark: dict[str, Any] = {}
    benchmark_rows: list[dict[str, Any]] = []
    if args.benchmark_results:
        benchmark = json.loads(args.benchmark_results.read_text(encoding="utf-8"))
        benchmark_rows = list(benchmark.get("results", []))

    prompt_events: list[dict[str, Any]] = []
    trace_starts = 0
    for _, event in iter_jsonl(args.trace):
        if event.get("event") == "trace_start":
            trace_starts += 1
        elif event.get("event") == "prompt":
            prompt_events.append(event)

    valid_prompt_keys = {
        (int(event.get("session_id", 0)), int(event.get("prompt_id", 0)))
        for event in prompt_events
    }
    prompt_info: dict[tuple[int, int], dict[str, Any]] = {}
    for index, event in enumerate(prompt_events):
        key = (int(event.get("session_id", 0)), int(event.get("prompt_id", 0)))
        bench = benchmark_rows[index] if index < len(benchmark_rows) else {}
        prompt_info[key] = {
            "index": index + 1,
            "id": bench.get("id", f"prompt_{index + 1}"),
            "category": bench.get("category", ""),
            "session_id": key[0],
            "prompt_id": key[1],
            "token_count": int(event.get("token_count", 0)),
            "text": event.get("text", ""),
            "raw_prompt": bench.get("prompt", ""),
            "completion_tokens": int(bench.get("completion_tokens", 0)),
            "answer": bench.get("answer", ""),
        }

    totals = empty_metrics()
    phase_metrics = defaultdict(empty_metrics)
    layer_metrics = defaultdict(empty_metrics)
    layer_phase_metrics: dict[int, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(empty_metrics)
    )
    prompt_metrics = defaultdict(empty_metrics)
    prompt_phase_metrics: dict[
        tuple[int, int], dict[str, dict[str, float]]
    ] = defaultdict(lambda: defaultdict(empty_metrics))
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    final_residents: dict[int, list[int]] = {}
    layer_capacity: dict[int, int] = {}
    malformed_routes = 0
    ignored_unbound_routes = 0
    data_quality = {
        "route_counter_mismatches": 0,
        "unique_count_mismatches": 0,
        "selection_count_mismatches": 0,
        "timed_cumulative_delta_mismatches": 0,
    }
    prefill_transactions: list[dict[str, int]] = []
    replay_events: dict[int, list[tuple[str, list[int], list[int]]]] = defaultdict(list)
    expert_payload_bytes = 0
    max_layer = -1
    max_expert = -1

    for _, event in iter_jsonl(args.trace):
        if event.get("event") != "expert_route":
            continue
        prompt_key = (int(event.get("session_id", 0)), int(event.get("prompt_id", 0)))
        if prompt_key not in valid_prompt_keys:
            ignored_unbound_routes += 1
            continue

        layer = int(event.get("layer", -1))
        phase = str(event.get("phase", "unknown"))
        max_layer = max(max_layer, layer)
        unique_status: dict[int, bool] = {}
        occurrence_count: Counter[int] = Counter()
        hit_selections = 0
        miss_selections = 0

        for route in event.get("routes", []):
            for selected in route.get("selected", []):
                expert = int(selected.get("expert_id", -1))
                if expert < 0 or layer < 0:
                    malformed_routes += 1
                    continue
                max_expert = max(max_expert, expert)
                hit = bool(selected.get("cache_before"))
                unique_status[expert] = hit
                occurrence_count[expert] += 1
                hit_selections += int(hit)
                miss_selections += int(not hit)
                cell = cells.setdefault(
                    (layer, expert),
                    {
                        "layer": layer,
                        "expert_id": expert,
                        "selected_count": 0,
                        "prefill_selected": 0,
                        "decode_selected": 0,
                        "rank0_count": 0,
                        "weight_sum": 0.0,
                        "selection_cache_hits": 0,
                        "selection_cache_misses": 0,
                        "transaction_cache_hits": 0,
                        "disk_loads": 0,
                        "estimated_disk_bytes": 0,
                        "prompt_keys": set(),
                    },
                )
                cell["selected_count"] += 1
                cell[f"{phase}_selected"] = cell.get(f"{phase}_selected", 0) + 1
                cell["rank0_count"] += int(int(selected.get("rank", -1)) == 0)
                weight = selected.get("weight")
                if isinstance(weight, (int, float)):
                    cell["weight_sum"] += float(weight)
                cell["selection_cache_hits"] += int(hit)
                cell["selection_cache_misses"] += int(not hit)
                cell[f"{phase}_selection_cache_hits"] = (
                    cell.get(f"{phase}_selection_cache_hits", 0) + int(hit)
                )
                cell[f"{phase}_selection_cache_misses"] = (
                    cell.get(f"{phase}_selection_cache_misses", 0) + int(not hit)
                )
                cell["prompt_keys"].add(prompt_key)

        after = event.get("cache_after") or {}
        bytes_per_expert = after.get("bytes_per_expert") or {}
        expert_bytes = sum(int(bytes_per_expert.get(name, 0)) for name in ("gate", "up", "down"))
        expert_payload_bytes = max(expert_payload_bytes, expert_bytes)
        replay_events[layer].append((phase, list(occurrence_count.elements()), list(unique_status)))
        for expert, hit in unique_status.items():
            cell = cells[(layer, expert)]
            if hit:
                cell["transaction_cache_hits"] += 1
                cell[f"{phase}_transaction_cache_hits"] = (
                    cell.get(f"{phase}_transaction_cache_hits", 0) + 1
                )
            else:
                cell["disk_loads"] += 1
                cell[f"{phase}_disk_loads"] = cell.get(f"{phase}_disk_loads", 0) + 1
                cell["estimated_disk_bytes"] += expert_bytes

        persistent = after.get("persistent") or {}
        route_hits = sum(unique_status.values())
        route_misses = len(unique_status) - route_hits
        if (route_hits, route_misses) != (
            int(persistent.get("last_hits", 0)),
            int(persistent.get("last_misses", 0)),
        ):
            data_quality["route_counter_mismatches"] += 1
        if len(unique_status) != int(after.get("unique_experts", 0)):
            data_quality["unique_count_mismatches"] += 1
        if hit_selections + miss_selections != (
            int(event.get("token_count", 0)) * int(event.get("experts_per_token", 0))
        ):
            data_quality["selection_count_mismatches"] += 1
        if event.get("cache_transaction_observed"):
            before_persistent = ((event.get("cache_before") or {}).get("persistent") or {})
            cumulative_fields = (
                "hits", "misses", "evictions", "insertions",
                "model_bytes_read", "cache_hit_bytes",
            )
            if any(
                int(persistent.get(name, 0)) - int(before_persistent.get(name, 0))
                != int(persistent.get(f"last_{name}", 0))
                for name in cumulative_fields
            ):
                data_quality["timed_cumulative_delta_mismatches"] += 1

        if phase == "prefill":
            capacity = int(persistent.get("layer_capacity", 0))
            single_use = sum(count == 1 for count in occurrence_count.values())
            repeat_use = len(occurrence_count) - single_use
            prefill_transactions.append(
                {
                    "unique_requests": len(unique_status),
                    "capacity": capacity,
                    "single_use_unique": single_use,
                    "repeat_use_unique": repeat_use,
                    "single_use_misses": sum(
                        occurrence_count[expert] == 1 and not hit
                        for expert, hit in unique_status.items()
                    ),
                    "repeat_use_misses": sum(
                        occurrence_count[expert] > 1 and not hit
                        for expert, hit in unique_status.items()
                    ),
                }
            )

        update_metrics(totals, event, hit_selections, miss_selections)
        update_metrics(phase_metrics[phase], event, hit_selections, miss_selections)
        update_metrics(layer_metrics[layer], event, hit_selections, miss_selections)
        update_metrics(layer_phase_metrics[layer][phase], event, hit_selections, miss_selections)
        update_metrics(prompt_metrics[prompt_key], event, hit_selections, miss_selections)
        update_metrics(
            prompt_phase_metrics[prompt_key][phase], event, hit_selections, miss_selections
        )

        layer_capacity[layer] = int(persistent.get("layer_capacity", 0))
        final_residents[layer] = [
            int(item.get("expert_id", -1))
            for item in persistent.get("layer_experts", [])
            if int(item.get("expert_id", -1)) >= 0
        ]

    layers = max(43, max_layer + 1)
    experts = max(256, max_expert + 1)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    serial_cells: list[dict[str, Any]] = []
    for layer in range(layers):
        for expert in range(experts):
            raw = cells.get((layer, expert))
            if not raw:
                continue
            item = {key: value for key, value in raw.items() if key != "prompt_keys"}
            item["prompt_presence"] = len(raw["prompt_keys"])
            item["prompt_presence_rate"] = safe_div(len(raw["prompt_keys"]), len(prompt_events))
            item["selection_hit_rate"] = safe_div(
                raw["selection_cache_hits"], raw["selected_count"]
            )
            item["transaction_hit_rate"] = safe_div(
                raw["transaction_cache_hits"],
                raw["transaction_cache_hits"] + raw["disk_loads"],
            )
            item["pinning_benefit_score"] = max(item["disk_loads"] - 1, 0) * item[
                "prompt_presence_rate"
            ]
            serial_cells.append(item)

    cell_lookup = {(item["layer"], item["expert_id"]): item for item in serial_cells}
    layer_rows = []
    for layer in range(layers):
        layer_cells = [item for item in serial_cells if item["layer"] == layer]
        top_selected = sorted(
            layer_cells,
            key=lambda item: (
                item["prompt_presence"], item["selected_count"], item["weight_sum"]
            ),
            reverse=True,
        )[: args.top]
        layer_rows.append(
            {
                "layer": layer,
                **finalized(layer_metrics[layer]),
                "capacity": layer_capacity.get(layer, 0),
                "final_residents": final_residents.get(layer, []),
                "phases": {
                    name: finalized(values)
                    for name, values in sorted(layer_phase_metrics[layer].items())
                },
                "top_experts": [
                    {
                        "expert_id": item["expert_id"],
                        "selected_count": item["selected_count"],
                        "prompt_presence": item["prompt_presence"],
                        "disk_loads": item["disk_loads"],
                        "transaction_cache_hits": item["transaction_cache_hits"],
                        "selection_hit_rate": item["selection_hit_rate"],
                    }
                    for item in top_selected
                ],
            }
        )

    pinning_candidates: dict[int, list[dict[str, Any]]] = {}
    avoidable_loads_upper_bound = 0
    for layer in range(layers):
        ranked = sorted(
            (item for item in serial_cells if item["layer"] == layer),
            key=lambda item: (
                item["pinning_benefit_score"],
                item["disk_loads"],
                item["selected_count"],
            ),
            reverse=True,
        )[:2]
        pinning_candidates[layer] = ranked
        avoidable_loads_upper_bound += sum(max(item["disk_loads"] - 1, 0) for item in ranked)

    capacity_replay = optimize_layer_capacities(
        replay_events, layer_capacity, layers, minimum=6, maximum=128
    )
    capacity_replay["baseline_matches_trace"] = (
        capacity_replay["baseline_replayed_misses"] == int(totals["unique_cache_misses"])
    )
    capacity_replay["predicted_disk_read_saved_gib"] = (
        capacity_replay["predicted_misses_saved"] * expert_payload_bytes / (1024**3)
    )
    capacity_replay["predicted_miss_reduction_rate"] = safe_div(
        capacity_replay["predicted_misses_saved"],
        capacity_replay["baseline_replayed_misses"],
    )

    decode_reclaim_replay: dict[str, Any] = {
        "assumptions": (
            "Prefill retains the traced capacities; decode expands evenly. Before the next "
            "prefill, excess residents are reduced by global LFU/recency. Allocation and "
            "resize overhead are not modeled."
        ),
        "reclaimable_gib": args.decode_reclaim_gib,
        "scenarios": [],
    }
    reclaim_slots = (
        int(args.decode_reclaim_gib * (1024**3) // expert_payload_bytes)
        if expert_payload_bytes
        else 0
    )
    base_budget = int(capacity_replay["slot_budget"])
    for fraction in (0.0, 1 / 3, 2 / 3, 1.0):
        added_slots = round(reclaim_slots * fraction)
        decode_budget = base_budget + added_slots
        decode_capacities = [
            decode_budget * (layer + 1) // layers - decode_budget * layer // layers
            for layer in range(layers)
        ]
        misses = sum(
            simulate_phase_resized_layer_cache(
                replay_events.get(layer, []),
                layer_capacity.get(layer, 0),
                decode_capacities[layer],
            )
            for layer in range(layers)
        )
        saved = int(totals["unique_cache_misses"]) - misses
        decode_reclaim_replay["scenarios"].append(
            {
                "reclaim_fraction": fraction,
                "added_slots": added_slots,
                "decode_slot_budget": decode_budget,
                "minimum_decode_slots_per_layer": min(decode_capacities),
                "maximum_decode_slots_per_layer": max(decode_capacities),
                "replayed_misses": misses,
                "predicted_misses_saved": saved,
                "predicted_disk_read_saved_gib": saved
                * expert_payload_bytes
                / (1024**3),
                "predicted_miss_reduction_rate": safe_div(
                    saved, totals["unique_cache_misses"]
                ),
            }
        )

    prompt_rows = []
    for event in prompt_events:
        key = (int(event.get("session_id", 0)), int(event.get("prompt_id", 0)))
        prompt_rows.append(
            {
                **prompt_info[key],
                **finalized(prompt_metrics[key]),
                "phases": {
                    name: finalized(values)
                    for name, values in sorted(prompt_phase_metrics[key].items())
                },
            }
        )

    prefill_unique = [item["unique_requests"] for item in prefill_transactions]
    prefill_pressure = {
        "transactions": len(prefill_transactions),
        "mean_unique_requests": safe_div(sum(prefill_unique), len(prefill_unique)),
        "median_unique_requests": percentile(prefill_unique, 0.5),
        "p90_unique_requests": percentile(prefill_unique, 0.9),
        "max_unique_requests": max(prefill_unique, default=0),
        "transactions_over_layer_capacity": sum(
            item["unique_requests"] > item["capacity"] for item in prefill_transactions
        ),
        "single_use_unique_requests": sum(
            item["single_use_unique"] for item in prefill_transactions
        ),
        "repeat_use_unique_requests": sum(
            item["repeat_use_unique"] for item in prefill_transactions
        ),
        "single_use_misses": sum(item["single_use_misses"] for item in prefill_transactions),
        "repeat_use_misses": sum(item["repeat_use_misses"] for item in prefill_transactions),
    }
    data_quality["validated_route_events"] = int(totals["route_events"])
    data_quality["all_checks_passed"] = not any(
        value for key, value in data_quality.items() if key.endswith("mismatches")
    )
    report = {
        "schema": "ds4-expert-trace-analysis-v2",
        "source": {
            "path": str(args.trace),
            "size_bytes": args.trace.stat().st_size,
            "sha256": sha256_file(args.trace),
            "trace_starts": trace_starts,
            "prompt_events": len(prompt_events),
            "ignored_unbound_route_events": ignored_unbound_routes,
            "malformed_routes": malformed_routes,
        },
        "model_shape": {"layers": layers, "experts_per_layer": experts},
        "data_quality": data_quality,
        "totals": finalized(totals),
        "phases": {name: finalized(values) for name, values in sorted(phase_metrics.items())},
        "prefill_pressure": prefill_pressure,
        "pinning_upper_bound": {
            "candidate_experts": 2 * layers,
            "avoidable_ssd_loads": avoidable_loads_upper_bound,
            "avoidable_disk_read_gib": avoidable_loads_upper_bound
            * expert_payload_bytes
            / (1024**3),
            "warning": "Gross replay-free upper bound; pinning consumes flexible slots and can create other misses.",
        },
        "capacity_replay": capacity_replay,
        "decode_reclaim_replay": decode_reclaim_replay,
        "prompts": prompt_rows,
        "layers": layer_rows,
        "cells": serial_cells,
    }
    (output_dir / "expert_trace_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    cell_fields = [
        "layer",
        "expert_id",
        "selected_count",
        "prefill_selected",
        "decode_selected",
        "rank0_count",
        "weight_sum",
        "prompt_presence",
        "prompt_presence_rate",
        "selection_cache_hits",
        "selection_cache_misses",
        "selection_hit_rate",
        "transaction_cache_hits",
        "disk_loads",
        "prefill_transaction_cache_hits",
        "prefill_disk_loads",
        "decode_transaction_cache_hits",
        "decode_disk_loads",
        "transaction_hit_rate",
        "estimated_disk_bytes",
        "pinning_benefit_score",
    ]
    with (output_dir / "expert_hitmap.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cell_fields)
        writer.writeheader()
        writer.writerows({field: item.get(field, 0) for field in cell_fields} for item in serial_cells)

    prompt_fields = [
        "index",
        "id",
        "category",
        "session_id",
        "prompt_id",
        "token_count",
        "completion_tokens",
        "route_events",
        "expert_selections",
        "unique_cache_hits",
        "unique_cache_misses",
        "transaction_hit_rate",
        "model_bytes_read",
        "disk_read_gib",
        "evictions",
        "cache_load_ms",
        "prefill_unique_hits",
        "prefill_unique_misses",
        "prefill_hit_rate",
        "prefill_disk_read_gib",
        "decode_unique_hits",
        "decode_unique_misses",
        "decode_hit_rate",
        "decode_disk_read_gib",
    ]
    with (output_dir / "prompt_cache_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=prompt_fields)
        writer.writeheader()
        for item in prompt_rows:
            flat = dict(item)
            prefill = item.get("phases", {}).get("prefill", {})
            decode = item.get("phases", {}).get("decode", {})
            flat.update(
                {
                    "prefill_unique_hits": prefill.get("unique_cache_hits", 0),
                    "prefill_unique_misses": prefill.get("unique_cache_misses", 0),
                    "prefill_hit_rate": prefill.get("transaction_hit_rate", 0),
                    "prefill_disk_read_gib": prefill.get("disk_read_gib", 0),
                    "decode_unique_hits": decode.get("unique_cache_hits", 0),
                    "decode_unique_misses": decode.get("unique_cache_misses", 0),
                    "decode_hit_rate": decode.get("transaction_hit_rate", 0),
                    "decode_disk_read_gib": decode.get("disk_read_gib", 0),
                }
            )
            writer.writerow({field: flat.get(field, "") for field in prompt_fields})

    capacity_fields = [
        "layer", "current_capacity", "recommended_capacity", "current_misses",
        "recommended_misses", "predicted_misses_saved",
    ]
    with (output_dir / "layer_capacity_replay.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=capacity_fields)
        writer.writeheader()
        writer.writerows(capacity_replay["rows"])
    (output_dir / "recommended_layer_capacities.txt").write_text(
        "# Analysis proposal only; the current runtime does not consume this file.\n"
        "# LAYER CAPACITY\n"
        + "\n".join(
            f"{row['layer']} {row['recommended_capacity']}"
            for row in capacity_replay["rows"]
        )
        + "\n",
        encoding="utf-8",
    )
    reclaim_fields = [
        "reclaim_fraction",
        "added_slots",
        "decode_slot_budget",
        "minimum_decode_slots_per_layer",
        "maximum_decode_slots_per_layer",
        "replayed_misses",
        "predicted_misses_saved",
        "predicted_disk_read_saved_gib",
        "predicted_miss_reduction_rate",
    ]
    with (output_dir / "decode_reclaim_replay.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=reclaim_fields)
        writer.writeheader()
        writer.writerows(decode_reclaim_replay["scenarios"])

    for metric in (
        "selected_count", "disk_loads", "transaction_cache_hits", "prompt_presence",
        "prefill_disk_loads", "decode_disk_loads",
    ):
        write_matrix(output_dir / f"matrix_{metric}.csv", cell_lookup, metric, layers, experts)

    recommended_lines = [
        "# Workload-specific top two experts per layer.",
        f"# Generated from {args.trace.name}; validate with an A/B benchmark before enabling.",
    ]
    for row in layer_rows:
        top = pinning_candidates[row["layer"]]
        if top:
            recommended_lines.append(
                " ".join([str(row["layer"]), *[str(item["expert_id"]) for item in top]])
            )
    (output_dir / "recommended_pinned_experts.txt").write_text(
        "\n".join(recommended_lines) + "\n", encoding="utf-8"
    )

    write_html_hitmap(
        output_dir / "expert_hitmap.html", serial_cells, layers, experts, args.trace.name
    )

    hottest_layers = sorted(layer_rows, key=lambda row: row["model_bytes_read"], reverse=True)[:10]
    globally_hot = sorted(
        serial_cells,
        key=lambda item: (item["prompt_presence"], item["selected_count"], item["weight_sum"]),
        reverse=True,
    )[:20]
    full_reclaim = decode_reclaim_replay["scenarios"][-1]
    reclaim_summary = (
        f"- Decode reuse of {args.decode_reclaim_gib:.2f} GiB prefill reserve: "
        f"{full_reclaim['predicted_misses_saved']} fewer misses "
        f"({full_reclaim['predicted_miss_reduction_rate']:.2%}, "
        f"{full_reclaim['predicted_disk_read_saved_gib']:.2f} GiB), before resize costs"
        if reclaim_slots
        else "- Decode prefill-reserve reuse was not evaluated (`--decode-reclaim-gib 0`)."
    )
    markdown = [
        "# DwarfStar expert trace analysis",
        "",
        f"- Trace: `{args.trace}`",
        f"- SHA-256: `{report['source']['sha256']}`",
        f"- Prompts: {len(prompt_events)}",
        f"- Route events: {int(totals['route_events'])}",
        f"- Cache-stat events: {int(totals['cache_stat_transactions'])}",
        f"- Directly timed cache transactions: {int(totals['observed_transactions'])}",
        f"- Expert selections: {int(totals['expert_selections'])}",
        f"- Unique cache requests: {int(totals['unique_requests'])}",
        f"- Unique cache hits: {int(totals['unique_cache_hits'])}",
        f"- Unique SSD misses: {int(totals['unique_cache_misses'])}",
        f"- Transaction hit rate: {finalized(totals)['transaction_hit_rate']:.2%}",
        f"- Model bytes read: {finalized(totals)['disk_read_gib']:.2f} GiB",
        f"- Evictions: {int(totals['evictions'])}",
        f"- Integrity checks: {'PASS' if data_quality['all_checks_passed'] else 'FAIL'}",
        "",
        "## Prefill cache pressure",
        "",
        f"- Mean/median/p90/max unique experts per layer transaction: "
        f"{prefill_pressure['mean_unique_requests']:.1f} / "
        f"{prefill_pressure['median_unique_requests']:.1f} / "
        f"{prefill_pressure['p90_unique_requests']:.1f} / "
        f"{prefill_pressure['max_unique_requests']}",
        f"- Transactions over their layer capacity: "
        f"{prefill_pressure['transactions_over_layer_capacity']} / "
        f"{prefill_pressure['transactions']}",
        f"- Single-use/repeated unique requests: "
        f"{prefill_pressure['single_use_unique_requests']} / "
        f"{prefill_pressure['repeat_use_unique_requests']}",
        f"- Gross top-two-per-layer pinning upper bound: "
        f"{avoidable_loads_upper_bound} SSD loads, "
        f"{report['pinning_upper_bound']['avoidable_disk_read_gib']:.2f} GiB",
        f"- Same-budget per-layer replay: "
        f"{capacity_replay['predicted_misses_saved']} fewer misses "
        f"({capacity_replay['predicted_miss_reduction_rate']:.2%}, "
        f"{capacity_replay['predicted_disk_read_saved_gib']:.2f} GiB); "
        f"baseline match={'yes' if capacity_replay['baseline_matches_trace'] else 'no'}",
        reclaim_summary,
        "",
        "## Phase summary",
        "",
        "| Phase | Selections | Unique hits | Unique misses | Hit rate | SSD GiB | Evictions | Cache-load s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in sorted(phase_metrics.items()):
        item = finalized(values)
        markdown.append(
            f"| {name} | {int(item['expert_selections'])} | {int(item['unique_cache_hits'])} | "
            f"{int(item['unique_cache_misses'])} | {item['transaction_hit_rate']:.2%} | "
            f"{item['disk_read_gib']:.2f} | {int(item['evictions'])} | {item['cache_load_ms']/1000:.2f} |"
        )
    markdown += [
        "",
        "## Layers with most SSD traffic",
        "",
        "| Layer | SSD GiB | Unique misses | Hit rate | Evictions | Top experts (id:prompt-count:selected) |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in hottest_layers:
        top_text = ", ".join(
            f"{item['expert_id']}:{item['prompt_presence']}:{item['selected_count']}"
            for item in row["top_experts"][:5]
        )
        markdown.append(
            f"| {row['layer']} | {row['disk_read_gib']:.2f} | {int(row['unique_cache_misses'])} | "
            f"{row['transaction_hit_rate']:.2%} | {int(row['evictions'])} | {top_text} |"
        )
    markdown += [
        "",
        "## Most stable layer/expert pairs",
        "",
        "| Layer | Expert | Prompts | Selections | SSD loads | Cache hits | Selection hit rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in globally_hot:
        markdown.append(
            f"| {item['layer']} | {item['expert_id']} | {item['prompt_presence']} | "
            f"{item['selected_count']} | {item['disk_loads']} | {item['transaction_cache_hits']} | "
            f"{item['selection_hit_rate']:.2%} |"
        )
    markdown += [
        "",
        "## Interpretation notes",
        "",
        "- `expert_selections` counts every routed occurrence. A repeated expert in one prefill batch is counted repeatedly.",
        "- `unique_cache_misses` and `disk_loads` count one load per expert per cache transaction.",
        "- `model_bytes_read` is the authoritative CUDA counter; per-cell bytes are estimates using the traced expert size.",
        "- All route/cache consistency checks passed only when `Integrity checks` is `PASS`.",
        "- `observed_transactions` means directly timed transactions; `cache_stat_transactions` includes untimed decode snapshots.",
        "- Tracing serializes the selected/shared path, so trace timing is diagnostic and must not replace the untraced performance benchmark.",
        "- The generated pinned profile is a workload-specific candidate, not an automatically enabled setting.",
        "",
    ]
    (output_dir / "analysis.md").write_text("\n".join(markdown), encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "totals": finalized(totals)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
