#!/usr/bin/env python3
"""Correctness/performance benchmark for an already-running ds4 server.

Cold/warm are labels only; this script never restarts or reconfigures ds4.
"""

import argparse
import datetime
import json
import math
import os
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request


CORPUS = (
    ("it_math", "it", "Rispondi in italiano con una sola frase: calcola 17 per 19 e spiega brevemente il calcolo."),
    ("en_cache", "en", "Answer in one English sentence: why does a cache hit usually reduce storage traffic?"),
    ("it_summary", "it", "Riassumi in una frase: Un sistema corretto conserva gli stessi risultati mentre riduce il lavoro non necessario."),
)


class BenchError(RuntimeError):
    pass


def num(value, *path):
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def api(endpoint, path, timeout, payload=None):
    url = endpoint.rstrip("/") + path
    data = None
    headers = {"Accept": "application/json", "User-Agent": "ds4-roadmap-bench/1"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise BenchError("%s: HTTP %d: %s" % (url, exc.code, exc.read().decode("utf-8", "replace"))) from exc
    except (OSError, ValueError) as exc:
        raise BenchError("%s: %s" % (url, exc)) from exc
    if not isinstance(result, dict):
        raise BenchError("%s returned non-object JSON" % url)
    return result


def snapshot(args):
    health = api(args.endpoint, "/health", args.timeout)
    profile = api(args.endpoint, "/profile", args.timeout)
    if health.get("status") != "ok" or not isinstance(profile.get("prompts"), list):
        raise BenchError("server health/profile endpoint is not ready")
    return health, profile


def select_new_profile(profile_before, profile_after):
    """Select the newest entry created after ``profile_before``.

    /profile returns its history newest-first, so slicing from the end would
    select stale entries.  The sequence captured immediately before the
    request is the authoritative lower bound for new prompt IDs.
    """
    before_seq = int(profile_before.get("seq", 0))
    candidates = []
    for item in profile_after.get("prompts", []):
        item_id = item.get("id") if isinstance(item, dict) else None
        if isinstance(item_id, int) and not isinstance(item_id, bool) and item_id > before_seq:
            candidates.append(item)
    candidates.sort(key=lambda item: item["id"], reverse=True)
    selected = candidates[0] if candidates else None
    warning = None
    if not candidates:
        warning = "no new /profile entry with id > seq_before"
    elif len(candidates) > 1:
        warning = (
            "%d new profile entries; selected max id=%d, concurrent traffic "
            "may affect attribution" % (len(candidates), selected["id"])
        )
    return selected, warning


def run_prompt(args, index, case, repeat):
    case_id, language, prompt = case
    request = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "thinking": False,
        "seed": 42000 + index,
        "stream": False,
    }
    profile_before = api(args.endpoint, "/profile", args.timeout)
    started = time.monotonic()
    response = api(args.endpoint, "/v1/chat/completions", args.timeout, request)
    elapsed = time.monotonic() - started
    choices = response.get("choices") or []
    if not choices:
        raise BenchError("response for %s has no choice" % case_id)
    message = choices[0].get("message") or {}
    usage = response.get("usage") or {}
    profile_after = api(args.endpoint, "/profile", args.timeout)
    metric, warning = select_new_profile(profile_before, profile_after)
    return {
        "case_id": case_id,
        "language": language,
        "repeat": repeat,
        "client_elapsed_s": round(elapsed, 6),
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "finish_reason": choices[0].get("finish_reason"),
        "usage": usage,
        "profile": metric,
        "profile_seq_before": int(profile_before.get("seq", 0)),
        "profile_seq_after": int(profile_after.get("seq", 0)),
        "request": request,
        "response": response,
    }, warning


def counter_delta(before, after, *path):
    old, new = num(before, *path), num(after, *path)
    return new - old if old is not None and new is not None and new >= old else None


def summarize(runs, before, after):
    profiles = [run["profile"] for run in runs if isinstance(run.get("profile"), dict)]
    completion = sum(num(run, "usage", "completion_tokens") or 0 for run in runs)
    gen_s = sum(num(item, "gen_s") or 0 for item in profiles)
    hits = sum(num(item, "expert_hits") or 0 for item in profiles)
    misses = sum(num(item, "expert_misses") or 0 for item in profiles)
    disk = sum(num(item, "disk_bytes") or 0 for item in profiles)
    def median(name):
        values = [num(item, name) for item in profiles]
        values = [value for value in values if value is not None]
        return round(statistics.median(values), 6) if values else None
    counters = {
        "expert_hits": counter_delta(before, after, "expert_cache", "hits"),
        "expert_misses": counter_delta(before, after, "expert_cache", "misses"),
        "disk_bytes": counter_delta(before, after, "disk", "bytes_read"),
        "disk_reads": counter_delta(before, after, "disk", "reads"),
        "disk_read_s": counter_delta(before, after, "disk", "total_read_s"),
    }
    return {
        "requests": len(runs),
        "prompt_tokens": sum(num(run, "usage", "prompt_tokens") or 0 for run in runs),
        "completion_tokens": completion,
        "median_ttft_s": median("ttft_s"),
        "median_gen_s": median("gen_s"),
        "median_tok_s": median("tok_s"),
        "aggregate_tok_s": round(completion / gen_s, 6) if completion and gen_s else None,
        "expert_hits": hits,
        "expert_misses": misses,
        "expert_hit_rate": round(hits / (hits + misses), 6) if hits + misses else None,
        "disk_bytes": disk,
        "disk_bytes_per_token": round(disk / completion, 6) if completion else None,
        "health_delta": counters,
    }


def signature(run):
    usage = run.get("usage") or {}
    return {
        "content": run.get("content") or "",
        "reasoning_content": run.get("reasoning_content") or "",
        "finish_reason": run.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def compare_reference(runs, summary, reference, warning_pct, failure_pct):
    current = {"%s#%s" % (r["case_id"], r["repeat"]): r for r in runs}
    old = {"%s#%s" % (r["case_id"], r["repeat"]): r for r in reference["runs"]}
    mismatches = []
    for key in sorted(set(current) | set(old)):
        if key not in current or key not in old or signature(current.get(key, {})) != signature(old.get(key, {})):
            mismatches.append({"key": key, "current": signature(current.get(key, {})), "reference": signature(old.get(key, {}))})
    performance, warnings, failures = {}, [], []
    definitions = (("aggregate_tok_s", True), ("median_ttft_s", False), ("median_gen_s", False), ("disk_bytes_per_token", False))
    old_summary = reference.get("summary") or {}
    for name, higher_is_better in definitions:
        value, baseline = summary.get(name), old_summary.get(name)
        regression = None
        if isinstance(value, (int, float)) and isinstance(baseline, (int, float)) and baseline > 0 and math.isfinite(value):
            regression = ((baseline - value) if higher_is_better else (value - baseline)) / baseline * 100
        warned = regression is not None and regression > warning_pct
        failed = failure_pct is not None and regression is not None and regression > failure_pct
        performance[name] = {"current": value, "reference": baseline, "regression_pct": round(regression, 6) if regression is not None else None,
                             "warning": warned, "failure": failed}
        if warned:
            warnings.append("%s regressed by %.2f%%" % (name, regression))
        if failed:
            failures.append("%s regressed by %.2f%%" % (name, regression))
    return mismatches, performance, warnings, failures


def write_json(path, value):
    target = os.path.abspath(path)
    directory = os.path.dirname(target) or "."
    if not os.path.isdir(directory):
        raise BenchError("output directory does not exist: %s" % directory)
    fd, temporary = tempfile.mkstemp(prefix=".bench_cuda_roadmap.", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def matcher_self_test():
    before = {"seq": 7, "prompts": [{"id": 7}, {"id": 6}]}
    newest_first = {
        "seq": 8,
        "prompts": [
            {"id": 8, "completion_tokens": 31},
            {"id": 7, "completion_tokens": 31},
            {"id": 6, "completion_tokens": 12},
        ],
    }
    selected, warning = select_new_profile(before, newest_first)
    assert selected["id"] == 8 and warning is None

    concurrent = {
        "seq": 10,
        "prompts": [{"id": 10}, {"id": 9}, {"id": 8}, {"id": 7}],
    }
    selected, warning = select_new_profile(before, concurrent)
    assert selected["id"] == 10 and "3 new profile entries" in warning

    selected, warning = select_new_profile(before, before)
    assert selected is None and "id > seq_before" in warning
    print("matcher self-test: PASS")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a running ds4 server without restarting it.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18099")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--output")
    parser.add_argument("--reference")
    parser.add_argument("--label", choices=("cold", "warm"), default="warm", help="annotation only")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--perf-warning-pct", type=float, default=5.0)
    parser.add_argument("--fail-perf-regression-pct", type=float, default=None)
    parser.add_argument("--self-test", action="store_true", help="test newest-first profile matching without contacting a server")
    args = parser.parse_args()
    if not args.self_test and not args.output:
        parser.error("--output is required unless --self-test is used")
    if args.repeats <= 0 or args.max_tokens <= 0 or args.timeout <= 0:
        parser.error("repeats, max-tokens, and timeout must be positive")
    if args.perf_warning_pct < 0 or (args.fail_perf_regression_pct is not None and args.fail_perf_regression_pct < 0):
        parser.error("performance thresholds must be non-negative")
    return args


def main():
    args = parse_args()
    if args.self_test:
        matcher_self_test()
        return 0
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    before_health, before_profile = snapshot(args)
    runs, warnings = [], []
    for repeat in range(args.repeats):
        for index, case in enumerate(CORPUS):
            run, warning = run_prompt(args, index, case, repeat)
            runs.append(run)
            if warning:
                warnings.append("%s#%d: %s" % (case[0], repeat, warning))
    after_health, after_profile = snapshot(args)
    summary = summarize(runs, before_health, after_health)
    comparison, failures = None, []
    if args.reference:
        try:
            with open(args.reference, "r", encoding="utf-8") as handle:
                reference = json.load(handle)
        except (OSError, ValueError) as exc:
            raise BenchError("cannot load reference: %s" % exc) from exc
        mismatches, perf, perf_warnings, perf_failures = compare_reference(
            runs, summary, reference, args.perf_warning_pct, args.fail_perf_regression_pct)
        warnings.extend(perf_warnings)
        failures.extend(perf_failures)
        if mismatches:
            failures.append("%d text/token comparison(s) differ" % len(mismatches))
        comparison = {"path": os.path.abspath(args.reference), "exact_match": not mismatches,
                      "mismatches": mismatches, "performance": perf,
                      "failure_threshold_pct": args.fail_perf_regression_pct}
    result = {
        "schema_version": 1,
        "started_at": started,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "configuration": {"endpoint": args.endpoint, "model": args.model, "repeats": args.repeats,
                          "max_tokens": args.max_tokens, "label": args.label,
                          "label_is_annotation_only": True, "temperature": 0, "thinking": False,
                          "perf_warning_pct": args.perf_warning_pct,
                          "fail_perf_regression_pct": args.fail_perf_regression_pct},
        "corpus": [{"id": c[0], "language": c[1], "prompt": c[2]} for c in CORPUS],
        "snapshots": {"before": {"health": before_health, "profile": before_profile},
                      "after": {"health": after_health, "profile": after_profile}},
        "runs": runs,
        "summary": summary,
        "reference_comparison": comparison,
        "warnings": warnings,
        "failures": failures,
    }
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "output": os.path.abspath(args.output),
                      "requests": len(runs), "aggregate_tok_s": summary["aggregate_tok_s"],
                      "warnings": len(warnings), "failures": len(failures)}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
