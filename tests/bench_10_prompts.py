#!/usr/bin/env python3
"""Run ten short, varied prompts against a DwarfStar OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PROMPTS = [
    {
        "id": "fact",
        "category": "conoscenza",
        "prompt": "Qual è la capitale del Portogallo? Rispondi con una sola frase.",
    },
    {
        "id": "arithmetic",
        "category": "calcolo",
        "prompt": "Calcola 17 × 24 e mostra soltanto il risultato.",
    },
    {
        "id": "translation",
        "category": "traduzione",
        "prompt": "Traduci in inglese: 'Il treno arriverà tra dieci minuti'.",
    },
    {
        "id": "summary",
        "category": "riassunto",
        "prompt": (
            "Riassumi in non più di 15 parole: Marta uscì presto, comprò il pane "
            "e tornò a casa prima che iniziasse la pioggia."
        ),
    },
    {
        "id": "sentiment",
        "category": "classificazione",
        "prompt": (
            "Classifica come positivo, neutro o negativo: "
            "'Il pacco è arrivato puntuale e in perfette condizioni'."
        ),
    },
    {
        "id": "python",
        "category": "codice",
        "prompt": (
            "Scrivi una funzione Python di una riga che restituisca il quadrato "
            "di ogni numero in una lista."
        ),
    },
    {
        "id": "logic",
        "category": "logica",
        "prompt": (
            "Anna è più alta di Luca e Luca è più alto di Marco. "
            "Chi è il più basso? Rispondi brevemente."
        ),
    },
    {
        "id": "formatting",
        "category": "formattazione",
        "prompt": "Elenca tre pianeti rocciosi separandoli soltanto con virgole.",
    },
    {
        "id": "creative",
        "category": "scrittura",
        "prompt": "Scrivi un micro-haiku italiano sul mare, massimo 12 parole.",
    },
    {
        "id": "practical",
        "category": "consiglio",
        "prompt": "Suggerisci due modi semplici per ridurre gli sprechi d'acqua in casa.",
    },
]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument(
        "--output", default="logs/bench_10_prompts_results.json"
    )
    args = parser.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    results = []

    for index, item in enumerate(PROMPTS, start=1):
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": False,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = json.loads(response.read().decode("utf-8"))
            elapsed = time.perf_counter() - started
            usage = payload.get("usage", {})
            prompt_details = usage.get("prompt_tokens_details", {}) or {}
            completion_tokens = int(usage.get("completion_tokens", 0))
            result = {
                **item,
                "status": "ok",
                "elapsed_s": elapsed,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": completion_tokens,
                "total_tokens": int(usage.get("total_tokens", 0)),
                "cached_tokens": int(prompt_details.get("cached_tokens", 0)),
                "end_to_end_completion_tps": (
                    completion_tokens / elapsed if elapsed > 0 else 0.0
                ),
                "answer": payload.get("choices", [{}])[0]
                .get("message", {})
                .get("content", ""),
                "finish_reason": payload.get("choices", [{}])[0].get(
                    "finish_reason", ""
                ),
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            elapsed = time.perf_counter() - started
            result = {
                **item,
                "status": "error",
                "elapsed_s": elapsed,
                "error": str(exc),
            }

        results.append(result)
        if result["status"] == "ok":
            print(
                f"[{index:02d}/10] {item['id']:<10} "
                f"{result['prompt_tokens']:>3} in + "
                f"{result['completion_tokens']:>3} out | "
                f"{result['elapsed_s']:>7.2f}s | "
                f"{result['end_to_end_completion_tps']:>5.2f} output t/s",
                flush=True,
            )
        else:
            print(
                f"[{index:02d}/10] {item['id']:<10} ERROR: {result['error']}",
                flush=True,
            )

    successful = [item for item in results if item["status"] == "ok"]
    elapsed_values = [item["elapsed_s"] for item in successful]
    total_elapsed = sum(elapsed_values)
    total_prompt = sum(item["prompt_tokens"] for item in successful)
    total_completion = sum(item["completion_tokens"] for item in successful)
    summary = {
        "successful_prompts": len(successful),
        "failed_prompts": len(results) - len(successful),
        "total_elapsed_s": total_elapsed,
        "mean_request_s": statistics.mean(elapsed_values) if elapsed_values else 0.0,
        "median_request_s": statistics.median(elapsed_values) if elapsed_values else 0.0,
        "p95_request_s": percentile(elapsed_values, 0.95),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "weighted_end_to_end_completion_tps": (
            total_completion / total_elapsed if total_elapsed > 0 else 0.0
        ),
    }
    report = {
        "schema": "ds4-bench-10-prompts-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "summary": summary,
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {output}", flush=True)
    return 0 if len(successful) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
