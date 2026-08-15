#!/usr/bin/env bash
set -euo pipefail

base_url="${DS4_SERVER_URL:-http://127.0.0.1:18099}"

request() {
    curl -fsS --max-time 120 \
        -H 'Content-Type: application/json' \
        -d "$1" \
        "$base_url/v1/chat/completions"
}

exact_json="$(request '{"model":"deepseek-chat","messages":[{"role":"user","content":"Reply with exactly this text and nothing else: VERDE-17"}],"max_tokens":16,"thinking":false,"temperature":0}')"
math_json="$(request '{"model":"deepseek-chat","messages":[{"role":"user","content":"Qual è la differenza tra 1 e 0,9 periodico? Rispondi in una frase."}],"max_tokens":48,"thinking":false,"temperature":0}')"

python3 - "$exact_json" "$math_json" <<'PY'
import json
import sys

exact = json.loads(sys.argv[1])["choices"][0]["message"]["content"]
math = json.loads(sys.argv[2])["choices"][0]["message"]["content"]

if exact != "VERDE-17":
    raise SystemExit(f"exact-output adherence failed: {exact!r}")

normalized = math.lower().replace(" ", "")
correct = (
    "differenzaèzero" in normalized
    or "differenzae'zero" in normalized
    or "0,999…èesattamenteugualea1" in normalized
    or "0,999...=1" in normalized
)
if not correct:
    raise SystemExit(f"math correctness failed: {math!r}")
if math.count(".") > 1 and "..." not in math:
    raise SystemExit(f"one-sentence adherence failed: {math!r}")

print("prompt adherence: PASS")
print(f"exact: {exact}")
print(f"math: {math}")
PY
