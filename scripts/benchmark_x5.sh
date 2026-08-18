#!/bin/bash
# 5-repeat benchmark for stable Table 7 numbers: CPU-only throughput depends on
# machine state (observed ±50% between sessions for the rule/CRF configs), so
# the paper reports the median [min–max] over five back-to-back repetitions.
# Run AFTER the final models are in place (uses the canonical model paths).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p reports/benchmark_reps
for i in 1 2 3 4 5; do
  echo "=== benchmark rep $i/5  $(date +%H:%M:%S) ==="
  $PY scripts/benchmark.py --configs --out "reports/benchmark_reps/benchmark_$i.json" || exit 1
done
$PY - <<'EOF'
import json, statistics
reps = [json.load(open(f"reports/benchmark_reps/benchmark_{i}.json")) for i in range(1, 6)]
cfgs = [c["configuration"] for c in reps[0]["configurations"]]
out = {"test_file": reps[0]["test_file"], "repetitions": 5,
       "platform": reps[0]["platform"], "configurations": []}
for name in cfgs:
    rows = [r for rep in reps for r in rep["configurations"] if r["configuration"] == name]
    tp = sorted(r["throughput_tokens_per_sec"] for r in rows)
    ram = sorted(r["ram_mb"] for r in rows)
    acc = {r["accuracy_pct"] for r in rows}
    out["configurations"].append({
        "configuration": name,
        "throughput_median": round(statistics.median(tp), 1),
        "throughput_min": round(tp[0], 1), "throughput_max": round(tp[-1], 1),
        "ram_median_mb": statistics.median(ram),
        "ram_min_mb": ram[0], "ram_max_mb": ram[-1],
        "accuracy_pct": sorted(acc),
        "throughput_all": tp,
    })
    print(f"{name:24s} tp_median={statistics.median(tp):8.1f} "
          f"[{tp[0]:8.1f} – {tp[-1]:8.1f}]  ram_med={statistics.median(ram):6.0f} MB  "
          f"acc={sorted(acc)}")
json.dump(out, open("reports/benchmark_x5.json", "w"), indent=2)
print("written reports/benchmark_x5.json")
EOF
