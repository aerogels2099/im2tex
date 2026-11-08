"""Collect runs/<model>/<test set>.metrics.json into markdown tables, one per metric.

    python scripts/results_table.py runs
"""
import json
import sys
from pathlib import Path

ORDER = ["unimer_spe", "unimer_cpe", "unimer_sce", "unimer_hwe", "crohme2014", "crohme2016", "crohme2019",
         "mathwriting", "im2latex"]
METRICS = {"image_match": "renders identically", "exprate": "exact match (normalized)",
           "bleu4": "BLEU-4 (normalized)", "sec_per_image": "seconds per image"}

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
results = {m.name: {f.name.removesuffix(".metrics.json"): json.loads(f.read_text())
                    for f in m.glob("*.metrics.json")} for m in sorted(root.iterdir()) if m.is_dir()}
results = {m: r for m, r in results.items() if r}
sets = [s for s in ORDER if any(s in r for r in results.values())]
for key, title in METRICS.items():
    print(f"\n{title}\n\n| model | " + " | ".join(sets) + " |\n|---|" + "---|" * len(sets))
    for model, r in results.items():
        cells = []
        for s in sets:
            v = r.get(s, {}).get(key)
            cells.append("" if v is None else f"{v:.2f}" if key == "sec_per_image" else f"{100 * v:.1f}")
        print(f"| {model} | " + " | ".join(cells) + " |")
