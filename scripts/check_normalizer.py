"""Check the normalizer on real labels: failure rate, idempotency, and that normalizing never changes the render.

    python scripts/check_normalizer.py formulas.txt [--n 5000]      (one formula per line, or a refs.jsonl)
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from im2tex.metrics import image_match
from im2tex.normalize import NormalizeError, normalize
from im2tex.render.latex_batch import render

ap = argparse.ArgumentParser()
ap.add_argument("path", type=Path)
ap.add_argument("--n", type=int, default=5000, help="random sample size (0 = all)")
ap.add_argument("--show", type=int, default=15, help="examples to print per problem")
args = ap.parse_args()

lines = args.path.read_text(encoding="utf-8").splitlines()
formulas = [json.loads(l)["ref"] if args.path.suffix == ".jsonl" else l for l in lines if l.strip()]
if args.n and len(formulas) > args.n:
    formulas = random.Random(0).sample(formulas, args.n)

normed, errors, failed = [], Counter(), []
for f in formulas:
    try:
        normed.append(normalize(f))
    except NormalizeError as e:
        normed.append(None)
        errors[str(e)] += 1
        failed.append((f, str(e)))
ok = [(f, g) for f, g in zip(formulas, normed) if g is not None]
not_idem = [(f, g) for f, g in ok if normalize(g) != g]

imgs = render([f for f, _ in ok] + [g for _, g in ok])
orig, new = imgs[: len(ok)], imgs[len(ok):]
both = [(o, n, f, g) for o, n, (f, g) in zip(orig, new, ok) if o is not None]
broke = [(f, g) for o, n, f, g in both if n is None]
changed = [(f, g) for o, n, f, g in both if n is not None and not image_match(o, n)]
fixed = sum(o is None and n is not None for o, n in zip(orig, new))

print(f"formulas:            {len(formulas)}")
print(f"normalize errors:    {len(formulas) - len(ok)}  {dict(errors.most_common(5))}")
print(f"not idempotent:      {len(not_idem)}")
print(f"original renders:    {len(both)}")
print(f"render unchanged:    {len(both) - len(broke) - len(changed)}  "
      f"({(len(both) - len(broke) - len(changed)) / max(len(both), 1):.2%})")
print(f"render changed:      {len(changed)}")
print(f"stopped compiling:   {len(broke)}")
print(f"started compiling:   {fixed}  (original failed, normalized renders)")
for title, rows in (("NORMALIZE ERROR", failed), ("NOT IDEMPOTENT", not_idem),
                    ("RENDER CHANGED", changed), ("STOPPED COMPILING", broke)):
    for f, g in rows[: args.show]:
        print(f"\n[{title}]\n  in:  {f}\n  out: {g}")
