"""Write the common split layout for IAM's handwritten English text lines (train / validation / test), from the
parquet files that scripts/download_data.sh fetches.

IAM's labels are tokenized ("Exchange .", "Nobody 's", '" prop up "'); they are written back as people write them
("Exchange.", "Nobody's", '"prop up"'). "#" marks a crossed-out word in IAM; page transcriptions leave crossed-out
writing out, so it is dropped. Images are kept as the parquet files store them (JPEG, 128 px high).
    python scripts/prepare_iam.py
"""
import json
import os
import re
from pathlib import Path

import pyarrow.parquet as pq

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data")) / "iam"


def detokenize(text: str) -> str:
    text = re.sub(r" '(s|ll|re|d|m|ve|t)\b", r"'\1", text.replace("#", " "))
    words, opening = [], {'"': True, "'": True}
    glue_next = False
    for w in text.split():
        if w in opening:  # a lone quote: opening ones attach to the next word, closing ones to the previous
            if opening[w] or not words:
                words.append(w)
                glue_next = True
            else:
                words[-1] += w
            opening[w] = not opening[w]
        elif glue_next:
            words[-1] += w
            glue_next = False
        else:
            words.append(w)
    text = " ".join(words)
    return re.sub(r"\( ", "(", re.sub(r" ([.,;:?!)])", r"\1", text))


if __name__ == "__main__":
    for split in ("train", "validation", "test"):
        out = DATA / split
        (out / "images").mkdir(parents=True, exist_ok=True)
        t = pq.read_table(DATA / f"{split}.parquet")
        rows = []
        for i, (text, image) in enumerate(zip(t.column("text").to_pylist(),
                                              (x["bytes"] for x in t.column("image").to_pylist()))):
            row_id = f"{split}-{i:05d}"
            path = out / "images" / f"{row_id}.jpg"
            if not path.exists():
                path.write_bytes(image)
            ref = detokenize(text)
            rows.append({"id": row_id, "image": f"images/{row_id}.jpg", "ref": ref, "norm": ref or None})
        with (out / "refs.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(f"{split}: {len(rows)} lines, {sum(not r['norm'] for r in rows)} empty -> {out}")
