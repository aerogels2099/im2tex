"""Label your own equation images in a local web page: each image beside an editable LaTeX label and its
pdflatex rendering (the evaluation's own renderer, so what you see is what gets scored).

New images in SOURCE (png/jpg/jpeg/webp/bmp, one equation each) are copied into SPLIT/images as PNG, and
SPLIT/all/refs.jsonl lists them all with empty labels, so any model can be run over it to pre-fill
suggestions. Saved labels go to SPLIT/refs.jsonl in the common split layout: the split becomes a test set.
    python scripts/label.py /mnt/c/Users/thoma/im2tex-real/slides ~/im2tex-data/real/slides
    python baselines/run_baseline.py im2tex-base ~/im2tex-data/real/slides/all /tmp/suggest.jsonl
    python scripts/label.py /mnt/c/Users/thoma/im2tex-real/slides ~/im2tex-data/real/slides --suggest /tmp/suggest.jsonl
then open http://localhost:8765 in a browser.
"""
import argparse
import base64
import io
import json
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

from PIL import Image, ImageOps

from im2tex.normalize import NormalizeError, normalize
from im2tex.render.latex_batch import render

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def import_images(source: Path, split: Path) -> list[str]:
    """Copy images not imported yet into split/images as PNG; -> all image ids, sorted."""
    (split / "images").mkdir(parents=True, exist_ok=True)
    for p in sorted(source.iterdir()):
        dst = split / "images" / (re.sub(r"[^A-Za-z0-9_-]+", "_", p.name) + ".png")
        if p.suffix.lower() in EXTENSIONS and not dst.exists():
            im = ImageOps.exif_transpose(Image.open(p))  # phone photos are often stored sideways
            if "A" in im.getbands():  # transparent pixels would turn black
                im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im.convert("RGBA"))
            im.convert("RGB").save(dst)
    ids = sorted(f.stem for f in (split / "images").glob("*.png"))
    (split / "all").mkdir(exist_ok=True)
    with (split / "all" / "refs.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps({"id": i, "image": f"../images/{i}.png", "ref": ""}) + "\n" for i in ids)
    return ids


def load_labels(split: Path) -> dict:
    """id -> {"ref": latex} or {"skip": True}."""
    path = split / "labels.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_label(split: Path, labels: dict, image_id: str, ref: str | None = None, skip: bool = False):
    """Record one label (or skip the image) and rewrite refs.jsonl; -> an error message, or None if saved."""
    if not skip:
        try:
            norm = normalize(ref)
        except NormalizeError as e:
            return f"the normalizer rejects this label: {e}"
        if not norm:
            return "empty label (use Skip for an unusable image)"
        if render([ref], workers=1)[0] is None:
            return "this label doesn't compile with the evaluation preamble"
    labels[image_id] = {"skip": True} if skip else {"ref": ref}
    tmp = split / "labels.json.tmp"
    tmp.write_text(json.dumps(labels, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, split / "labels.json")
    with (split / "refs.jsonl").open("w", encoding="utf-8") as f:
        for i in sorted(labels):
            if "ref" in labels[i]:
                f.write(json.dumps({"id": i, "image": f"images/{i}.png", "ref": labels[i]["ref"],
                                    "norm": normalize(labels[i]["ref"])}) + "\n")
    return None


PAGE = """<!doctype html><meta charset=utf-8><title>Label equations</title>
<style>
body { font: 15px system-ui, sans-serif; margin: 16px auto; max-width: 1100px; padding: 0 16px; }
.box { border: 1px solid #bbb; padding: 8px; margin: 8px 0; min-height: 48px; overflow: auto; background: #fff; }
.box img { max-width: 100%; }
textarea { width: 100%; height: 96px; font: 15px monospace; box-sizing: border-box; }
.err { color: #b00020; }
</style>
<div><b id=pos></b> &middot; <span id=state></span> &middot; <span id=counts></span></div>
<div class=box><img id=src alt=""></div>
<textarea id=tex spellcheck=false></textarea>
<div><button id=prev>&larr; Prev (Alt+&larr;)</button> <button id=next>Next (Alt+&rarr;)</button>
<button id=skip>Skip: unusable image</button> <button id=save><b>Save &amp; next (Ctrl+Enter)</b></button>
<span id=msg class=err></span></div>
<div>Rendered (what the evaluation sees):</div>
<div class=box id=out></div>
<script>
let items = [], i = 0, timer = null;
const $ = id => document.getElementById(id);
const post = (path, data) => fetch(path, {method: "POST", body: JSON.stringify(data)}).then(r => r.json());
function show() {
  const it = items[i];
  $("pos").textContent = `${i + 1} / ${items.length} (${it.id})`;
  $("state").textContent = it.skip ? "skipped" : it.label !== null ? "labeled"
    : it.suggestion !== null ? "model suggestion: check it against the image" : "unlabeled";
  const done = items.filter(x => x.label !== null).length, skipped = items.filter(x => x.skip).length;
  $("counts").textContent = `${done} labeled, ${skipped} skipped, ${items.length - done - skipped} to go`;
  $("src").src = "/image/" + encodeURIComponent(it.id);
  $("tex").value = it.label ?? it.suggestion ?? "";
  $("msg").textContent = "";
  preview();
}
async function preview() {
  const latex = $("tex").value, r = await post("/render", {latex});
  if (latex !== $("tex").value) return;  // the text changed while rendering
  const out = $("out");
  out.replaceChildren();
  if (r.png) { const img = new Image(); img.src = "data:image/png;base64," + r.png; out.append(img); }
  else { out.textContent = r.error; out.className = "box err"; return; }
  out.className = "box";
}
async function save(skip) {
  const it = items[i];
  const r = await post("/save", skip ? {id: it.id, skip: true} : {id: it.id, ref: $("tex").value});
  if (r.error) { $("msg").textContent = r.error; return; }
  Object.assign(it, skip ? {skip: true, label: null} : {skip: false, label: $("tex").value});
  go(1);
}
function go(d) { i = Math.min(Math.max(i + d, 0), items.length - 1); show(); }
$("tex").oninput = () => { clearTimeout(timer); timer = setTimeout(preview, 400); };
$("prev").onclick = () => go(-1); $("next").onclick = () => go(1);
$("skip").onclick = () => save(true); $("save").onclick = () => save(false);
document.onkeydown = e => {
  if (e.ctrlKey && e.key === "Enter") { e.preventDefault(); save(false); }
  else if (e.altKey && e.key === "ArrowRight") { e.preventDefault(); go(1); }
  else if (e.altKey && e.key === "ArrowLeft") { e.preventDefault(); go(-1); }
};
fetch("/items").then(r => r.json()).then(d => {
  items = d;
  i = Math.max(0, items.findIndex(x => x.label === null && !x.skip));  // resume at the first open image
  if (items.length) show(); else $("pos").textContent = "no images found";
});
</script>
"""


def serve(split: Path, ids: list[str], suggestions: dict, port: int):
    labels = load_labels(split)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, kind: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            self._send(json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/items":
                self._json([{"id": i, "label": labels.get(i, {}).get("ref"), "skip": "skip" in labels.get(i, {}),
                             "suggestion": suggestions.get(i)} for i in ids])
            elif self.path.startswith("/image/") and unquote(self.path[7:]) in ids:
                self._send((split / "images" / f"{unquote(self.path[7:])}.png").read_bytes(), "image/png")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/render":
                img = render([body["latex"]], dpi=150, workers=1)[0] if body["latex"].strip() else None
                if img is None:
                    self._json({"error": "doesn't compile" if body["latex"].strip() else "(empty)"})
                    return
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, "PNG")
                self._json({"png": base64.b64encode(buf.getvalue()).decode()})
            elif self.path == "/save" and body.get("id") in ids:
                self._json({"error": save_label(split, labels, body["id"], body.get("ref"), body.get("skip", False))})
            else:
                self._send(b"not found", "text/plain", 404)

        def log_message(self, *args):
            pass

    print(f"{len(ids)} images, {sum('ref' in v for v in labels.values())} labeled: open http://localhost:{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="folder of equation images (one equation per image)")
    ap.add_argument("split", type=Path, help="output split directory, e.g. ~/im2tex-data/real/slides")
    ap.add_argument("--suggest", type=Path, help="predictions JSONL (run_baseline output) to pre-fill labels")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    ids = import_images(args.source, args.split)
    suggestions = {}
    if args.suggest:
        suggestions = {r["id"]: r["pred"] for r in map(json.loads, args.suggest.open(encoding="utf-8"))}
    serve(args.split, ids, suggestions, args.port)
