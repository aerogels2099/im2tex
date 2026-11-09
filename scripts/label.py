"""Label your own equation images in a local web page: each image beside an editable LaTeX label and its
pdflatex rendering (the evaluation's own renderer, so what you see is what gets scored).

New images in SOURCE (png/jpg/jpeg/webp/bmp, one equation each) are copied into SPLIT/images as PNG, and
SPLIT/all/refs.jsonl lists them all with empty labels, so any model can be run over it to pre-fill
suggestions. Saved labels go to SPLIT/refs.jsonl in the common split layout: the split becomes a test set.
    python scripts/label.py /mnt/c/Users/thoma/im2tex-real/slides ~/im2tex-data/real/slides
    python baselines/run_baseline.py im2tex-base ~/im2tex-data/real/slides/all /tmp/suggest.jsonl
    python scripts/label.py /mnt/c/Users/thoma/im2tex-real/slides ~/im2tex-data/real/slides --suggest /tmp/suggest.jsonl
then open http://localhost:8765 in a browser.

With --pages each image is a whole page and its label a page transcription (format in im2tex/page.py),
shown beside the page with every formula rendered in place:
    python scripts/label.py ~/im2tex-data/notes/pages ~/im2tex-data/notes/linalg --pages
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

from PIL import Image

from im2tex.inference import load_image
from im2tex.normalize import NormalizeError, normalize
from im2tex.page import PageError, normalize_page, split_page
from im2tex.render.latex_batch import render

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def import_images(source: Path, split: Path) -> list[str]:
    """Copy images not imported yet into split/images as PNG; -> all image ids, sorted."""
    (split / "images").mkdir(parents=True, exist_ok=True)
    for p in sorted(source.iterdir()):
        dst = split / "images" / (re.sub(r"[^A-Za-z0-9_-]+", "_", p.name) + ".png")
        if p.suffix.lower() in EXTENSIONS and not dst.exists():
            load_image(p).save(dst)
    ids = sorted(f.stem for f in (split / "images").glob("*.png"))
    (split / "all").mkdir(exist_ok=True)
    with (split / "all" / "refs.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps({"id": i, "image": f"../images/{i}.png", "ref": ""}) + "\n" for i in ids)
    return ids


def load_labels(split: Path) -> dict:
    """id -> {"ref": latex} or {"skip": True}."""
    path = split / "labels.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_label(split: Path, labels: dict, image_id: str, ref: str | None = None, skip: bool = False,
               pages: bool = False):
    """Record one label (or skip the image) and rewrite refs.jsonl; -> an error message, or None if saved."""
    norm_of = normalize_page if pages else normalize
    if not skip:
        try:
            norm = norm_of(ref)
        except (NormalizeError, PageError) as e:
            return f"the normalizer rejects this label: {e}"
        if not norm:
            return "empty label (use Skip for an unusable image)"
        formulas = [c for k, c in split_page(ref) if k != "text"] if pages else [ref]
        for formula, img in zip(formulas, render(formulas, workers=1) if formulas else []):
            if img is None:
                return "doesn't compile with the evaluation preamble" + (f": {formula}" if pages else "")
    labels[image_id] = {"skip": True} if skip else {"ref": ref}
    tmp = split / "labels.json.tmp"
    tmp.write_text(json.dumps(labels, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, split / "labels.json")
    with (split / "refs.jsonl").open("w", encoding="utf-8") as f:
        for i in sorted(labels):
            if "ref" in labels[i]:
                f.write(json.dumps({"id": i, "image": f"images/{i}.png", "ref": labels[i]["ref"],
                                    "norm": norm_of(labels[i]["ref"])}) + "\n")
    return None


PAGE = """<!doctype html><meta charset=utf-8><title>Label equations</title>
<style>
body { font: 15px system-ui, sans-serif; margin: 16px auto; max-width: 1100px; padding: 0 16px; }
.box { border: 1px solid #bbb; padding: 8px; margin: 8px 0; min-height: 48px; overflow: auto; background: #fff; }
.box img { max-width: 100%; }
.box img.zoom { max-width: none; }
textarea { width: 100%; height: 96px; font: 15px monospace; box-sizing: border-box; }
.err, .bad { color: #b00020; }
.bad { font-family: monospace; }
.hint { display: none; color: #555; }
body.pages { max-width: none; }
.pages .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.pages .src { height: calc(100vh - 70px); }
.pages textarea { height: 40vh; }
.pages .hint { display: block; }
.pages #out { white-space: pre-wrap; font: 17px serif; line-height: 1.9; }
#out img { vertical-align: middle; }
#out img.display { display: block; margin: 6px auto; }
</style>
<body class=__MODE__>
<div><b id=pos></b> &middot; <span id=state></span> &middot; <span id=counts></span></div>
<div class=cols><div class="box src"><img id=src alt="" title="click to zoom"></div><div>
<div class=hint>One written line per line &middot; math as $inline$ or $$display$$ &middot; a drawing is [figure]
&middot; leave out crossed-out writing &middot; a dollar sign is \\$</div>
<textarea id=tex spellcheck=false></textarea>
<div><button id=prev>&larr; Prev (Alt+&larr;)</button> <button id=next>Next (Alt+&rarr;)</button>
<button id=skip>Skip: unusable image</button> <button id=save><b>Save &amp; next (Ctrl+Enter)</b></button>
<span id=msg class=err></span></div>
<div>Rendered (what the evaluation sees):</div>
<div class=box id=out></div>
</div></div>
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
  out.className = r.error ? "box err" : "box";
  if (r.error) { out.textContent = r.error; return; }
  for (const p of r.parts ?? [{png: r.png}]) {  // a page: text and formulas in order
    if (p.text !== undefined) { out.append(p.text); continue; }
    if (p.png) {
      const img = new Image();
      img.src = "data:image/png;base64," + p.png;
      if (p.display) img.className = "display";
      out.append(img);
    } else {  // this formula doesn't compile: show its source
      const bad = document.createElement("span");
      bad.className = "bad";
      bad.textContent = p.display ? `$$${p.bad}$$` : `$${p.bad}$`;
      out.append(bad);
    }
  }
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
$("src").onclick = () => $("src").classList.toggle("zoom");
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


def _png(img) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def serve(split: Path, ids: list[str], suggestions: dict, port: int, pages: bool = False):
    labels = load_labels(split)
    rendered = {}  # formula -> image or None: a page re-renders only the formulas that changed

    def preview(latex: str) -> dict:
        if not latex.strip():
            return {"error": "(empty)"}
        try:
            parts = split_page(latex) if pages else [("inline", latex)]
        except PageError as e:
            return {"error": str(e)}
        new = list(dict.fromkeys(c for k, c in parts if k != "text" and c not in rendered))
        if new:
            rendered.update(zip(new, render(new, dpi=150, workers=1)))
        if not pages:
            return {"png": _png(rendered[latex])} if rendered[latex] is not None else {"error": "doesn't compile"}

        def part(kind, content):
            if kind == "text":
                return {"text": content}
            img = rendered[content]
            return {"display": kind == "display"} | ({"png": _png(img)} if img is not None else {"bad": content})
        return {"parts": [part(k, c) for k, c in parts]}

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
                self._send(PAGE.replace("__MODE__", "pages" if pages else "formulas").encode(), "text/html; charset=utf-8")
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
                self._json(preview(body["latex"]))
            elif self.path == "/save" and body.get("id") in ids:
                self._json({"error": save_label(split, labels, body["id"], body.get("ref"), body.get("skip", False),
                                                pages)})
            else:
                self._send(b"not found", "text/plain", 404)

        def log_message(self, *args):
            pass

    print(f"{len(ids)} images, {sum('ref' in v for v in labels.values())} labeled: open http://localhost:{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="folder of images (one equation each, or one page with --pages)")
    ap.add_argument("split", type=Path, help="output split directory, e.g. ~/im2tex-data/real/slides")
    ap.add_argument("--pages", action="store_true", help="images are whole pages, labels page transcriptions")
    ap.add_argument("--suggest", type=Path, help="predictions JSONL (run_baseline output) to pre-fill labels")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    ids = import_images(args.source, args.split)
    suggestions = {}
    if args.suggest:
        suggestions = {r["id"]: r["pred"] for r in map(json.loads, args.suggest.open(encoding="utf-8"))}
    serve(args.split, ids, suggestions, args.port, args.pages)
