import glob
import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

APP_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(APP_DIR), static_folder=str(APP_DIR))
BASE_DIR = Path(os.environ.get("TAGGER_BASE", ".")).resolve()
JSONL_GLOB = os.environ.get("TAGGER_JSONL", "data/ocr/*/index.jsonl")
LABELS_PATH = Path(os.environ.get("TAGGER_LABELS", "data/labels.jsonl"))


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_all():
    items = []
    for jsonl_path in sorted(
        (BASE_DIR / Path(p) for p in glob.glob(str(BASE_DIR / JSONL_GLOB))),
        key=str,
    ):
        for row in read_jsonl(jsonl_path):
            row["_src"] = str(jsonl_path)
            items.append(row)
    labels_by_path = {}
    for row in read_jsonl(BASE_DIR / LABELS_PATH):
        input_path = row.get("input_path")
        if input_path:
            labels_by_path[input_path] = row.get("categories", [])
    for item in items:
        input_path = item.get("input_path")
        if input_path in labels_by_path:
            item["categories"] = labels_by_path[input_path]
    return items


def strip_src(item):
    return {k: v for k, v in item.items() if k != "_src"}


def read_labels_by_path():
    labels_by_path = {}
    for row in read_jsonl(BASE_DIR / LABELS_PATH):
        input_path = row.get("input_path")
        if input_path:
            labels_by_path[input_path] = row.get("categories", [])
    return labels_by_path


def write_labels(labels_by_path):
    labels_path = BASE_DIR / LABELS_PATH
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w") as handle:
        for input_path, categories in sorted(labels_by_path.items()):
            handle.write(json.dumps({"input_path": input_path, "categories": categories}) + "\n")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/style.css")
def style():
    return send_file(APP_DIR / "style.css")


@app.get("/api/items")
def get_items():
    return jsonify([strip_src(i) for i in load_all()])


@app.get("/api/tags")
def get_tags():
    tags = set()
    for item in load_all():
        tags.update(item.get("categories", []))
    return jsonify(sorted(tags))


@app.patch("/api/item/<int:idx>")
def patch_item(idx):
    body = request.get_json(silent=True) or {}
    categories = body.get("categories", [])
    if not isinstance(categories, list) or any(not isinstance(entry, str) for entry in categories):
        return jsonify({"error": "categories must be a list of strings"}), 400
    items = load_all()
    if idx >= len(items):
        return jsonify({"error": "out of range"}), 404
    items[idx]["categories"] = categories
    labels_by_path = read_labels_by_path()
    labels_by_path[items[idx]["input_path"]] = categories
    write_labels(labels_by_path)
    return jsonify(strip_src(items[idx]))


@app.get("/image")
def serve_image():
    rel = request.args.get("path", "")
    if not rel:
        return "no path", 400
    abs_path = (BASE_DIR / rel).resolve()
    if not str(abs_path).startswith(str(BASE_DIR)):
        return "forbidden", 403
    return send_file(abs_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
