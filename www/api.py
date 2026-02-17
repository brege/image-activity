import glob
import importlib.util
import json
import os
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

APP_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(APP_DIR), static_folder=str(APP_DIR))
BASE_DIR = Path(os.environ.get("TAGGER_BASE", ".")).resolve()
STATE_DIR = Path(os.environ.get("TAGGER_STATE_DIR", "www/state"))
STATE_PATH = (BASE_DIR / STATE_DIR).resolve()
CONFIG_PATH = Path(os.environ.get("TAGGER_CONFIG", "config.yaml")).expanduser().resolve()
JSONL_GLOB = os.environ.get("TAGGER_JSONL", str(STATE_DIR / "items.jsonl"))
LABELS_PATH = Path(os.environ.get("TAGGER_LABELS", str(STATE_DIR / "labels.jsonl")))

ml_spec = importlib.util.spec_from_file_location("www_ml", APP_DIR / "ml.py")
if ml_spec is None or ml_spec.loader is None:
    raise RuntimeError("failed to load www/ml.py")
ml_pipeline = importlib.util.module_from_spec(ml_spec)
ml_spec.loader.exec_module(ml_pipeline)
SOURCE_ROOTS = [Path(root) for root in ml_pipeline.source_roots(CONFIG_PATH)]
LABELS_LOCK = threading.Lock()


def read_jsonl(path, strict=True):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if strict:
                        raise ValueError(f"invalid JSONL in {path}:{line_no}") from None
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
    with LABELS_LOCK:
        labels_by_path = read_labels_by_path_unlocked()
    for item in items:
        input_path = item.get("input_path")
        if input_path in labels_by_path:
            item["categories"] = labels_by_path[input_path]
    return items


def strip_src(item):
    return {k: v for k, v in item.items() if k != "_src"}


def read_labels_by_path_unlocked():
    labels_by_path = {}
    for row in read_jsonl(BASE_DIR / LABELS_PATH, strict=False):
        input_path = row.get("input_path")
        if input_path:
            labels_by_path[input_path] = row.get("categories", [])
    return labels_by_path


def read_labels_by_path():
    with LABELS_LOCK:
        return read_labels_by_path_unlocked()


def write_labels_unlocked(labels_by_path):
    labels_path = BASE_DIR / LABELS_PATH
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = labels_path.with_suffix(labels_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for input_path, categories in sorted(labels_by_path.items()):
            handle.write(json.dumps({"input_path": input_path, "categories": categories}) + "\n")
    tmp_path.replace(labels_path)


def write_labels(labels_by_path):
    with LABELS_LOCK:
        write_labels_unlocked(labels_by_path)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/style.css")
def style():
    return send_file(APP_DIR / "style.css")


@app.get("/app.js")
def app_script():
    return send_file(APP_DIR / "app.js")


@app.get("/favicon.svg")
def favicon():
    return send_file(APP_DIR / "favicon.svg")


@app.get("/api/items")
def get_items():
    selected_cluster = request.args.get("cluster", "")
    payload = []
    for item_idx, item in enumerate(load_all()):
        if selected_cluster:
            cluster_id = item.get("cluster")
            if str(cluster_id) != selected_cluster:
                continue
        row = strip_src(item)
        row["_idx"] = item_idx
        payload.append(row)
    return jsonify(payload)


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
    with LABELS_LOCK:
        labels_by_path = read_labels_by_path_unlocked()
        labels_by_path[items[idx]["input_path"]] = categories
        write_labels_unlocked(labels_by_path)
    response = strip_src(items[idx])
    response["_idx"] = idx
    return jsonify(response)


@app.post("/api/purge")
def purge_labels():
    items = load_all()
    valid_paths = {item["input_path"] for item in items if "input_path" in item}
    with LABELS_LOCK:
        labels_by_path = read_labels_by_path_unlocked()
        before_count = len(labels_by_path)
        labels_by_path = {
            input_path: categories
            for input_path, categories in labels_by_path.items()
            if input_path in valid_paths
        }
        after_count = len(labels_by_path)
        write_labels_unlocked(labels_by_path)
    return jsonify({"removed": before_count - after_count, "remaining": after_count})


@app.get("/api/purge-preview")
def purge_preview():
    items = load_all()
    valid_paths = {item["input_path"] for item in items if "input_path" in item}
    labels_by_path = read_labels_by_path()
    removed = sorted([input_path for input_path in labels_by_path if input_path not in valid_paths])
    return jsonify({"remove": removed, "count": len(removed)})


@app.post("/api/ml/start")
def start_ml():
    started = ml_pipeline.start_job(config_path=CONFIG_PATH, state_dir=STATE_PATH)
    return jsonify({"started": started})


@app.get("/api/ml/status")
def ml_status():
    return jsonify(ml_pipeline.get_status(config_path=CONFIG_PATH, state_dir=STATE_PATH))


@app.get("/api/ml/clusters")
def ml_clusters():
    return jsonify(ml_pipeline.get_clusters(state_dir=STATE_PATH))


def path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@app.get("/image")
def serve_image():
    path_text = request.args.get("path", "")
    if not path_text:
        return "no path", 400
    candidate = Path(path_text).expanduser()
    abs_path = candidate.resolve() if candidate.is_absolute() else (BASE_DIR / candidate).resolve()
    allowed_roots = [BASE_DIR] + SOURCE_ROOTS
    if not any(path_within(abs_path, root.resolve()) for root in allowed_roots):
        return "forbidden", 403
    if not abs_path.exists() or not abs_path.is_file():
        return "not found", 404
    return send_file(abs_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
