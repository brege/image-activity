import json
import os
import threading
from pathlib import Path
from typing import cast

from flask import Flask, jsonify, render_template, request, send_file

from bruki.server import ml as ml_pipeline

APP_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(APP_DIR),
    static_folder=str(APP_DIR),
    static_url_path="",
)
BASE_DIR = Path(os.environ.get("TAGGER_BASE", ".")).resolve()
STATE_DIR = Path(os.environ.get("TAGGER_STATE_DIR", "data/server"))
STATE_PATH = (BASE_DIR / STATE_DIR).resolve()
CONFIG_PATH = Path(os.environ.get("TAGGER_CONFIG", "config.yaml")).expanduser().resolve()
JSONL_GLOB = os.environ.get("TAGGER_JSONL", str(STATE_DIR / "items.jsonl"))
LABELS_PATH = Path(os.environ.get("TAGGER_LABELS", str(STATE_DIR / "labels.jsonl")))
_, _, _source_roots = ml_pipeline.resolve_screenshot_records(CONFIG_PATH)
SOURCE_ROOTS = [Path(root) for root in _source_roots]

_CACHE_LOCK = threading.Lock()
_ITEMS_CACHE: list[dict] = []
_ITEMS_FINGERPRINT: tuple[tuple[str, int, int], ...] | None = None
_LABELS_CACHE: dict[str, list[str]] = {}
_LABELS_MTIME_NS: int | None = None


def read_jsonl(path: Path, strict: bool = True) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if strict:
                    raise ValueError(f"invalid JSONL in {path}:{line_no}") from None
    return rows


def item_paths() -> list[Path]:
    paths = (Path(str(path)) for path in BASE_DIR.glob(JSONL_GLOB))
    return cast(list[Path], sorted(paths, key=str))


def item_fingerprint(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    fingerprint: list[tuple[str, int, int]] = []
    for path in paths:
        if not path.exists():
            continue
        stat = path.stat()
        fingerprint.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(fingerprint)


def labels_mtime_ns() -> int | None:
    path = BASE_DIR / LABELS_PATH
    if not path.exists():
        return None
    return path.stat().st_mtime_ns


def load_items_cached() -> list[dict]:
    global _ITEMS_CACHE, _ITEMS_FINGERPRINT
    paths = item_paths()
    fingerprint = item_fingerprint(paths)
    with _CACHE_LOCK:
        if _ITEMS_FINGERPRINT == fingerprint:
            return _ITEMS_CACHE
        rows: list[dict] = []
        for jsonl_path in paths:
            for row in read_jsonl(jsonl_path):
                row["_src"] = str(jsonl_path)
                rows.append(row)
        _ITEMS_CACHE = rows
        _ITEMS_FINGERPRINT = fingerprint
        return _ITEMS_CACHE


def read_labels_file() -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for row in read_jsonl(BASE_DIR / LABELS_PATH, strict=False):
        input_path = row.get("input_path")
        if input_path:
            labels[input_path] = row.get("categories", [])
    return labels


def load_labels_cached() -> dict[str, list[str]]:
    global _LABELS_CACHE, _LABELS_MTIME_NS
    mtime_ns = labels_mtime_ns()
    with _CACHE_LOCK:
        if _LABELS_MTIME_NS == mtime_ns:
            return _LABELS_CACHE
        _LABELS_CACHE = read_labels_file()
        _LABELS_MTIME_NS = mtime_ns
        return _LABELS_CACHE


def write_labels_file(labels_by_path: dict[str, list[str]]) -> None:
    global _LABELS_CACHE, _LABELS_MTIME_NS
    labels_path = BASE_DIR / LABELS_PATH
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = labels_path.with_suffix(labels_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for input_path, categories in sorted(labels_by_path.items()):
            handle.write(json.dumps({"input_path": input_path, "categories": categories}) + "\n")
    tmp_path.replace(labels_path)
    with _CACHE_LOCK:
        _LABELS_CACHE = {}
        _LABELS_MTIME_NS = None


def load_all() -> list[dict]:
    items = load_items_cached()
    labels = load_labels_cached()
    merged: list[dict] = []
    for item in items:
        row = dict(item)
        input_path = row.get("input_path")
        if input_path in labels:
            row["categories"] = labels[input_path]
        merged.append(row)
    return merged


def strip_src(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "_src"}


@app.get("/")
def index():
    return render_template("index.html")


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
    items = load_items_cached()
    if idx < 0 or idx >= len(items):
        return jsonify({"error": "out of range"}), 404
    input_path = items[idx].get("input_path")
    if not input_path:
        return jsonify({"error": "missing input_path"}), 500
    labels_by_path = dict(load_labels_cached())
    labels_by_path[input_path] = categories
    write_labels_file(labels_by_path)
    response = strip_src(items[idx])
    response["categories"] = categories
    response["_idx"] = idx
    return jsonify(response)


@app.post("/api/purge")
def purge_labels():
    items = load_all()
    valid_paths = {item["input_path"] for item in items if "input_path" in item}
    labels_by_path = dict(load_labels_cached())
    before_count = len(labels_by_path)
    labels_by_path = {
        input_path: categories
        for input_path, categories in labels_by_path.items()
        if input_path in valid_paths
    }
    write_labels_file(labels_by_path)
    after_count = len(labels_by_path)
    return jsonify({"removed": before_count - after_count, "remaining": after_count})


@app.get("/api/purge-preview")
def purge_preview():
    items = load_all()
    valid_paths = {item["input_path"] for item in items if "input_path" in item}
    labels_by_path = load_labels_cached()
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


def main() -> None:
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
