import argparse
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

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
CONFIG_PATH = Path(os.environ.get("TAGGER_CONFIG", "config.yaml")).expanduser().resolve()
STATE_DB = Path(os.environ.get("TAGGER_DB", str(STATE_DIR / "state.sqlite3")))
LABELS_PATH = Path(os.environ.get("TAGGER_LABELS", str(STATE_DIR / "labels.jsonl")))
SAMPLE_LABELS_PATH = Path("data/notebook/labels.jsonl")
SAMPLE_STATE_DB = Path("data/notebook/state.sqlite3")
SAMPLE_PATH = Path("data/notebook/sample.jsonl")
SAMPLE_MODE = False
_, _, _source_roots = ml_pipeline.resolve_screenshot_records(CONFIG_PATH)
SOURCE_ROOTS = [Path(root) for root in _source_roots]
DEFAULT_STATE_DB = STATE_DB
DEFAULT_LABELS_PATH = LABELS_PATH
DEFAULT_SOURCE_ROOTS = list(SOURCE_ROOTS)

_CACHE_LOCK = threading.Lock()
_LABELS_CACHE: dict[str, list[str]] = {}
_LABELS_MTIME_NS: int | None = None


class AccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return " - - [" not in message or "HTTP/" not in message


def resolve_sample_path() -> Path:
    candidate = SAMPLE_PATH.expanduser()
    return candidate.resolve() if candidate.is_absolute() else (BASE_DIR / candidate).resolve()


def set_sample_mode(enabled: bool) -> None:
    global SAMPLE_MODE, STATE_DB, LABELS_PATH, SOURCE_ROOTS, _LABELS_CACHE, _LABELS_MTIME_NS
    SAMPLE_MODE = enabled
    if enabled:
        STATE_DB = SAMPLE_STATE_DB
        LABELS_PATH = SAMPLE_LABELS_PATH
        SOURCE_ROOTS = list(DEFAULT_SOURCE_ROOTS)
    else:
        STATE_DB = DEFAULT_STATE_DB
        LABELS_PATH = DEFAULT_LABELS_PATH
        SOURCE_ROOTS = list(DEFAULT_SOURCE_ROOTS)
    with _CACHE_LOCK:
        _LABELS_CACHE = {}
        _LABELS_MTIME_NS = None


def resolve_state_db() -> Path:
    candidate = STATE_DB.expanduser()
    return candidate.resolve() if candidate.is_absolute() else (BASE_DIR / candidate).resolve()


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


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def init_review_tables(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tag_assignment (
                input_path TEXT NOT NULL,
                tag TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(input_path, tag)
            );
            CREATE INDEX IF NOT EXISTS idx_tag_assignment_tag ON tag_assignment(tag);

            CREATE TABLE IF NOT EXISTS review_event (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_path TEXT NOT NULL,
                before_tags TEXT NOT NULL,
                after_tags TEXT NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
    conn.close()


def sync_tag_assignment(db_path: Path, input_path: str, categories: list[str]) -> None:
    init_review_tables(db_path)
    now = now_iso()
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("DELETE FROM tag_assignment WHERE input_path = ?", (input_path,))
        if categories:
            conn.executemany(
                """
                INSERT INTO tag_assignment(input_path, tag, source, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(input_path, tag, "human", 1.0, now) for tag in categories],
            )
    conn.close()


def log_review_event(
    db_path: Path,
    input_path: str,
    before_tags: list[str],
    after_tags: list[str],
    actor: str = "ui",
) -> None:
    init_review_tables(db_path)
    before_set = set(before_tags)
    after_set = set(after_tags)
    if not before_set and after_set:
        action = "add"
    elif before_set and not after_set:
        action = "clear"
    else:
        action = "update"

    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO review_event(input_path, before_tags, after_tags, actor, action, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                input_path,
                json.dumps(sorted(before_set)),
                json.dumps(sorted(after_set)),
                actor,
                action,
                now_iso(),
            ),
        )
    conn.close()


def labels_mtime_ns() -> int | None:
    path = BASE_DIR / LABELS_PATH
    if not path.exists():
        return None
    return path.stat().st_mtime_ns


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


def load_sample_items() -> list[dict]:
    sample_path = resolve_sample_path()
    if not sample_path.exists():
        return []
    items: list[dict] = []
    for row in read_jsonl(sample_path, strict=True):
        input_path = row.get("input_path")
        if not isinstance(input_path, str) or not input_path:
            raise ValueError(f"invalid input_path in {sample_path}")
        series = row.get("series", "sample")
        source = row.get("source", "sample")
        if not isinstance(series, str) or not series:
            raise ValueError(f"invalid series in {sample_path}")
        if not isinstance(source, str) or not source:
            raise ValueError(f"invalid source in {sample_path}")
        items.append(
            {
                "input_path": input_path,
                "series": series,
                "source": source,
                "cluster": 0,
            }
        )
    return items


def load_items() -> list[dict]:
    if SAMPLE_MODE:
        return load_sample_items()
    return ml_pipeline.get_items(db_path=resolve_state_db())


def load_all() -> list[dict]:
    labels = load_labels_cached()
    merged: list[dict] = []
    for item in load_items():
        row = dict(item)
        input_path = row.get("input_path")
        if input_path in labels:
            row["categories"] = labels[input_path]
        merged.append(row)
    return merged


@app.get("/")
def index():
    return render_template("index.html", sample_mode=SAMPLE_MODE)


@app.get("/api/items")
def get_items():
    selected_cluster = request.args.get("cluster", "")
    payload = []
    for item_idx, item in enumerate(load_all()):
        if selected_cluster:
            cluster_id = item.get("cluster")
            if str(cluster_id) != selected_cluster:
                continue
        row = dict(item)
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
    categories_clean = sorted({entry.strip() for entry in categories if entry.strip()})
    items = load_items()
    if idx < 0 or idx >= len(items):
        return jsonify({"error": "out of range"}), 404
    input_path = items[idx].get("input_path")
    if not input_path:
        return jsonify({"error": "missing input_path"}), 500
    labels_by_path = dict(load_labels_cached())
    previous_raw = labels_by_path.get(input_path, [])
    previous = sorted({entry.strip() for entry in previous_raw if entry.strip()})
    if previous == categories_clean:
        response = dict(items[idx])
        response["categories"] = previous
        response["_idx"] = idx
        return jsonify(response)

    labels_by_path[input_path] = categories_clean
    write_labels_file(labels_by_path)
    db_path = resolve_state_db()
    sync_tag_assignment(db_path=db_path, input_path=input_path, categories=categories_clean)
    log_review_event(
        db_path=db_path,
        input_path=input_path,
        before_tags=previous,
        after_tags=categories_clean,
    )
    response = dict(items[idx])
    response["categories"] = categories_clean
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
    if SAMPLE_MODE:
        return jsonify({"started": False, "disabled": True})
    started = ml_pipeline.start_job(config_path=CONFIG_PATH, db_path=resolve_state_db())
    return jsonify({"started": started})


@app.get("/api/ml/status")
def ml_status():
    if SAMPLE_MODE:
        return jsonify({"stage": "disabled", "disabled": True})
    return jsonify(ml_pipeline.get_status(config_path=CONFIG_PATH, db_path=resolve_state_db()))


@app.get("/api/ml/clusters")
def ml_clusters():
    if SAMPLE_MODE:
        return jsonify([])
    return jsonify(ml_pipeline.get_clusters(db_path=resolve_state_db()))


@app.post("/api/ml/ocr")
def ml_ocr():
    if SAMPLE_MODE:
        return jsonify({"disabled": True})
    return jsonify(ml_pipeline.sync_ocr_db(config_path=CONFIG_PATH, db_path=resolve_state_db()))


@app.get("/api/review/summary")
def review_summary():
    db_path = resolve_state_db()
    init_review_tables(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS events,
            SUM(CASE WHEN action = 'add' THEN 1 ELSE 0 END) AS adds,
            SUM(CASE WHEN action = 'update' THEN 1 ELSE 0 END) AS updates,
            SUM(CASE WHEN action = 'clear' THEN 1 ELSE 0 END) AS clears,
            SUM(CASE WHEN action = 'noop' THEN 1 ELSE 0 END) AS noops
        FROM review_event
        """,
    ).fetchone()
    top_tags = conn.execute(
        """
        SELECT tag, COUNT(*) AS c
        FROM tag_assignment
        GROUP BY tag
        ORDER BY c DESC, tag ASC
        LIMIT 20
        """,
    ).fetchall()
    conn.close()
    events, adds, updates, clears, noops = row if row else (0, 0, 0, 0, 0)
    return jsonify(
        {
            "events": int(events or 0),
            "adds": int(adds or 0),
            "updates": int(updates or 0),
            "clears": int(clears or 0),
            "noops": int(noops or 0),
            "top_tags": [{"tag": tag, "count": int(count)} for tag, count in top_tags],
        }
    )


@app.get("/api/review/events")
def review_events():
    raw_limit = request.args.get("limit", "50")
    try:
        limit = int(raw_limit)
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400
    if limit < 1 or limit > 500:
        return jsonify({"error": "limit must be in [1, 500]"}), 400

    db_path = resolve_state_db()
    init_review_tables(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT id, input_path, before_tags, after_tags, actor, action, created_at
        FROM review_event
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    payload = []
    for event_id, input_path, before_tags, after_tags, actor, action, created_at in rows:
        payload.append(
            {
                "id": int(event_id),
                "input_path": input_path,
                "before_tags": json.loads(before_tags),
                "after_tags": json.loads(after_tags),
                "actor": actor,
                "action": action,
                "created_at": created_at,
            }
        )
    return jsonify(payload)


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
    parser = argparse.ArgumentParser(description="Run tagger server.")
    parser.add_argument("--sample", action="store_true", help="Run labeling-only sample mode.")
    args = parser.parse_args()
    set_sample_mode(args.sample)

    access_log = os.environ.get("TAGGER_ACCESS_LOG", "").lower() in {"1", "true", "yes", "on"}
    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(logging.INFO)
    if not access_log and not any(
        isinstance(existing_filter, AccessLogFilter) for existing_filter in werkzeug_logger.filters
    ):
        werkzeug_logger.addFilter(AccessLogFilter())
    debug = os.environ.get("TAGGER_DEBUG", "1").lower() in {"1", "true", "yes", "on"}
    app.run(debug=debug, port=5000)


if __name__ == "__main__":
    main()
