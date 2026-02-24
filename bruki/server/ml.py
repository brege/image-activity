import hashlib
import importlib
import json
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

from bruki.config import load_config, resolve_paths

MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_EMBED_DIM = 512

_JOB_LOCK = threading.Lock()
_JOB_THREAD: threading.Thread | None = None


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def migrate_clip_embedding_schema(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(clip_embedding)").fetchall()}
    if "valid" not in columns:
        try:
            conn.execute("ALTER TABLE clip_embedding ADD COLUMN valid INTEGER NOT NULL DEFAULT 1")
        except sqlite3.OperationalError as exc:
            if "duplicate column name: valid" not in str(exc).lower():
                raise


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ml_status (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS clip_item (
                input_path TEXT PRIMARY KEY,
                series TEXT NOT NULL,
                source TEXT NOT NULL,
                cluster INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS clip_cluster (
                cluster_id INTEGER PRIMARY KEY,
                count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS clip_embedding (
                input_path TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                valid INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS ocr_doc (
                input_path TEXT PRIMARY KEY,
                text TEXT NOT NULL
            );
            """
        )
        migrate_clip_embedding_schema(conn)
    conn.close()


def read_status(db_path: Path, default: dict | None = None) -> dict:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT payload FROM ml_status WHERE id = 1").fetchone()
    conn.close()
    if row is None:
        return {} if default is None else default
    return json.loads(row[0])


def update_status(db_path: Path, **fields: object) -> dict:
    payload = read_status(db_path, default={})
    payload.update(fields)
    payload["updated_at"] = now_iso()
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO ml_status(id, payload) VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET payload = excluded.payload
            """,
            (json.dumps(payload, sort_keys=True),),
        )
    conn.close()
    return payload


def resolve_screenshot_records(config_path: Path) -> tuple[list[dict], list[dict], list[str]]:
    config = load_config(str(config_path))
    rows: list[dict] = []
    source_stats: list[dict] = []
    source_roots: list[str] = []
    for series_name, source_name, image_paths in resolve_paths(config, prefix="screenshot"):
        source_spec = config.data[series_name].sources[source_name]
        root = Path(source_spec.path).expanduser().resolve()
        source_roots.append(str(root))
        source_stats.append(
            {
                "series": series_name,
                "source": source_name,
                "root": str(root),
                "count": len(image_paths),
            }
        )
        for path in image_paths:
            rows.append(
                {
                    "series": series_name,
                    "source": source_name,
                    "input_path": str(path),
                }
            )
    return rows, source_stats, sorted(set(source_roots))


def records_signature(rows: list[dict]) -> str:
    values: list[str] = []
    for row in rows:
        input_path = row["input_path"]
        try:
            stat = Path(input_path).stat()
            values.append(f"{input_path}\t{int(stat.st_mtime_ns)}\t{int(stat.st_size)}")
        except OSError:
            values.append(f"{input_path}\t0\t0")
    joined = "\n".join(sorted(values))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def ocr_image(path: Path, psm: int = 6, oem: int = 3) -> str:
    from PIL import Image

    pytesseract = importlib.import_module("pytesseract")
    with Image.open(path) as image:
        if image.size[0] < 10 or image.size[1] < 10:
            return ""
        return pytesseract.image_to_string(
            image.convert("RGB"),
            lang="eng",
            config=f"--psm {psm} --oem {oem}",
        )


def sync_ocr_db(
    config_path: Path,
    db_path: Path,
    paths: list[str] | None = None,
    progress: Callable[[int, int, float, int], None] | None = None,
) -> dict:
    if paths is None:
        records, _, _ = resolve_screenshot_records(config_path)
        paths = list(dict.fromkeys(record["input_path"] for record in records))
    else:
        paths = list(dict.fromkeys(paths))

    tqdm = importlib.import_module("tqdm").tqdm
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    known_paths = {input_path for (input_path,) in conn.execute("SELECT input_path FROM ocr_doc")}
    new_docs: list[tuple[str, str]] = []
    skipped = 0
    total = len(paths)
    status_every_images = 10
    status_every_seconds = 0.5
    last_status_time = 0.0
    with tqdm(total=total, desc="ocr") as progress_bar:
        for index, input_path in enumerate(paths, start=1):
            if input_path in known_paths:
                progress_bar.update(1)
            else:
                try:
                    text = ocr_image(Path(input_path)).lower()
                except OSError:
                    skipped += 1
                    text = ""
                new_docs.append((input_path, text))
                progress_bar.update(1)

            now = time.time()
            should_report = (
                index == total
                or index % status_every_images == 0
                or (now - last_status_time) >= status_every_seconds
            )
            if should_report:
                rate = progress_bar.format_dict.get("rate") or 0.0
                remaining = max(total - progress_bar.n, 0)
                eta_seconds = int(remaining / rate) if rate > 0 else 0
                progress_bar.set_postfix(rate=f"{rate:.3f}/s", eta=eta_seconds, skipped=skipped)
                if progress is not None:
                    progress(index, total, float(rate), eta_seconds)
                last_status_time = now

    if new_docs:
        with conn:
            conn.executemany("INSERT INTO ocr_doc(input_path, text) VALUES (?, ?)", new_docs)

    with conn:
        conn.execute("CREATE TEMP TABLE current_path(input_path TEXT PRIMARY KEY)")
        conn.executemany(
            "INSERT INTO current_path(input_path) VALUES (?)",
            [(input_path,) for input_path in paths],
        )
        deleted_rows = conn.execute(
            """
            DELETE FROM ocr_doc
            WHERE NOT EXISTS (
                SELECT 1 FROM current_path
                WHERE current_path.input_path = ocr_doc.input_path
            )
            """
        ).rowcount
        conn.execute("DROP TABLE current_path")

    total_rows = conn.execute("SELECT COUNT(*) FROM ocr_doc").fetchone()[0]
    conn.close()
    return {
        "resolved_paths": total,
        "new_rows": len(new_docs),
        "deleted_rows": int(deleted_rows),
        "skipped_rows": skipped,
        "total_rows": int(total_rows),
    }


def default_cluster_count(total: int) -> int:
    if total < 2:
        raise ValueError("Need at least 2 images to cluster.")
    if total <= 20:
        return max(2, total // 2)
    return int(np.clip(round(total**0.5), 8, 128))


def embed_images(
    paths: list[str],
    path_stats: dict[str, tuple[int, int]],
    db_path: Path,
    model_name: str,
    batch_size: int,
    cached_images: int,
) -> tuple[int, int]:
    from PIL import Image

    torch = importlib.import_module("torch")
    tqdm = importlib.import_module("tqdm").tqdm
    transformers = importlib.import_module("transformers")
    clip_processor = transformers.CLIPProcessor
    clip_model = transformers.CLIPModel

    del batch_size
    total = len(paths)
    if total == 0:
        update_status(
            db_path,
            stage="embedding",
            processed_images=0,
            total_images=0,
            skipped_images=0,
            cached_images=cached_images,
            rate_images_per_second=0.0,
            eta_seconds=0,
        )
        return 0, 0

    processor = clip_processor.from_pretrained(
        model_name,
        use_fast=False,
        local_files_only=True,
    )
    model = clip_model.from_pretrained(
        model_name,
        local_files_only=True,
    )
    model.eval()
    model.cpu()
    embed_dim = int(model.visual_projection.out_features)
    if embed_dim != CLIP_EMBED_DIM:
        raise ValueError(f"Unexpected CLIP projection dim: {embed_dim}")

    rows: list[tuple[str, str, int, int, int, bytes, int]] = []
    skipped = 0
    min_size = 10
    status_every_images = 10
    status_every_seconds = 0.5
    last_status_time = 0.0
    with tqdm(total=total, desc="embedding") as progress:
        for index, path_str in enumerate(paths, start=1):
            vector = np.zeros(embed_dim, dtype=np.float32)
            valid = 0
            try:
                with Image.open(path_str) as image:
                    if image.size[0] < min_size or image.size[1] < min_size:
                        skipped += 1
                    else:
                        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
                        with torch.no_grad():
                            outputs = model.vision_model(pixel_values=inputs["pixel_values"])
                            image_embeds = model.visual_projection(outputs.pooler_output)
                        vector = image_embeds.squeeze(0).cpu().numpy().astype(np.float32)
                        norm = float(np.linalg.norm(vector))
                        if np.isfinite(norm) and norm > 0:
                            vector /= norm
                            valid = 1
                        else:
                            skipped += 1
            except OSError:
                skipped += 1

            mtime_ns, size_bytes = path_stats[path_str]
            rows.append(
                (
                    path_str,
                    model_name,
                    mtime_ns,
                    size_bytes,
                    embed_dim,
                    vector.astype(np.float32).tobytes(),
                    valid,
                )
            )

            progress.update(1)
            now = time.time()
            should_report = (
                index == total
                or index % status_every_images == 0
                or (now - last_status_time) >= status_every_seconds
            )
            if should_report:
                rate = progress.format_dict.get("rate") or 0.0
                remaining = max(total - progress.n, 0)
                eta_seconds = int(remaining / rate) if rate > 0 else 0
                progress.set_postfix(skipped=skipped, rate=f"{rate:.3f}/s", eta=eta_seconds)
                update_status(
                    db_path,
                    stage="embedding",
                    processed_images=index,
                    total_images=total,
                    skipped_images=skipped,
                    cached_images=cached_images,
                    rate_images_per_second=round(rate, 3),
                    eta_seconds=eta_seconds,
                )
                last_status_time = now

    init_db(db_path)
    conn = sqlite3.connect(db_path)
    with conn:
        conn.executemany(
            """
            INSERT INTO clip_embedding(input_path, model, mtime_ns, size_bytes, dim, vector, valid)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(input_path) DO UPDATE SET
                model = excluded.model,
                mtime_ns = excluded.mtime_ns,
                size_bytes = excluded.size_bytes,
                dim = excluded.dim,
                vector = excluded.vector,
                valid = excluded.valid
            """,
            rows,
        )
    conn.close()
    return total, skipped


def resolve_embeddings(
    paths: list[str],
    db_path: Path,
    model_name: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    path_stats: dict[str, tuple[int, int]] = {}
    for path in paths:
        try:
            stat = Path(path).stat()
            path_stats[path] = (int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            path_stats[path] = (0, 0)

    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cache_rows = conn.execute(
        "SELECT input_path, model, mtime_ns, size_bytes, dim, vector, valid FROM clip_embedding",
    ).fetchall()
    cached_meta: dict[str, tuple[str, int, int, int, int, bool]] = {}
    for input_path, model, mtime_ns, size_bytes, dim, vector_blob, valid in cache_rows:
        dim_value = int(dim)
        valid_value = int(valid)
        has_nonzero_vector = False
        if dim_value == CLIP_EMBED_DIM:
            vector = np.frombuffer(vector_blob, dtype=np.float32)
            if vector.size == dim_value:
                norm = float(np.linalg.norm(vector))
                has_nonzero_vector = np.isfinite(norm) and norm > 0
        cached_meta[input_path] = (
            model,
            int(mtime_ns),
            int(size_bytes),
            dim_value,
            valid_value,
            has_nonzero_vector,
        )
    needs_embed = [
        path
        for path in paths
        if (
            cached_meta.get(path) is None
            or cached_meta[path][0] != model_name
            or cached_meta[path][1] != path_stats[path][0]
            or cached_meta[path][2] != path_stats[path][1]
            or cached_meta[path][3] != CLIP_EMBED_DIM
            or cached_meta[path][4] not in (0, 1)
            or (cached_meta[path][4] == 1 and not cached_meta[path][5])
        )
    ]
    cached_images = len(paths) - len(needs_embed)

    with conn:
        conn.execute("CREATE TEMP TABLE current_path(input_path TEXT PRIMARY KEY)")
        conn.executemany(
            "INSERT INTO current_path(input_path) VALUES (?)",
            [(input_path,) for input_path in paths],
        )
        deleted_rows = conn.execute(
            """
            DELETE FROM clip_embedding
            WHERE NOT EXISTS (
                SELECT 1 FROM current_path
                WHERE current_path.input_path = clip_embedding.input_path
            )
            """
        ).rowcount
        conn.execute("DROP TABLE current_path")
    conn.close()

    embedded_images, skipped_images = embed_images(
        needs_embed,
        path_stats=path_stats,
        db_path=db_path,
        model_name=model_name,
        batch_size=batch_size,
        cached_images=cached_images,
    )

    conn = sqlite3.connect(db_path)
    embedding_rows = conn.execute(
        "SELECT input_path, dim, vector, valid FROM clip_embedding",
    ).fetchall()
    conn.close()
    vectors_by_path = {}
    valid_by_path = {}
    for input_path, _, vector_blob, valid in embedding_rows:
        vectors_by_path[input_path] = np.frombuffer(vector_blob, dtype=np.float32)
        valid_by_path[input_path] = bool(int(valid))

    vectors = []
    valid_mask: list[bool] = []
    for path in paths:
        vector = vectors_by_path.get(path)
        if vector is None:
            raise ValueError(f"Missing CLIP vector for {path}")
        vectors.append(vector)
        valid_mask.append(valid_by_path[path])
    embeddings = np.vstack(vectors).astype(np.float32)
    valid_array = np.asarray(valid_mask, dtype=bool)
    return (
        embeddings,
        valid_array,
        {
            "embedded_images": embedded_images,
            "cached_images": cached_images,
            "deleted_rows": int(deleted_rows),
            "skipped_images": skipped_images,
            "valid_images": int(valid_array.sum()),
            "invalid_images": int((~valid_array).sum()),
        },
    )


def run(
    config_path: Path,
    db_path: Path,
    model_name: str = MODEL_NAME,
    batch_size: int = 24,
    cluster_count: int = 0,
) -> dict:
    mini_batch_k_means = importlib.import_module("sklearn.cluster").MiniBatchKMeans

    init_db(db_path)
    update_status(
        db_path,
        stage="scanning",
        error="",
        started_at=now_iso(),
        model=model_name,
        processed_images=0,
        total_images=0,
        rate_images_per_second=0.0,
        eta_seconds=0,
    )

    rows, source_stats, source_roots = resolve_screenshot_records(config_path)
    total_images = len(rows)
    signature = records_signature(rows)
    update_status(
        db_path,
        stage="scanning",
        source_stats=source_stats,
        source_roots=source_roots,
        total_images=total_images,
        records_signature=signature,
    )
    if total_images < 2:
        raise ValueError("Found fewer than 2 screenshot images.")

    paths = [row["input_path"] for row in rows]
    embeddings, valid_mask, clip_stats = resolve_embeddings(
        paths,
        db_path=db_path,
        model_name=model_name,
        batch_size=batch_size,
    )

    valid_indices = np.flatnonzero(valid_mask)
    valid_images = int(valid_indices.size)
    if valid_images < 2:
        raise ValueError("Found fewer than 2 valid screenshot images for CLIP clustering.")
    k = cluster_count if cluster_count > 0 else default_cluster_count(valid_images)
    update_status(
        db_path,
        stage="clustering",
        cluster_count=k,
        valid_images=valid_images,
        invalid_images=total_images - valid_images,
    )
    clusterer = mini_batch_k_means(
        n_clusters=k,
        random_state=0,
        batch_size=max(256, min(4096, k * 16)),
        n_init="auto",
    )
    labels = clusterer.fit_predict(embeddings[valid_mask])
    counts = Counter(int(label) for label in labels)

    clip_rows = []
    for label_idx, row_idx in enumerate(valid_indices):
        row = rows[int(row_idx)]
        clip_rows.append(
            (
                row["input_path"],
                row["series"],
                row["source"],
                int(labels[label_idx]),
            )
        )
    cluster_rows = [(int(cluster_id), int(count)) for cluster_id, count in sorted(counts.items())]
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("DELETE FROM clip_item")
        conn.execute("DELETE FROM clip_cluster")
        conn.executemany(
            "INSERT INTO clip_item(input_path, series, source, cluster) VALUES (?, ?, ?, ?)",
            clip_rows,
        )
        conn.executemany(
            "INSERT INTO clip_cluster(cluster_id, count) VALUES (?, ?)",
            cluster_rows,
        )
    conn.close()

    update_status(
        db_path,
        stage="ocr",
        processed_images=0,
        total_images=total_images,
        rate_images_per_second=0.0,
        eta_seconds=0,
    )
    ocr_stats = sync_ocr_db(
        config_path=config_path,
        db_path=db_path,
        paths=paths,
        progress=lambda done, total, rate, eta: update_status(
            db_path,
            stage="ocr",
            processed_images=done,
            total_images=total,
            rate_images_per_second=round(rate, 3),
            eta_seconds=eta,
        ),
    )
    return update_status(
        db_path,
        stage="done",
        processed_images=total_images,
        total_images=total_images,
        cluster_count=k,
        source_stats=source_stats,
        source_roots=source_roots,
        records_signature=signature,
        clip_embedded_images=clip_stats["embedded_images"],
        clip_cached_images=clip_stats["cached_images"],
        clip_deleted_embeddings=clip_stats["deleted_rows"],
        clip_skipped_images=clip_stats["skipped_images"],
        clip_valid_images=clip_stats["valid_images"],
        clip_invalid_images=clip_stats["invalid_images"],
        ocr_new_rows=ocr_stats["new_rows"],
        ocr_deleted_rows=ocr_stats["deleted_rows"],
        ocr_skipped_rows=ocr_stats["skipped_rows"],
        ocr_total_rows=ocr_stats["total_rows"],
        rate_images_per_second=0.0,
        eta_seconds=0,
    )


def _run_job(config_path: Path, db_path: Path, model_name: str, batch_size: int) -> None:
    try:
        run(
            config_path=config_path,
            db_path=db_path,
            model_name=model_name,
            batch_size=batch_size,
        )
    except Exception as exc:
        update_status(db_path, stage="error", error=str(exc))


def start_job(config_path: Path, db_path: Path, model_name: str = MODEL_NAME) -> bool:
    global _JOB_THREAD
    with _JOB_LOCK:
        status = read_status(db_path, default={})
        if _JOB_THREAD is not None and _JOB_THREAD.is_alive():
            return False
        if status.get("stage") == "done":
            rows, _, _ = resolve_screenshot_records(config_path)
            current_signature = records_signature(rows)
            has_validity_stats = "clip_valid_images" in status and "clip_invalid_images" in status
            if status.get("records_signature") == current_signature and has_validity_stats:
                return False
        thread = threading.Thread(
            target=_run_job,
            kwargs={
                "config_path": config_path,
                "db_path": db_path,
                "model_name": model_name,
                "batch_size": 24,
            },
            daemon=True,
        )
        thread.start()
        _JOB_THREAD = thread
        return True


def get_status(config_path: Path, db_path: Path) -> dict:
    payload = read_status(db_path, default={"stage": "idle"})
    if "source_roots" not in payload:
        _, _, roots = resolve_screenshot_records(config_path)
        payload["source_roots"] = roots
    if "source_stats" not in payload:
        payload["source_stats"] = []
    return payload


def get_clusters(db_path: Path) -> list[dict]:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT cluster_id, count FROM clip_cluster ORDER BY cluster_id",
    ).fetchall()
    conn.close()
    return [{"id": int(cluster_id), "count": int(count)} for cluster_id, count in rows]


def get_items(db_path: Path) -> list[dict]:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT input_path, series, source, cluster FROM clip_item ORDER BY input_path",
    ).fetchall()
    conn.close()
    return [
        {
            "input_path": input_path,
            "series": series,
            "source": source,
            "cluster": int(cluster),
        }
        for input_path, series, source, cluster in rows
    ]
