import argparse
import importlib
import json
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_activity = importlib.import_module("image_activity.activity")
list_image_paths = _activity.list_image_paths
load_config = _activity.load_config

MODEL_NAME = "openai/clip-vit-base-patch32"
STATUS_FILE = "ml_status.json"
ITEMS_FILE = "items.jsonl"
EMBEDDINGS_FILE = "clip_embeddings.npy"
CLUSTERS_FILE = "clusters.json"

_JOB_LOCK = threading.Lock()
_JOB_THREAD: threading.Thread | None = None


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def update_status(status_path: Path, **fields: object) -> dict:
    payload = read_json(status_path, default={})
    payload.update(fields)
    payload["updated_at"] = now_iso()
    write_json(status_path, payload)
    return payload


def resolve_screenshot_records(config_path: Path) -> tuple[list[dict], list[dict], list[str]]:
    config = load_config(str(config_path))
    rows: list[dict] = []
    source_stats: list[dict] = []
    source_roots: list[str] = []
    for series_name, series_config in sorted(config.data.items()):
        if not series_name.startswith("screenshot"):
            continue
        anti_patterns = config.anti_patterns + series_config.anti_patterns
        for source_name, source_spec in sorted(series_config.sources.items()):
            root = Path(source_spec.path).expanduser().resolve()
            source_roots.append(str(root))
            image_paths = list_image_paths(source_spec, config.extensions, anti_patterns)
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


def source_roots(config_path: Path) -> list[str]:
    config = load_config(str(config_path))
    roots: list[str] = []
    for series_name, series_config in sorted(config.data.items()):
        if not series_name.startswith("screenshot"):
            continue
        for _, source_spec in sorted(series_config.sources.items()):
            roots.append(str(Path(source_spec.path).expanduser().resolve()))
    return sorted(set(roots))


def default_cluster_count(total: int) -> int:
    if total < 2:
        raise ValueError("Need at least 2 images to cluster.")
    if total <= 20:
        return max(2, total // 2)
    return int(np.clip(round(total**0.5), 8, 128))


def embed_images(
    paths: list[str],
    status_path: Path,
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    del batch_size
    processor = CLIPProcessor.from_pretrained(
        model_name,
        use_fast=False,
        local_files_only=True,
    )
    model = CLIPModel.from_pretrained(
        model_name,
        local_files_only=True,
    ).vision_model
    model.eval()
    model.to("cpu")

    total = len(paths)
    started = time.time()
    vectors: list[np.ndarray] = []
    skipped = 0
    min_size = 10
    for index, path_str in enumerate(paths, start=1):
        with Image.open(path_str) as image:
            if image.size[0] < min_size or image.size[1] < min_size:
                skipped += 1
                vectors.append(np.zeros(768, dtype=np.float32))
                elapsed = max(time.time() - started, 1e-9)
                rate = index / elapsed
                eta_seconds = int((total - index) / rate) if rate > 0 else 0
                update_status(
                    status_path,
                    stage="embedding",
                    processed_images=index,
                    total_images=total,
                    skipped_images=skipped,
                    rate_images_per_second=round(rate, 3),
                    eta_seconds=eta_seconds,
                )
                continue
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            vec = outputs.pooler_output.squeeze(0).cpu().numpy().astype(np.float32)
        vectors.append(vec)

        elapsed = max(time.time() - started, 1e-9)
        rate = index / elapsed
        eta_seconds = int((total - index) / rate) if rate > 0 else 0
        update_status(
            status_path,
            stage="embedding",
            processed_images=index,
            total_images=total,
            skipped_images=skipped,
            rate_images_per_second=round(rate, 3),
            eta_seconds=eta_seconds,
        )

    return np.vstack(vectors).astype(np.float32)


def run(
    config_path: Path,
    state_dir: Path,
    model_name: str = MODEL_NAME,
    batch_size: int = 24,
    cluster_count: int = 0,
) -> dict:
    status_path = state_dir / STATUS_FILE
    items_path = state_dir / ITEMS_FILE
    embeddings_path = state_dir / EMBEDDINGS_FILE
    clusters_path = state_dir / CLUSTERS_FILE

    state_dir.mkdir(parents=True, exist_ok=True)
    update_status(
        status_path,
        stage="scanning",
        started_at=now_iso(),
        model=model_name,
        processed_images=0,
        total_images=0,
        rate_images_per_second=0.0,
        eta_seconds=0,
    )

    rows, source_stats, source_roots = resolve_screenshot_records(config_path)
    total_images = len(rows)
    update_status(
        status_path,
        stage="scanning",
        source_stats=source_stats,
        source_roots=source_roots,
        total_images=total_images,
    )
    if total_images < 2:
        raise ValueError("Found fewer than 2 screenshot images.")

    paths = [row["input_path"] for row in rows]
    embeddings = embed_images(paths, status_path, model_name=model_name, batch_size=batch_size)
    np.save(embeddings_path, embeddings)

    k = cluster_count if cluster_count > 0 else default_cluster_count(total_images)
    update_status(
        status_path,
        stage="clustering",
        cluster_count=k,
    )
    clusterer = MiniBatchKMeans(
        n_clusters=k,
        random_state=0,
        batch_size=max(256, min(4096, k * 16)),
        n_init="auto",
    )
    labels = clusterer.fit_predict(embeddings)
    counts = Counter(int(label) for label in labels)

    for idx, label in enumerate(labels):
        rows[idx]["cluster"] = int(label)

    write_jsonl(items_path, rows)
    cluster_rows = [
        {"id": int(cluster_id), "count": int(count)}
        for cluster_id, count in sorted(counts.items(), key=lambda item: item[0])
    ]
    write_json(
        clusters_path,
        {
            "cluster_count": k,
            "clusters": cluster_rows,
            "created_at": now_iso(),
        },
    )
    return update_status(
        status_path,
        stage="done",
        processed_images=total_images,
        total_images=total_images,
        rate_images_per_second=0.0,
        eta_seconds=0,
    )


def _run_job(config_path: Path, state_dir: Path, model_name: str, batch_size: int) -> None:
    status_path = state_dir / STATUS_FILE
    try:
        run(
            config_path=config_path,
            state_dir=state_dir,
            model_name=model_name,
            batch_size=batch_size,
        )
    except Exception as exc:
        update_status(status_path, stage="error", error=str(exc))


def start_job(config_path: Path, state_dir: Path, model_name: str = MODEL_NAME) -> bool:
    global _JOB_THREAD
    with _JOB_LOCK:
        status_path = state_dir / STATUS_FILE
        status = read_json(status_path, default={})
        if status.get("stage") in {"scanning", "embedding", "clustering", "done"}:
            return False
        if _JOB_THREAD is not None and _JOB_THREAD.is_alive():
            return False
        thread = threading.Thread(
            target=_run_job,
            kwargs={
                "config_path": config_path,
                "state_dir": state_dir,
                "model_name": model_name,
                "batch_size": 24,
            },
            daemon=True,
        )
        thread.start()
        _JOB_THREAD = thread
        return True


def get_status(config_path: Path, state_dir: Path) -> dict:
    status_path = state_dir / STATUS_FILE
    payload = read_json(status_path, default={"stage": "idle"})
    if "source_roots" not in payload:
        payload["source_roots"] = source_roots(config_path)
    if "source_stats" not in payload:
        payload["source_stats"] = []
    return payload


def get_clusters(state_dir: Path) -> list[dict]:
    clusters_path = state_dir / CLUSTERS_FILE
    payload = read_json(clusters_path, default={})
    return payload.get("clusters", [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLIP embeddings and clusters for www UI.")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-s", "--state", default="www/state")
    parser.add_argument("-m", "--model", default=MODEL_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(config_path=Path(args.config), state_dir=Path(args.state), model_name=args.model)


if __name__ == "__main__":
    main()
