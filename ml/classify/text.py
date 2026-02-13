#!/usr/bin/env python3

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

LABELS_PATH = Path("data/labels.jsonl")
OCR_ROOT = Path("data/ocr")
MODEL_OUT = Path("data/models/text.joblib")
TEST_SIZE = 0.2
SEED = 42
PREDICTION_THRESHOLD = 0.2


def read_labels():
    rows = []
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_dataset(rows):
    texts = []
    labels = []
    for row in rows:
        image_path = Path(row["input_path"])
        ocr_path = OCR_ROOT / image_path.parent.name / (image_path.stem + ".txt")
        text = ocr_path.read_text(encoding="utf-8")
        texts.append(" ".join(text.lower().split()))
        labels.append(row["categories"])
    return texts, labels


def split_indices(number_items):
    split_index = int(number_items * (1 - TEST_SIZE))
    indices = np.random.RandomState(SEED).permutation(number_items)
    return indices[:split_index], indices[split_index:]


def ensure_label_coverage(targets, train_indices, test_indices):
    train_set = set(train_indices.tolist())
    test_set = set(test_indices.tolist())
    for label_index in range(targets.shape[1]):
        if targets[list(train_set), label_index].sum() > 0:
            continue
        candidates = [index for index in test_set if targets[index, label_index] == 1]
        if candidates:
            move_index = candidates[0]
            test_set.remove(move_index)
            train_set.add(move_index)
    return np.array(sorted(train_set)), np.array(sorted(test_set))


def train():
    print("Loading labels")
    texts, labels = build_dataset(read_labels())
    print(f"Samples: {len(texts)}")
    binarizer = MultiLabelBinarizer()
    targets = binarizer.fit_transform(labels)
    print(f"Labels: {len(binarizer.classes_)}")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_df=0.95)
    print("Vectorizing OCR")
    features = vectorizer.fit_transform(texts)

    print("Splitting train/test")
    train_indices, test_indices = split_indices(targets.shape[0])
    train_indices, test_indices = ensure_label_coverage(targets, train_indices, test_indices)
    train_features = features[train_indices]
    test_features = features[test_indices]
    train_targets = targets[train_indices]
    test_targets = targets[test_indices]

    print("Training classifier")
    classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000, solver="liblinear"))
    classifier.fit(train_features, train_targets)
    probabilities = classifier.predict_proba(test_features)
    predictions = (probabilities >= PREDICTION_THRESHOLD).astype(int)

    print("Evaluating")
    print("F1 micro:", f1_score(test_targets, predictions, average="micro", zero_division=0))
    print("F1 macro:", f1_score(test_targets, predictions, average="macro", zero_division=0))
    print("F1 samples:", f1_score(test_targets, predictions, average="samples", zero_division=0))

    print("Saving model")
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "classifier": classifier, "binarizer": binarizer},
        MODEL_OUT,
    )


if __name__ == "__main__":
    train()
