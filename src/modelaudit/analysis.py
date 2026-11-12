"""Deterministic statistical evidence; never delegates arithmetic to the LLM."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_predictions


def _safe_float(v: float) -> float:
    return round(float(v), 6)


def compute_ece(frame: pd.DataFrame, n_bins: int = 10) -> dict:
    """Top-label ECE. Confidence means P(predicted class), not raw logits."""
    conf = frame["confidence"].to_numpy(dtype=float)
    correct = frame["y_true"].eq(frame["y_pred"]).to_numpy(dtype=float)
    bin_ids = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    bins = []
    total = len(frame)
    ece = 0.0
    for idx in range(n_bins):
        mask = bin_ids == idx
        count = int(mask.sum())
        avg_confidence = float(conf[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        if count:
            ece += count / total * abs(accuracy - avg_confidence)
        bins.append({
            "lower": round(idx / n_bins, 2),
            "upper": round((idx + 1) / n_bins, 2),
            "count": count,
            "accuracy": _safe_float(accuracy) if count else None,
            "confidence": _safe_float(avg_confidence) if count else None,
        })
    return {"ece": _safe_float(ece), "bins": bins}


def per_class_metrics(frame: pd.DataFrame, classes: list[str]) -> list[dict]:
    results = []
    for label in classes:
        actual = frame["y_true"].eq(label)
        predicted = frame["y_pred"].eq(label)
        tp = int((actual & predicted).sum())
        support = int(actual.sum())
        predicted_count = int(predicted.sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results.append({
            "class": label, "support": support, "predicted": predicted_count,
            "precision": _safe_float(precision), "recall": _safe_float(recall),
            "f1": _safe_float(f1),
        })
    return results


def performance_analysis(df: pd.DataFrame) -> dict:
    classes = sorted(set(df["y_true"]).union(set(df["y_pred"])))
    if len(classes) > 100:
        raise ValueError("Maximum supported class count is 100")
    splits = {}
    for split, frame in df.groupby("split", sort=True):
        correct = frame["y_true"].eq(frame["y_pred"])
        class_rows = per_class_metrics(frame, classes)
        actual_support = sum(row["support"] for row in class_rows)
        weighted_f1 = sum(row["support"] * row["f1"] for row in class_rows) / actual_support
        matrix = [
            [int(((frame["y_true"] == a) & (frame["y_pred"] == b)).sum()) for b in classes]
            for a in classes
        ]
        split_summary = {
            "n": int(len(frame)),
            "accuracy": _safe_float(correct.mean()),
            "macro_f1": _safe_float(np.mean([r["f1"] for r in class_rows])),
            "weighted_f1": _safe_float(weighted_f1),
            "ece": compute_ece(frame),
            "per_class": class_rows,
            "confusion_matrix": matrix,
        }
        if "group" in frame.columns:
            split_summary["groups"] = [
                {"group": str(group), "n": int(len(chunk)),
                 "accuracy": _safe_float(chunk["y_true"].eq(chunk["y_pred"]).mean())}
                for group, chunk in frame.groupby("group", sort=True) if len(chunk) >= 5
            ]
        splits[str(split)] = split_summary
    return {"total_rows": int(len(df)), "classes": classes, "splits": splits}


def hard_examples(df: pd.DataFrame, split: str = "val", limit: int = 10) -> list[dict]:
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    frame = df[(df["split"] == split) & (df["y_true"] != df["y_pred"])]
    frame = frame.sort_values("confidence", ascending=False).head(limit)
    cols = ["sample_id", "split", "y_true", "y_pred", "confidence"]
    if "group" in frame.columns:
        cols.append("group")
    if "image_path" in frame.columns:
        cols.append("image_path")
    result = frame[cols].to_dict(orient="records")
    for row in result:
        row["confidence"] = _safe_float(row["confidence"])
    return result


def leakage_analysis(df: pd.DataFrame, csv_path: str | Path) -> dict:
    """Detect exact cross-split IDs and optional binary-identical local image files."""
    sample_splits: dict[str, set[str]] = defaultdict(set)
    for row in df[["sample_id", "split"]].itertuples(index=False):
        sample_splits[str(row.sample_id)].add(str(row.split))
    duplicate_ids = [
        {"sample_id": sample_id, "splits": sorted(splits)}
        for sample_id, splits in sample_splits.items() if len(splits) > 1
    ]

    hash_splits: dict[str, set[str]] = defaultdict(set)
    hash_examples: dict[str, list[str]] = defaultdict(list)
    missing = 0
    blocked = 0
    root = Path(csv_path).expanduser().resolve().parent
    if "image_path" in df.columns:
        seen_paths = {}
        for row in df[["image_path", "split"]].itertuples(index=False):
            raw = str(row.image_path)
            if not raw or raw == "<NA>":
                continue
            p = Path(raw)
            try:
                resolved = (root / p).resolve()
                resolved.relative_to(root)  # prevent reading files outside dataset directory
            except (OSError, ValueError):
                blocked += 1
                continue
            if not resolved.is_file():
                missing += 1
                continue
            if resolved.stat().st_size > 50_000_000:
                blocked += 1
                continue
            key = str(resolved)
            if key not in seen_paths:
                h = hashlib.sha256()
                with resolved.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        h.update(chunk)
                seen_paths[key] = h.hexdigest()
            digest = seen_paths[key]
            hash_splits[digest].add(str(row.split))
            if len(hash_examples[digest]) < 3 and raw not in hash_examples[digest]:
                hash_examples[digest].append(raw)
    duplicate_images = [
        {"sha256_prefix": digest[:16], "splits": sorted(splits), "paths": hash_examples[digest]}
        for digest, splits in hash_splits.items() if len(splits) > 1
    ]
    return {
        "cross_split_sample_ids": sorted(duplicate_ids, key=lambda x: x["sample_id"])[:30],
        "cross_split_sample_id_count": len(duplicate_ids),
        "cross_split_identical_images": duplicate_images[:30],
        "cross_split_identical_image_count": len(duplicate_images),
        "missing_images": missing,
        "blocked_image_paths": blocked,
        "note": "Repeated IDs require investigation; identical file hashes show repeated bytes, not necessarily the reason for duplication. Hash checks run only on accessible images within the CSV folder.",
    }


def diagnose(performance: dict, leakage: dict, hard: list[dict]) -> list[dict]:
    issues = []
    splits = performance["splits"]
    if "train" in splits and "val" in splits:
        gap = splits["train"]["accuracy"] - splits["val"]["accuracy"]
        if gap > 0.10:
            issues.append({"severity": "high", "type": "generalization_gap", "evidence": f"train-val accuracy gap = {gap:.3f}", "suggestion": "Inspect augmentation, regularization, split quality and class imbalance; the gap alone does not prove overfitting."})
    eval_split = "val" if "val" in splits else "test" if "test" in splits else "train"
    result = splits[eval_split]
    if result["ece"]["ece"] > 0.12:
        issues.append({"severity": "medium", "type": "confidence_miscalibration", "evidence": f"{eval_split} ECE = {result['ece']['ece']:.3f}", "suggestion": "Evaluate temperature scaling on a separate calibration split; report ECE alongside accuracy."})
    weak = [x for x in result["per_class"] if x["support"] >= 10 and x["recall"] < .6]
    if weak:
        names = ", ".join(x["class"] for x in weak[:5])
        issues.append({"severity": "medium", "type": "low_recall_classes", "evidence": f"{eval_split} recall < 0.60 for: {names}", "suggestion": "Inspect the confusion matrix and class-level examples, then evaluate targeted data collection or reweighting."})
    confident_errors = [x for x in hard if x["confidence"] >= .9]
    if confident_errors:
        issues.append({"severity": "medium", "type": "high_confidence_errors", "evidence": f"At least {len(confident_errors)} of the sampled top errors have confidence >= 0.9", "suggestion": "Review the sampled labels and visual conditions; look for systematic class confusion."})
    if leakage["cross_split_sample_id_count"] or leakage["cross_split_identical_image_count"]:
        issues.append({"severity": "high", "type": "possible_split_leakage", "evidence": f"{leakage['cross_split_sample_id_count']} repeated IDs; {leakage['cross_split_identical_image_count']} cross-split identical image hashes", "suggestion": "Audit split construction and provenance before trusting validation scores; repeated IDs alone are not conclusive."})
    if not issues:
        issues.append({"severity": "info", "type": "no_threshold_alerts", "evidence": "No built-in rule crossed its threshold", "suggestion": "Review per-class results and run external evaluations; no alerts does not imply the model is safe or correct."})
    return issues


def full_audit(csv_path: str | Path) -> dict:
    df = load_predictions(csv_path)
    performance = performance_analysis(df)
    leakage = leakage_analysis(df, csv_path)
    eval_split = "val" if "val" in performance["splits"] else "test" if "test" in performance["splits"] else "train"
    hard = hard_examples(df, eval_split, 20)
    issues = diagnose(performance, leakage, hard)
    return {"dataset": str(Path(csv_path).resolve()), "performance": performance,
            "leakage": leakage, "hard_examples": hard, "issues": issues}
