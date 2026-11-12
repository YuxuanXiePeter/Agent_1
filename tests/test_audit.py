from pathlib import Path

import pandas as pd
import pytest

from modelaudit.analysis import compute_ece, full_audit, leakage_analysis, performance_analysis
from modelaudit.data import DataValidationError, load_predictions
from modelaudit.demo import make_demo
from modelaudit.report import AuditEngine


def test_perfect_predictions_have_perfect_metrics():
    frame = pd.DataFrame({"split": ["val"] * 4, "y_true": ["a", "a", "b", "b"],
                          "y_pred": ["a", "a", "b", "b"], "confidence": [1.0] * 4})
    perf = performance_analysis(frame)["splits"]["val"]
    assert perf["accuracy"] == 1.0
    assert perf["macro_f1"] == 1.0
    assert perf["ece"]["ece"] == 0.0
    assert perf["confusion_matrix"] == [[2, 0], [0, 2]]


def test_confidence_validation(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"sample_id": ["1"], "split": ["val"], "y_true": ["car"],
                  "y_pred": ["bus"], "confidence": [1.25]}).to_csv(csv, index=False)
    with pytest.raises(DataValidationError):
        load_predictions(csv)


def test_cross_split_id_leakage(tmp_path):
    csv = tmp_path / "predictions.csv"
    pd.DataFrame({"sample_id": ["same", "same"], "split": ["train", "val"],
                  "y_true": ["car", "car"], "y_pred": ["car", "car"],
                  "confidence": [.9, .9]}).to_csv(csv, index=False)
    data = load_predictions(csv)
    result = leakage_analysis(data, csv)
    assert result["cross_split_sample_id_count"] == 1


def test_cross_split_image_hash(tmp_path):
    (tmp_path / "a.png").write_bytes(b"identical-image-binary")
    (tmp_path / "b.png").write_bytes(b"identical-image-binary")
    csv = tmp_path / "predictions.csv"
    pd.DataFrame({"sample_id": ["a", "b"], "split": ["train", "val"],
                  "y_true": ["car", "car"], "y_pred": ["car", "car"],
                  "confidence": [.9, .9], "image_path": ["a.png", "b.png"]}).to_csv(csv, index=False)
    result = leakage_analysis(load_predictions(csv), csv)
    assert result["cross_split_identical_image_count"] == 1


def test_block_traversal(tmp_path):
    csv = tmp_path / "predictions.csv"
    pd.DataFrame({"sample_id": ["a"], "split": ["train"],
                  "y_true": ["car"], "y_pred": ["car"],
                  "confidence": [.9], "image_path": ["../../secrets.txt"]}).to_csv(csv, index=False)
    assert leakage_analysis(load_predictions(csv), csv)["blocked_image_paths"] == 1


def test_demo_report_artifacts(tmp_path):
    csv = make_demo(tmp_path / "data.csv")
    result = AuditEngine(csv, tmp_path / "results").write_all()
    assert len(result["files"]) == 5
    assert all((tmp_path / "results" / file).exists() for file in result["files"])
    snapshot = full_audit(csv)
    assert snapshot["performance"]["total_rows"] == 300
    assert snapshot["leakage"]["cross_split_sample_id_count"] == 3
    assert snapshot["performance"]["splits"]["train"]["accuracy"] > snapshot["performance"]["splits"]["val"]["accuracy"]
