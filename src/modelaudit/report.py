"""Save reproducible artifacts: JSON, plots, and a rule-based audit report."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import full_audit


def evaluation_split(snapshot: dict) -> str:
    splits = snapshot["performance"]["splits"]
    return "val" if "val" in splits else "test" if "test" in splits else "train"


def save_charts(snapshot: dict, output_dir: Path) -> list[str]:
    split = evaluation_split(snapshot)
    result = snapshot["performance"]["splits"][split]
    names = snapshot["performance"]["classes"]
    paths = []

    matrix = np.array(result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(max(6, min(12, len(names) * .6)), 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {split}")
    if len(names) <= 12:
        threshold = matrix.max() / 2 if matrix.size else 0
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > threshold else "black", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=.85)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    paths.append("confusion_matrix.png")

    labels = [row["class"] for row in result["per_class"]]
    recalls = [row["recall"] for row in result["per_class"]]
    fig, ax = plt.subplots(figsize=(max(6, min(14, len(labels) * .55)), 4))
    ax.bar(range(len(labels)), recalls)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title(f"Per-class recall — {split}")
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_recall.png", dpi=140)
    plt.close(fig)
    paths.append("per_class_recall.png")

    bins = [b for b in result["ece"]["bins"] if b["count"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    ax.scatter([b["confidence"] for b in bins], [b["accuracy"] for b in bins],
               s=[min(260, 25 + b["count"] * 5) for b in bins], label="Observed bins")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed accuracy",
           title=f"Top-label reliability — {split} (ECE {result['ece']['ece']:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reliability_diagram.png", dpi=140)
    plt.close(fig)
    paths.append("reliability_diagram.png")
    return paths


def markdown_report(snapshot: dict) -> str:
    perf = snapshot["performance"]
    split = evaluation_split(snapshot)
    lines = [
        "# ModelAudit Agent — Experiment Diagnosis", "",
        f"**Dataset:** `{snapshot['dataset']}`", "",
        f"**Analyzed split:** `{split}` (val > test > train)", "",
        "## Executive summary", "",
        f"- Predictions assessed: **{perf['total_rows']}**",
        f"- Classes: **{len(perf['classes'])}**", "",
        "## Quantitative evidence", "",
        "| Split | Samples | Accuracy | Macro F1 | Weighted F1 | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in perf["splits"].items():
        lines.append(f"| {name} | {row['n']} | {row['accuracy']:.3f} | {row['macro_f1']:.3f} | {row['weighted_f1']:.3f} | {row['ece']['ece']:.3f} |")
    lines.extend(["", f"### Per-class metrics — {split}", "",
                  "| Class | Support | Precision | Recall | F1 |",
                  "|---|---:|---:|---:|---:|"])
    for row in perf["splits"][split]["per_class"]:
        lines.append(f"| {row['class']} | {row['support']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |")
    lines.extend(["", "## Evidence-based findings", ""])
    for idx, issue in enumerate(snapshot["issues"], start=1):
        lines.extend([
            f"{idx}. **{issue['type']}** ({issue['severity']}): {issue['evidence']}",
            f"   - Next experiment: {issue['suggestion']}",
        ])
    lines.extend(["", "## Hard examples (highest-confidence mistakes)", "",
                  "| Sample | True | Predicted | Confidence |",
                  "|---|---|---|---:|"])
    for row in snapshot["hard_examples"][:10]:
        lines.append(f"| {row['sample_id']} | {row['y_true']} | {row['y_pred']} | {row['confidence']:.3f} |")
    lines.extend([
        "", "## Evidence plots", "",
        "![Confusion matrix](confusion_matrix.png)", "",
        "![Per-class recall](per_class_recall.png)", "",
        "![Reliability diagram](reliability_diagram.png)", "",
        "## Limitations", "",
        "- This audit analyzes supplied predictions, not neural-network weights or source images.",
        "- ECE estimates depend on binning and confidence interpretation.",
        "- A train/validation gap is a diagnostic signal, not proof of overfitting.",
        "- Duplicate sample IDs can reflect metadata collisions; binary-identical files are stronger evidence.",
        "- Suggestions are hypotheses requiring controlled follow-up experiments.", "",
    ])
    return "\n".join(lines)


class AuditEngine:
    """A fixed dataset and output location; the LLM cannot choose arbitrary paths."""

    def __init__(self, csv_path: str | Path, output_dir: str | Path):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self._snapshot = None

    @property
    def snapshot(self) -> dict:
        if self._snapshot is None:
            self._snapshot = full_audit(self.csv_path)
        return self._snapshot

    def write_all(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(self.snapshot)
        payload["created_utc"] = datetime.now(timezone.utc).isoformat()
        (self.output_dir / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pngs = save_charts(self.snapshot, self.output_dir)
        (self.output_dir / "report.md").write_text(markdown_report(self.snapshot), encoding="utf-8")
        return {"output_dir": str(self.output_dir),
                "files": ["report.md", "metrics.json", *pngs]}
