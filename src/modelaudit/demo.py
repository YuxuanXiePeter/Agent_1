"""Reproducible synthetic classification output; deliberately includes audit signals."""
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo(path: str | Path, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    labels = ["car", "truck", "bus"]
    rows = []
    for split, n in (("train", 150), ("val", 90), ("test", 60)):
        for i in range(n):
            true = labels[i % 3]
            group = "night" if i % 4 == 0 else "day"
            accuracy = .97 if split == "train" else (.49 if group == "night" else .75)
            is_correct = rng.random() < accuracy
            if is_correct:
                pred = true
                conf = rng.uniform(.71, .99)
            else:
                # Model often confuses trucks and buses, sometimes with high confidence.
                pred = "bus" if true == "truck" else labels[(labels.index(true) + 1) % 3]
                conf = rng.uniform(.89, .99) if i % 3 == 0 else rng.uniform(.55, .88)
            sample_id = f"{split}-{i:03}"
            if split == "val" and i in (5, 17, 39):
                sample_id = f"train-{i:03}"  # intentionally suspicious ID reuse
            rows.append({"sample_id": sample_id, "split": split,
                         "y_true": true, "y_pred": pred, "confidence": round(conf, 5),
                         "group": group})
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
