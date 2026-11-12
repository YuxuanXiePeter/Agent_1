# Detailed implementation roadmap

## Day 1 — Data contract and baseline
- Decide labels and splits; export sample_id, y_true, y_pred, confidence.
- Create deterministic metric functions and unit tests.
- Capture known invariants: perfect predictions yield F1=1, ECE=0 for confidence=1.

## Day 2 — Diagnostic evidence
- Add cross-split ID checks and identical-byte image hashing.
- Add high-confidence errors, subgroup performance and calibration visualizations.
- Benchmark on intentionally corrupted and clean samples.

## Day 3 — Agent layer
- Wrap existing functions as limited-scope tools.
- Add strict prompts, memory, execution limits, and no shell access.
- Compare tool-use decisions across a bank of 20 prompts.

## Day 4 — Output and engineering
- Produce JSON + report + charts; add tests, README and reproducible demo.
- Run with predictions from a real CIFAR-like model.
- Add CI, linting and GitHub demo GIF once interface exists.

## Day 5+ — Advanced ML/CV features
- PyTorch adapter: batch inference from saved checkpoints (locally trusted checkpoints only).
- Grad-CAM / attention visualization (not available in v0.1).
- Embedding-based failure clusters (UMAP/HDBSCAN optional).
- MLflow/W&B integration and automated controlled ablations.
- Human approval before modifying training configurations or launching GPU jobs.
