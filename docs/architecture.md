# Technical design notes

## Why an Agent instead of a single prompt?

A prompt-only report generator can invent performance claims. This design separates *decision-making* and *measurement*. The LLM sees only tool responses (structured, checked values) and its tool list is intentionally narrow. The agent has a chance to select among performance, failure, leakage and reporting tools based on the user's stated goal. The offline path remains usable without an LLM.

## Data flow

1. `load_predictions` validates CSV size, required columns, split vocabulary and finite probabilities in [0,1].
2. `performance_analysis` computes deterministic per-split and per-class metrics, confusion matrices and 10-bin ECE.
3. `leakage_analysis` checks cross-split sample IDs; optional SHA-256 binary hashes of images whose paths stay under the dataset directory.
4. `hard_examples` sorts wrong predictions by descending predicted-class confidence.
5. `diagnose` emits evidence-backed and noncausal diagnostic flags.
6. `AuditEngine.write_all` persists JSON + 3 plots + Markdown report; `Agent` tools can trigger the same operations.

## Metric definitions

- Accuracy: fraction with `y_pred == y_true`.
- Precision(class c): TP/(TP+FP), zero when denominator is zero.
- Recall(class c): TP/(TP+FN), zero when denominator is zero.
- Macro-F1: unweighted mean across union of true/predicted classes.
- Weighted-F1: support-weighted average across classes.
- Top-label ECE: sum over bins: (count/total) * abs(mean top-1 confidence - observed accuracy). If all confidences are 1.0, they belong to the final bin.
- Confusion matrix rows=truth, columns=prediction.

## Security and reproducibility

- The LLM cannot select an arbitrary file or tool; all file paths are selected at CLI startup and bound in the AuditEngine closure.
- No generic `exec`, shell, arbitrary browser or remote URL access.
- Image files restricted to within the CSV parent path, including resolved symlinks; 50MB per file cap.
- The demo is seeded for reproducibility.
- Prompt injection through labels/group values is treated as data by the agent instructions. More sophisticated attack testing remains an important future improvement.
- Repeated IDs are evidence of overlap requiring provenance checks, not necessarily duplicated images.
- Scores are descriptive; high confidence plus incorrect does not independently establish label errors.
- Group metrics with n<5 are suppressed, but small-sample uncertainty still needs to be quantified in v0.2.

## Future project differentiators

For a strong portfolio presentation, contribute 2-3 genuine model results (small CNN, widened CNN, augmentation change), compare performance, and build an evaluated recommendation benchmark. A key research question is whether grounded tool use improves the correctness and reproducibility of diagnosis compared with pure LLM analysis.
