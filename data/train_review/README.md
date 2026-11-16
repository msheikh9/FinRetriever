# Training-pair review decisions

Quality decisions on synthetic training pairs, kept apart from evaluation data (`data/eval/`).

- `pilot_v2_quality_decisions.jsonl`: one row per pair of the first Qwen pilot (`bge_base_ft_v1`,
  37 pairs), from the audit in [docs/training_pilot_v2_quality_report.md](../../docs/training_pilot_v2_quality_report.md).
  Each row is bound to its pair by `passage_id` and the SHA-256 of the exact question
  (`question_sha256`), so a decision never applies to an edited or different question.
  - `decision`: `accept`, `revise` or `reject` (a reason is required for the last two).
  - `issues`: `boilerplate`, `misinterpreted_table_value`, `incorrect_period`,
    `weak_financial_relevance`, `inaccurate_wording`.
  - `reviewer: ai_review` and `gold: false`: an AI review, not human annotation. These are never
    gold labels and never evaluation data.

- `pilot_v3_quality_decisions.jsonl`: the same format, for the 70 pairs of the second pilot
  (`bge_base_ft_v2`), from [docs/training_pilot_v3_quality_report.md](../../docs/training_pilot_v3_quality_report.md).

The generated questions themselves are not edited. `carry-over` (docs/training_data_v2.md) and
`build-candidates` (configs/train/candidates_v1.yaml, docs/training_trainer_v1.md) read these
decisions; only accepted pairs that pass every current check become training candidates. A
candidate dataset records each file's SHA-256, and the trainer refuses it once a decision file
changes: edit decisions only by writing a new file and a new candidate version.
