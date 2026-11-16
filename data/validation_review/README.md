# Validation-question review decisions

Quality decisions on synthetic validation questions, kept apart from the training-pair decisions
(`data/train_review/`) and from evaluation data (`data/eval/`).

- `val_pilot_v1_quality_decisions.jsonl`: one row for each of the 43 questions the generator
  accepted in the validation pilot `val_pilot_v1` (`bge_base_ft_v2`, 50 validation passages), from
  the audit in [docs/validation_pilot_v1_quality_report.md](../../docs/validation_pilot_v1_quality_report.md).
  The format is that of `data/train_review/`, plus one outcome for each reviewed aspect:
  - `pair_id`: `validation_<NNNNNN>`, the passage's position in the plan, fixed before any check.
  - `passage_id` and `question_sha256` bind the decision to the exact question, so it never
    applies to an edited or different question.
  - `decision`: `accept`, `revise` or `reject` (a reason is required for the last two).
  - `aspects`: `factual_accuracy`, `financial_relevance`, `company`, `period`, `figures` and
    `units`, each `pass`, `minor` (imprecise, but the passage still answers the question as asked),
    `fail` or `n/a`. An accepted question fails none; a revised or rejected one fails at least one.
  - `issues`: the same vocabulary as the training decisions.
  - `reviewer: ai_review` and `gold: false`: an AI review, not human annotation. These are never
    gold labels and never evaluation data.

- `val_pilot_v2e_batch1_quality_decisions.jsonl`: one row for each of the 39 questions the generator
  accepted in the first batch (plan positions 1–50) of `val_pilot_v2e` (`bge_base_ft_v3`).
  - The audit is [docs/validation_pilot_v2e_batch1_audit.md](../../docs/validation_pilot_v2e_batch1_audit.md).
  - The format is the same as above: 36 accept, 1 revise, 2 reject.
- `val_pilot_v2e_batch1_outcomes.jsonl`: one row for each of the 50 passages of that batch, an audit log
  that `build-validation` does not read. Each row records:
  - the plan position, `pair_id` and `passage_id`;
  - the generation record's SHA-256 (`record_sha256`);
  - the question, verbatim, with its SHA-256;
  - the generator's outcome (`generator.accepted`, with its rejection reasons);
  - the validation evidence rule's outcome (`evidence_rule.passed`, with the reasons it excluded the
    question);
  - the review decision and issues;
  - `excluded_by`: `generator`, `evidence_rule` and/or `review`, and `usable`. 15 are usable.
  - The decisions file holds no generator-rejected passage, because a decision needs a question that
    passed the generator. The outcomes file is where all 50 are recorded.

- `val_pilot_v2f_positions1-147_quality_decisions.jsonl`: one row for each of the 115 questions the
  generator accepted in plan positions 1–147 of `val_pilot_v2f` (`bge_base_ft_v3`), the windows of
  batches 1–3.
  - The audits are [batch 1](../../docs/validation_plan_v2f_batch1_audit.md) (positions 1–49),
    [batch 2](../../docs/validation_plan_v2f_batch2_audit.md) (50–97),
    [batch 3](../../docs/validation_plan_v2f_batch3_audit.md) (98–143) and
    [positions 144–147](../../docs/validation_plan_v2f_positions_144_147_audit.md); each row's `source`
    names its audit.
  - The format is the same as above: 101 accept, 9 revise, 5 reject.
  - The plan has 361 positions. Positions 148–361 have no decision here, so `build-validation` would
    exclude them (`no_quality_decision`); their decisions go into a new file, not into this one.
- `val_pilot_v2f_positions1-147_outcomes.jsonl`: one row for each of those 147 positions, in the
  format of `val_pilot_v2e_batch1_outcomes.jsonl`, plus the candidate's `answer`, `evidence` (the
  quote, verbatim) and `question_type`. For a generator-rejected position these are the rejected
  candidate's. 50 are usable; the others are excluded by the generator (32), the evidence rule (51),
  the review (10) or both of the last two (4).

- `val_pilot_v2f_positions148-234_quality_decisions.jsonl`: one row for each of the 66 questions the
  generator accepted in plan positions 148–234 of `val_pilot_v2f`: batch 4 (148–195) and the records
  written after it, up to position 234, where generation was stopped
  (`plans/validation.val_pilot_v2f.stopped.json`).
  - The audits are [batch 4](../../docs/validation_plan_v2f_batch4_audit.md) (148–195) and
    [positions 196–234](../../docs/validation_plan_v2f_positions_196_234_audit.md); each row's `source`
    names its audit.
  - The format is the same as above: 63 accept, 2 revise, 1 reject.
  - No position after 234 has a record or a decision: the plan is closed there.
- `val_pilot_v2f_positions148-234_outcomes.jsonl`: one row for each of those 87 positions, in the
  format of `val_pilot_v2f_positions1-147_outcomes.jsonl`. 12 are usable; the others are excluded by
  the generator (21), the evidence rule (51), the review (1) or both of the last two (2).

`validation_pairs_v4` (configs/train/validation_v4.yaml) reads both `val_pilot_v2f` decision files
together; no question is decided in both.

The questions themselves are not edited. `build-validation` (configs/train/validation_v1.yaml)
reads these decisions; only accepted questions that pass every check enter the validation set. The
set records this file's SHA-256 and refuses to load once it changes: edit decisions only by
writing a new file and a new validation-set version.
