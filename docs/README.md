# Documentation index

Start with the documents under *Start here*. The rest are dated records of how the project got where it is: plans,
audits and reports, each written at its date and kept as the project history. Since 2026-11-16, local paths have
been removed from them; nothing else has changed.

## Start here

| Document | What it covers |
|---|---|
| [architecture.md](architecture.md) | The whole system: data, training, retrieval (v1, v2 and the release default), evaluation, integrity |
| [retrieval_v2_metadata_hybrid.md](retrieval_v2_metadata_hybrid.md) | FinRetriever v2: metadata-aware hybrid retrieval experiments and findings, and v2.1 (fiscal-year roles) |
| [retrieval_analysis_v1.md](retrieval_analysis_v1.md) | Retrieval Performance Analysis v1: seven systems, error analysis |
| [demo.md](demo.md) | The search demo: usage, what it shows, and how ready it is |
| [environment.md](environment.md) | Python version, pinned lock files, installation, dependency roles |

## Release and publication

| Document | What it covers |
|---|---|
| [licensing_and_permissions.md](licensing_and_permissions.md) | Licences and terms for the base model, the fine-tuned checkpoints, SEC filing text, the synthetic training data and the dependencies |
| [research_readiness.md](research_readiness.md) | Academic publication: what the evidence supports and what a paper would still need |
| [ci_gradcache_investigation.md](ci_gradcache_investigation.md) | Why CI failed (float32 rounding across platforms) and how the exactness tests were made stricter |

## Data and corpus

- [data_pipeline.md](data_pipeline.md): SEC EDGAR download, parsing, chunking, the corpus manifest

## Training (A1, three seeds)

- **Summary:** [a1_three_seed_summary.md](a1_three_seed_summary.md) (the runs, Stage A decision, disclosures)
- **Training data:**
  - [training_data_v1.md](training_data_v1.md), [training_data_v2.md](training_data_v2.md) and [training_data_v3.md](training_data_v3.md);
  - [train_candidates_v3_build_report.md](train_candidates_v3_build_report.md);
  - the P1 generation audits, `train_v3_p1_batch1…6_generation_audit.md`.
- **Validation set:**
  - [validation_pairs_v4.md](validation_pairs_v4.md) and [validation_split_v3_design.md](validation_split_v3_design.md);
  - the `validation_pilot_*` and `validation_plan_*` audits.
- **Trainer and runs:**
  - [training_trainer_v1.md](training_trainer_v1.md), [training_full_v1.md](training_full_v1.md) and [kaggle_training.md](kaggle_training.md);
  - [kaggle_smoke_test.md](kaggle_smoke_test.md) and [gpu_rehearsal_failure_analysis.md](gpu_rehearsal_failure_analysis.md).
- **Decision rule and safeguards:**
  - [training_b1_b3_proposal.md](training_b1_b3_proposal.md) and [training_a1_implementation_report.md](training_a1_implementation_report.md);
  - [training_a1_execution_safeguards.md](training_a1_execution_safeguards.md) and [training_a1_kaggle_rehearsal_report.md](training_a1_kaggle_rehearsal_report.md).
- **Per-seed audits:** [seed42_final_audit.md](seed42_final_audit.md), [seed1337_final_audit.md](seed1337_final_audit.md) and [seed2026_final_audit.md](seed2026_final_audit.md).
- **Earlier plans (historical):**
  - [training_plan_v1.md](training_plan_v1.md) and [final_training_readiness_audit.md](final_training_readiness_audit.md);
  - [training_a1_seed42_plan.md](training_a1_seed42_plan.md) and [training_a1_seed2026_readiness.md](training_a1_seed2026_readiness.md);
  - the training pilot quality reports.

## Evaluation

- **Methodology:** [eval_methodology.md](eval_methodology.md) and [checkpoint_evaluation.md](checkpoint_evaluation.md).
- **Development labels:** [pool_review_pilot_v3.md](pool_review_pilot_v3.md), [flag_review_pilot_v3.md](flag_review_pilot_v3.md) and [audit_review_pilot_v1.md](audit_review_pilot_v1.md).
- **Gold set v1, closed as an incomplete pilot:**
  - [gold_set_v1_closure.md](gold_set_v1_closure.md), the closure record; read it first;
  - then the protocol, query plan, question validation, readiness, pooling, grading, annotator guide and remote
    annotation documents (`gold_set_v1_*.md`).
