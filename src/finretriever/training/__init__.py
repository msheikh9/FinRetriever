"""Training subpackage.

Leakage-safe training-data preparation (``overlap``, ``evidence_reuse``,
``blocklist``, ``splits``, ``visible_text``, ``data_prep``), source selection
(``source_quality``, ``sampling_plan``, with the evidence-pass predictor in ``evidence_prediction``), synthetic query generation
(``query_generation``, with quote matching in ``quotes``), pair quality control
(``pair_quality``, with the pair contract in ``pair_contract``), false-negative
protection (``false_negatives``), explicit carry-over of earlier pairs
(``carryover``), versioned training candidates accepted by an AI quality review
(``candidates``), a separate, write-once validation set (``validation_set``), split pins that keep
existing data in place across preparations (``split_pins``, ``pinning``)
and the interface a trainer must use (``training_data``),
driven by ``data_cli`` / ``scripts/02_generate_queries.py``. The contrastive
bi-encoder trainer (``trainer_setup``, ``biencoder_trainer``, with GradCache,
seeds and validation-based checkpoint selection: ``validation_eval``,
``checkpoint_selection``) is driven by ``scripts/05_train_biencoder.py``; its
GPU rehearsal (``gpu_rehearsal``, no optimizer step) by ``scripts/gpu_rehearsal.py``,
and the selection-aware A0-versus-A1 decision (``decision_rule``, simulated in
``decision_simulation``) by ``scripts/07_decide_a1_vs_a0.py``. The semantic leakage
screen (``semantic_screen``) is consulted only when candidates are built.
Hard-negative mining (``negative_mining``) is still a stub.
"""
