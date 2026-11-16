# Trainer smoke test: bge_base_ft_v2_smoke-seed42

Mechanics check only: no retrieval evaluation was run and no retrieval improvement is claimed. The training data are 48 synthetic queries accepted by an AI quality review (not human-validated, not gold labels), and no checkpoint was selected by any metric.

| Item | Value |
|---|---|
| Run | bge_base_ft_v2_smoke-seed42 |
| Seed | 42 |
| Base model | BAAI/bge-base-en-v1.5 @ a5beb1e3e68b9ab74eb54cfd186867f64f240e1a |
| Training data | 48 candidates of train_candidates_v1 (AI-reviewed; not gold) |
| Batch size, fp16, gradient checkpointing | 16, True, False |
| Steps completed | 15 of 15 |
| Training loss, first step -> last step | 0.6985 -> 0.0382 (range 0.0001-0.7776) |
| Non-finite losses / gradient norms | 0 / 0 |
| Peak GPU memory (measured) | 4.997 GiB on Tesla T4 |
| Duration: training steps / wall clock | 6.8 s / 12.0 s |
| Checkpoints expected / found | [5, 10, 15] / [5, 10, 15] |
| Checkpoint reload | checkpoint-step-000005: reloads, finite; sentence-transformers loads; checkpoint-step-000010: reloads, finite; sentence-transformers loads; checkpoint-step-000015: reloads, finite; sentence-transformers loads |
| Resumable state | step 15, schedule matches: True |
| Final weights vs base | 147 of 149 tensors changed (mean /change/ 4.62e-05) |
| Console: tokenizer, model and other library output | 1 distinct lines need review (tokenizer 1); listed below |
| Model load report keys | embeddings.position_ids UNEXPECTED (known benign) |
| Mechanics OK | True |

Console lines needing review:

- tokenizer: `Token indices sequence length is longer than the specified maximum sequence length for this model (739 > 512). Running this sequence through the model will result in indexing errors` (1x)
