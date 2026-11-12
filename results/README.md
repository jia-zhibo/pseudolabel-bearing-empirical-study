# Raw result files

Every number in the manuscript tables is recomputed from these JSON files by
`compute_statistics.py`. No value was transcribed by hand.

| File | n | Content |
|---|---|---|
| `authoritative_bilstm_n50_n20.json` | 20 | **Table 1 (FINAL)**. BiLSTM-Attention, CWRU 0->3 hp, 4 branches, per-repetition accuracies + `_meta` (seeds, threads, epochs, window, stride). |
| `cwru_n50_da_comparison.json` | 5 | Section 7 **negative control**: the original execution whose baseline branch was unseeded (+4.01 pp, p=0.0352). Retained for provenance only; NOT used in any table. |
| `authoritative_bilstm_n50.json` | 5 | Fully seeded, thread-pinned n=5 run; intermediate replication step. |
| `authoritative_fastcnn_main.json` | 10 | **Table 2**. FastCNN, CWRU + PU, 6 configurations, plus identical per-epoch `train_loss` / `val_acc` / `target_acc` curves for every branch. |
| `authoritative_fastcnn_ablation.json` | 3 | **Tables 3 & 4**. Ablation under `plain` (as-run) and `aligned` (loss-matched) objectives, with per-epoch curves. |
| `revision_pseudo_matched_loss.json` | 10 | Section 6.1 loss-matched pseudo-label comparison. |
| `revision_training_curves.json` | 3 | Section 6.4 training dynamics; contains a `consistency_check` block. |
| `revision_cwru_10repeats.json`, `revision_pu_10repeats.json`, `revision_ablation_3repeats.json` | 10 / 10 / 3 | Cross-check replicates behind the Appendix tables. |

Not shipped (too large / not needed): `revision_cwru_visualization.json` (7.5 MB),
`revision_pu_visualization.json` (44 MB), `revision_full_instrumented.json` (441 KB,
bit-identical to `authoritative_fastcnn_main.json`) and
`revision_ablation_instrumented.json` (329 KB, bit-identical to
`authoritative_fastcnn_ablation.json`), plus checkpoints (88 MB) and raw datasets (5.8 GB).
