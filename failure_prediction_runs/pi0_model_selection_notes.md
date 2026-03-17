# PI0 Model Selection Notes

- The strongest held-out run was `pi0_sweep_combo_decision_auroc`.
- The key change was concatenating `feat_action_prefix_flat_10` with `feat_action_chunk_mean`.
- The `decision_only` path remained the most reliable training contract for PI0.
- Selecting the combined model by AUROC worked better than selecting it directly by F1 for downstream intervention sweeps.
