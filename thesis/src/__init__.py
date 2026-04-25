"""
thesis/src — Theta-Augmented GBT Pipeline modules

Module load order reflects pipeline execution order:
    1. data_loader       — Step 1:  Load & preprocess
    2. feature_ordering  — Step 2:  Feature ordering (3 strategies)
    3. braid_word        — Step 3:  Braid word generation
    4. braid_closure     — Step 3b: Braid closure (trace closure)
    5. theta_eval        — Step 4:  Approximate Theta (linear)
    6. alexander         — Step 4b: Exact Theta (Alexander matrix)
    7. sparse_handler    — Step 5:  Sparse feature filtering
    8. feature_augment   — Step 6:  Feature augmentation
    9. model_training    — Step 7:  LGBM training & classification metrics
   10. recall_eval       — Step 7b: 3-condition Recall@k evaluation
   11. ann_baselines     — Step 7c: ANN baselines (HNSW, Annoy)
   12. efficiency_benchmark — Step 8: Computational efficiency
   13. sparsity_ablation    — Step 9: Sparsity ablation
   14. visualization        — Step 10: Plots & CSV reporting
"""
