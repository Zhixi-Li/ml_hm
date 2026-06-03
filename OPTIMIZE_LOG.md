# LC Optimization Log

This file tracks optimization ideas, code changes, and experiment analysis for the LC/POMO route.

## 2026-06-02: Distance-aware LC and mixed-scale training

### Motivation

- The original LC baseline uses Transformer attention over node embeddings but does not explicitly inject pairwise distances into encoder attention or decoder logits.
- TSP decisions are strongly local-geometric, so the policy should see both learned context and current-to-candidate distances.
- The project evaluates in-distribution TSP-50, OOD TSP-50, and cross-scale TSP-100; training only on TSP-50 is likely weak for TSP-100.

### Code changes

- `Project/baselines/lc_baseline/model/lc_model.py`
  - Kept the public `LCModel` interface compatible with the project specification.
  - Replaced encoder self-attention with distance-aware multi-head attention.
  - Added per-head distance bias to encoder attention scores.
  - Added decoder distance logit bias from current node to candidate nodes.
  - Cached the distance matrix during `pre_forward`.
- `Project/baselines/lc_baseline/model/tsp_env.py`
  - Kept the public `TSPEnv` interface compatible with the project specification.
  - Added dynamic node-count synchronization in `load_problems_manual`.
  - This allows the same evaluation path to run TSP-50 and TSP-100 without manually changing `env_params`.
- `Project/baselines/lc_baseline/train_lc.py`
  - Added D4 geometric augmentation on generated coordinates.
  - Added 50/75/100 mixed-scale curriculum after epoch 60.
  - Switched to AdamW and added gradient clipping.
  - Validates on all three public validation sets and saves by weighted validation gap.
  - Saves the submit checkpoint to `checkpoints/best_model.pth`.
- `Project/baselines/lc_baseline/evaluate_lc.py`
  - Evaluates all three public validation sets by default.
  - Prints cost, optimal cost, gap, and timing.

### Experiment result

Compared with the saved LC baseline on TSP-50 uniform:

| Model | TSP-50 Uniform Gap | Avg Time / Instance |
|---|---:|---:|
| LC baseline | 3.23% | 0.0253s |
| Distance-aware LC | 2.03% | 0.0201s |

Full public validation result for distance-aware LC:

| Dataset | Average Cost | Optimal Cost | Gap | Avg Time / Instance |
|---|---:|---:|---:|---:|
| TSP-50 Uniform | 5.7863 | 5.6709 | 2.03% | 0.0201s |
| TSP-50 OOD | 5.0089 | 4.8343 | 3.64% | 0.0184s |
| TSP-100 Uniform | 8.2796 | 7.8196 | 5.87% | 0.0348s |

Weighted validation gap:

```text
0.6 * 2.03 + 0.2 * 3.64 + 0.2 * 5.87 = 3.12%
```

### Analysis

- The in-distribution TSP-50 result is clearly better than baseline, so the distance-aware policy is useful.
- OOD and TSP-100 remain weaker than TSP-50 uniform.
- The largest remaining issue is cross-scale generalization: TSP-100 gap is 5.87%, much higher than the 2.03% TSP-50 uniform gap.
- Checkpoints showed validation score still improving late in training, so longer training is likely beneficial.

## 2026-06-03: Stronger cross-scale curriculum

### Motivation

- The previous curriculum starts mixed-scale training only at epoch 60 and samples TSP-100 with probability 0.2.
- The current weakest metric is TSP-100 uniform, so the model should see larger instances earlier and more often.
- Model size is kept unchanged to avoid increasing inference cost.

### Code changes

- `Project/baselines/lc_baseline/train_lc.py`
  - Increased `EPOCHS` from 100 to 150.
  - Moved `CURRICULUM_START_EPOCH` from 60 to 35.
  - Added explicit curriculum constants:
    - `CURRICULUM_NODE_CNTS = [50, 75, 100]`
    - `CURRICULUM_PROBS = [0.4, 0.2, 0.4]`
  - Printed curriculum configuration at training start for reproducibility.

### Expected effect

- TSP-100 gap should improve because the policy trains on 100-node rollouts for more epochs.
- TSP-50 uniform may slightly fluctuate because less post-curriculum training mass is assigned to 50-node instances.
- The weighted validation checkpoint rule should prevent selecting a model that sacrifices too much TSP-50 performance.

### Next experiment to run

Run:

```bash
cd Project/baselines/lc_baseline
python train_lc.py
python evaluate_lc.py > checkpoints/results_curriculum_150.txt
```

Compare `results_curriculum_150.txt` against `checkpoints/results.txt`.

### Experiment result

Result file: `Project/baselines/lc_baseline/checkpoints/results_curriculum_150.txt`

| Dataset | Previous Gap | New Gap | Delta |
|---|---:|---:|---:|
| TSP-50 Uniform | 2.03% | 1.90% | -0.13 |
| TSP-50 OOD | 3.64% | 4.36% | +0.72 |
| TSP-100 Uniform | 5.87% | 4.98% | -0.89 |

Weighted validation gap:

```text
previous = 0.6 * 2.03 + 0.2 * 3.64 + 0.2 * 5.87 = 3.12%
new      = 0.6 * 1.90 + 0.2 * 4.36 + 0.2 * 4.98 = 3.01%
```

### Analysis

- The stronger curriculum achieved the intended cross-scale improvement: TSP-100 gap dropped from 5.87% to 4.98%.
- TSP-50 uniform also improved slightly from 2.03% to 1.90%.
- TSP-50 OOD worsened from 3.64% to 4.36%, which suggests the curriculum shifted capacity toward uniform large-scale instances but did not improve distribution robustness.
- The new checkpoint is still better under the project weighted metric, but the gain is modest because OOD degradation offsets part of the TSP-100 gain.

### Decision

- Keep the 150-epoch stronger curriculum result as the current best weighted checkpoint.
- Next optimization should target OOD robustness without giving up the TSP-100 gain.
