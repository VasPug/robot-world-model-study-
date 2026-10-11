# Push-T world-model value pilot

This is a controlled CPU pilot using the official released **low-dimensional Transformer Diffusion Policy** EMA checkpoint. It is not retraining the paper's policy and is not a full reproduction of its table results. Source revision and per-file hashes are in `vendor/provenance.json`; model and dataset downloads are checksum-pinned in `assets.json`.

## Question and fixed protocol

Does an action-conditioned predictor of future object keypoints improve a frozen policy's decisions enough to justify candidate generation and selection cost, compared with the original policy and a direct progress scorer?

- Official Push-T simulator, legacy compatibility mode, full keypoint observations, fixed target geometry.
- Original 100-step DDPM sampler, eight executed controls per decision, maximum 300 controls per episode.
- Baseline executes one policy sample. Three selection variants each generate four samples, then select by learned world model or learned direct scorer. Candidate zero is generated separately to preserve its random sample at identical histories. Selection arms have identical sampling budgets.
- 12 fresh reset seeds (610000–610011), each with standard and +40-pixel initial block translation: 24 paired episodes per method. The shifted condition is a controlled displacement, **not a claim of out-of-distribution evaluation**. No test tuning.
- Recover the 90 demonstration episodes used by the released checkpoint; use 72 to fit both auxiliaries and 18 for validation/model selection. No extra interaction data. Whole episodes stay in one split.
- Both auxiliaries take two observations and the proposed eight actions. World model predicts eight future sets of block keypoints via a residual MLP. Predicted keypoints are projected onto the known rigid object to compute task reward. Direct scorer predicts maximum reward over the same eight controls. It predicts short-horizon progress, **not eventual success**.
- Both use two 256-unit hidden layers, Adam 0.001, batch 256, 80 epochs, seed 42, best validation MSE. Parameter counts differ due to output size and are reported. Shared data and sampling budgets do **not** imply perfectly matched training FLOPs or model capacity.
- Primary outcome: success rate (>95% goal coverage). Secondary: maximum normalized coverage reward, mean/p95 decision latency, process CPU seconds, task steps, wall time, selected-action prediction error, auxiliary training time and parameter count.
- Simulator time pauses during inference; wall-clock latency is reported separately and does not model real-time control degradation. Time-to-success alone is not averaged after dropping failed trials: all-episode lengths are capped at the fixed horizon.
- 95% intervals resample reset seeds as clusters, retaining both conditions together. One released policy seed and one auxiliary seed: intervals reflect reset variation, not training uncertainty.

## Interpretation and limitations

A win over the baseline demonstrates usefulness of this candidate-selection procedure in this task; a win over the direct scorer is stronger evidence for the explicit future-prediction representation. A loss does not prove robots do not need world models. Demonstration-only training may predict proposed off-demonstration actions poorly. The short-horizon geometric score can favor locally good but globally poor moves. A simulator-oracle diagnostic and multiple policy/training seeds are follow-ups, not included in this pilot. Model training cost must be amortized across deployments before claiming net value.

The official checkpoint was selected using the authors' original evaluation; our reset seeds are different. Library versions are pinned but modernized for a CPU Python 3.11 environment, so no claim is made of matching the published benchmark score.

## Reproduction

Run `bash run.sh --config configs/pusht/baseline.json` to train auxiliaries and evaluate the policy. Run the other configurations to evaluate selectors with the included, checksum-verified auxiliary weights. Pass `--auxiliary-dir` to use newly trained weights. Commands and output locations are documented in [reproduction.md](reproduction.md).

Artifacts are written under the configured `artifact_root`; each method has JSON and CSV episodes, per-decision JSONL, effective configuration, and a summary. The run prints provenance, effective configuration, training curves, each episode, and final metrics.

Original work: https://www.alphaxiv.org/pdf/2303.04137
Official implementation: https://github.com/real-stanford/diffusion_policy

## Latest predictive variants

Model A regresses the next 20 state coordinates from two observations and the first candidate action, then recursively rolls eight controls. Model B predicts all eight future 20-dimensional states directly. Both use known rigid-body geometry to score predicted goal coverage. The direct scorer estimates short-horizon progress without predicting states. Selection configurations pin the shared auxiliary weight hashes; these weights are included under `results/pusht-pilot/`.
