# CPU scenario study: value of world-model consultation

This original small navigation simulator supplements the separate Push-T study. It is not a reproduction of a published robotics benchmark and does not establish a universal need for world models.

## Scenarios

Four fully observed 11×11 grid families: open navigation; staggered walls requiring detours; collect a key, pass a locked door, then reach the goal; and staggered-wall navigation with control directions rotated 90 degrees. The rotated-control condition is absent from pretraining. Starts, goals, wall gaps, and key/door positions vary. All methods receive the same 148-dimensional observation of the map, positions, inventory, local occupancy, last action, and last displacement. Hidden control rotation is not supplied to policies or learned predictors.

## Shared data and methods

Three independent training seeds. Each training set contains 40 layouts from each of the first three families. All navigation networks use two width-128 hidden layers and train for 35 epochs with Adam (learning rate 0.001). Adaptation fine-tunes for 20 epochs at learning rate 0.0005. A BFS expert supplies action-chunk labels, with 25% random exploration producing off-expert transitions. Every method accesses the identical dataset (expert action labels and observed action consequences). The expert is a data generator and shortest-path diagnostic, never a deployed planner.

- Baseline: a two-layer, width-128 MLP predicts six future actions directly.
- A: an action-conditioned MLP classifies one-step displacement; roll it forward six times.
- B: an action-conditioned MLP predicts all six future displacements directly.
- Direct scorer: an MLP predicts a candidate sequence's maximum short-horizon task-progress score without future states.

The baseline uses its most likely first action and does not pay candidate-sampling overhead. Each selector sees the same 12 action-sequence proposals (including the baseline sequence). Only the first selected action is executed, then the policy replans. Scores use Manhattan distance to the active subgoal and a fixed key-acquisition bonus; no shortest-path value is available to deployed methods. Both world models learn motion, while map observation reconstruction, inventory updates, bounds, and the reward function are known. These are deliberately small hybrid models, not general learned simulators.

One-step A and six-step B differ in output representation and parameter count; equal architecture width and examples do not imply equal compute. Parameter counts and training time are recorded. Model B can predict geometrically impossible trajectories; this limitation is intentionally measurable, not repaired using true transitions.

## Adaptation and evaluation

Zero, one, or five new expert interaction episodes from disjoint layouts. All methods receive exactly the same adaptation episodes and 500 shared replay windows; 20 fixed fine-tuning epochs. The number of windows/steps is recorded separately because episodes vary in length. Terminal states are absorbing when padding supervised future windows.

Per training seed and condition: 12 validation layouts and 20 held-out test layouts, paired across methods. 80-step trial limit. Primary success rate; secondary capped steps, success-weighted shortest-path efficiency, blocked moves, mean/p95 inference latency, total inference CPU time, and next-state error. Simulated time pauses during inference. Wall time is not a robot real-time control experiment.

The consultation rule is fitted only on validation trials. It uses initial map density, goal distance, key-task indicator, baseline action entropy, initial vertical position, and adaptation budget. Ridge regression predicts each strategy's success-minus-lambda-time utility. It chooses a strategy once at the episode start: this is **episode-level selection, not a validated per-decision consultation gate**. All chosen-strategy test outcomes come from held-out measured episodes. Lambdas are prespecified: 0, 0.1, 1, 10 per second. Timing excludes episode-gate feature extraction and evaluation, which is a limitation for near-zero differences.

Run `bash run_navigation.sh`; dependencies are pinned in the CPU lockfile. Raw per-episode rows, training metrics, scenario summaries, and held-out gate results are written to the configured project artifact directory and printed in run logs.
