# GPU extension: useful planning horizon under nominal and shifted physics

This is a separate extension of the published CPU Push-T pilot. The pilot and extension use separate protocols. Reference measurement files accompany this release.
The public package preserves the experimental design and provides portable entry points.

## Questions and scope

Does consulting a learned dynamics predictor improve success enough to justify
its measured inference cost? Does increasing model capacity help independently
of increasing imagined horizon? Can physics mismatch make consultation harmful?
This is a state/keypoint-based hybrid world model, not a video foundation model.
Frozen policy/predictor/scorer arms evaluate physics shifts zero-shot. The later
episode gate pools validation outcomes from all four conditions, including the
shifts, so the gate is not zero-shot to those distributions. Its paired recovery
and shifted starts have indistinguishable initial features. Online adaptation is a separate hypothesis
and is not part of this original extension.

## Scenarios

- Easy: block orientation within 0.1 radians of the target, 25-45 pixels below
  it and at most 12 pixels lateral offset. Pusher on the far side from the goal.
- Recovery: block rotation differs by 0.8-1.5 radians, signed randomly; translation
  is within 65 pixels of the goal on each axis. Pusher begins on the goal side.
- Friction shift: exactly the same recovery reset, with both pusher and block
  collision-shape friction set to 0.9 rather than the simulator default zero.
- Damping shift: exactly the same recovery reset, space damping 0.8 instead of zero.

The last two are separate strata in one changed-physics scenario family. Mass
is not used because the kinematic pusher and simulator damping can make simple
mass scaling an ineffective intervention. A contact-rich probe must establish
that each selected intervention changes observations by more than 0.01 pixels.
Difficulty names express the design intent; success rates must establish actual
difficulty. The oracle determines whether longer horizons help under the shared
candidate and geometric-score restrictions. Do not select final test resets by
method performance. Starts already meeting the success criterion fail a check.

## Shared proposals and models

Use the same released 100-DDPM-step diffusion policy, full keypoint observations,
and target geometry as the original pilot. Execute eight controls and then replan
in every arm, irrespective of lookahead. Candidate zero is sampled separately
and must match the single-sample policy at the identical history and random seed.
All learned selectors and the oracle receive eight candidates. After each sampled
eight-action prefix, retrieve a continuation from the original training-only
demonstrations, matching the prefix among 64 nearest-observation windows. Blend
the endpoint displacement back to the demonstration over the remaining 56 steps.
The full 64-control candidates are generated once, with horizons selecting prefixes.
Neither true simulator futures nor any learned world model generates proposals.

This proposal restriction is substantive: the released policy cannot directly
generate a 64-step plan. The oracle is an upper bound only for this candidate set
and score, not for robotics performance in general.

World models are two-hidden-layer SiLU MLPs with widths 128, 512 and 2560,
approximately 0.2M, 1M and 10M parameters. Each predicts 64 future 20-dimensional
observations relative to the present. Inputs include two observations, masked
candidate actions, and the requested horizon mask. No action after the horizon
is visible. Train losses cover only requested future steps. The same model
supports horizons 8, 32 and 64, each sampled uniformly during training.

Direct scorers have the same family and inputs, with width adjusted to match the
world model's total parameter count as closely as possible. They predict maximum
normalized coverage over the requested horizon. World-model trajectories are
projected onto the known rigid block and scored with the same coverage objective.
Exact simulator selection uses the same objective, candidates and execution
interval; its true physics is privileged and diagnostic, never a deployed arm.
It replays all executed actions from reset to preserve contact-solver history.

## Data and fairness

Recover the original checkpoint's eligible 90 demonstration episodes; use the
same 72/18 whole-episode train/validation split as the published auxiliary pilot.
Half the examples are observed 64-step demonstration windows. Half use perturbed
demonstration actions simulated from training/validation episode states with
nominal physics, generating off-demonstration action consequences. Split IDs
remain disjoint. Terminal success becomes absorbing; never cross episode ends.
Episodes with fewer than 66 observations cannot provide the two-observation
history plus complete 64-step future; exclude them from window sampling and
record the excluded IDs in the run log.
All auxiliary sizes/seeds and scorers use identical examples. No shifted physics
or final test reset contributes training labels. Save generated arrays and split
IDs, input checksums, source checksums, checkpoint hashes and learning curves.

The released direct policy is frozen and does not train on these extra simulated
interactions. Therefore policy-versus-selector comparisons assess the incremental
procedure, not perfectly data-matched model-free learning. The matched direct
scorer is the main control for the value of explicit state prediction. There is
one released policy training seed; additional seeds measure auxiliary training
uncertainty, not policy-training uncertainty.

## Feasibility and subsequent evaluation

The feasibility stage uses one auxiliary seed, 240 training windows, 60 validation
windows, 20 epochs, three model sizes, and closed-loop evaluation of the middle
size across all three horizons. Two independent reset seeds per stratum and a
128-control episode cap keep this strictly a feasibility experiment. It measures
memory, runtime, geometry agreement, prefix masking, candidate-zero consistency,
oracle replay and the effect of physics interventions. No confirmatory hypothesis
decision should be based on it.

The full study retains the working setup, increases data/training,
uses three auxiliary seeds and 100 paired held-out resets per stratum,
and restores the 300-control task budget. It reserves independent validation reset
seeds for any model/horizon choice and consultation rule. Finalize the exact
matrix from measured feasibility costs before running; record all choices in
the committed full-study configuration. Do not tune on held-out outcomes.

Primary endpoint: success (>95% target coverage). Secondary: capped controls,
maximum coverage, mean and p95 total decision latency, total inference seconds,
selected prediction RMSE, and first-decision candidate-ranking regret against
simulator rollouts on a prespecified diagnostic subset. Synchronize CUDA before
and after timing. Diagnostics are excluded from deployment latency; proposal and
learned scoring costs are included. Simulator time pauses during inference.
Report training cost and peak allocated CUDA memory separately.

Use paired differences on matching resets. Bootstrap auxiliary seeds and reset
IDs as two crossed sampling axes, preserving arms and shift pairing. With one
auxiliary seed, intervals address only reset variation. Multiple horizons and
sizes are exploratory comparisons, not multiplicity-corrected significance tests.

## Reproduction

Run `bash run_gpu.sh` on Linux or `powershell -NoProfile -ExecutionPolicy Bypass -File run_gpu.ps1` on Windows. The command recreates the CUDA environment from `gpu_extension/uv.lock`. Output uses a new timestamped directory. See [reproduction.md](reproduction.md) for requirements and command-line overrides.

## Full-study configuration

A small feasibility stage preceded the full study. Its outcomes are not included as confirmatory evidence.

The full study uses 3,000 training and 600 validation windows, 80 epochs, all three
model sizes and auxiliary seeds 11/22/33. Training uses AdamW with batch size 128
and learning rate 0.0003. It evaluates all 54 learned arms plus
the policy baseline on 12 validation resets (810000-810011) and 100 held-out test
resets (820000-820099), each in four strata: 24,640 learned/policy episodes in
total, of which 22,000 are held out. All episodes have the original 300-control
limit. The primary representation contrast is width-512 world model versus
its matched scorer at 32-step lookahead. Capacity/horizon comparisons are
exploratory and pointwise intervals are not corrected for multiplicity.

Up to 64 independent episodes share a policy forward pass. The sampling update
is specialized to the checkpoint's epsilon/fixed-small DDPM configuration, with
all 100 diffusion steps retained. Independent CPU noise streams are indexed by
reset, decision and candidate, then transferred to CUDA together. Candidate zero
shares its noise with the baseline. Unit tests compare the mathematical update
with pinned Diffusers and check batch invariance to subpixel tolerance. GPU
batch-size changes can produce small floating-point differences. The noise
streams differ from the historical pilot, so raw outcomes are not pooled.

GPU batch wall time is recorded as throughput. Deployment latency is measured
separately on 12 fixed training-library histories, combining single-episode
proposal and scoring measurements. Estimated serial inference cost equals the
mean measured single-decision latency times an episode's decision count. It is
explicitly labeled an estimate; gate utility uses this estimate, not amortized
batch throughput. Training cost is recorded separately and not charged in the
gate utility. Episode-gate feature extraction and evaluation are excluded from
these inference-cost estimates.

Four validation resets per stratum also receive exact candidate-ranking
diagnostics and closed-loop oracle selection at each horizon (48 oracle episodes).
These privileged simulator results remain diagnostic, not held-out main-arm
performance. Candidate-ranking diagnostics occur at the initial state only.

For each auxiliary seed and each prespecified cost preference 0, 0.01, 0.1 or 1
per inference second, fit an episode-level ridge rule choosing between the
policy and primary world-model arm on validation outcomes. Its six features are
intercept, initial coverage, normalized block mean coordinates, and relative
pusher coordinates. No condition label, mass, friction, damping, test outcome or
oracle prediction is supplied to the gate. Regularization is fixed at 1 with an
unpenalized intercept. Freeze weights before applying them to held-out resets.
This is an episode-level selector, not online adaptation or a per-action gate.
