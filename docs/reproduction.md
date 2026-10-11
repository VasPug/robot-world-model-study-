# Reproduction guide

## What is reproduced

The repository contains three distinct protocols. Navigation evaluates 4,608 episodes. The CPU Push-T pilot evaluates 24 paired episodes for each of four methods. The extended manipulation study evaluates 22,000 test episodes, 2,640 validation episodes, and 48 privileged oracle diagnostics. Repeated methods and seeds share resets; these counts are not counts of independent tasks. Keep pilot and extension results separate.

There are two levels of reproduction:

1. **Reference-data reproduction** checks file hashes and episode coverage, recomputes statistics from saved episodes, and rebuilds figures. It needs only CPU and does not download a policy or dataset.
2. **Experiment reproduction** downloads the pinned inputs, trains auxiliary models, and evaluates the recorded configurations. The GPU study requires CUDA. This release was validated without rerunning the full experiments.

## Requirements and setup

- Python 3.11 and uv. `uv sync --locked` installs the CPU environment, including PyTorch 2.5.1. The Python requirement and `.python-version` select 3.11.
- Linux is the reference CPU platform. On Windows, use WSL2 for `experiment.py`, which uses POSIX process-resource accounting.
- The CUDA environment uses PyTorch 2.5.1 with CUDA 12.4 wheels and a compatible NVIDIA driver. `uv sync --project gpu_extension --locked` creates its separate environment. A system CUDA toolkit is not required for these prebuilt PyTorch wheels.
- External assets are fetched on the first manipulation run from the official Diffusion Policy host. Their expected hashes are in `assets.json`. Downloads, expanded data, environments, checkpoints, and per-decision traces need additional disk space beyond the Git checkout.
- Full configurations are substantial runs. The GPU reference was produced on an RTX 4070 with four CPU threads and batches of up to 64 concurrent episodes. Runtime depends on the host. This release adds no claim that a fresh execution is bitwise identical across operating systems or hardware.

Run commands from the repository root. Shell wrappers also locate the root themselves. Relative paths inside configuration files are interpreted relative to this repository; explicit CLI paths are interpreted relative to the current working directory.

## Configuration and outputs

All experiment entry points accept:

```text
--config PATH       JSON configuration
--output-dir PATH   new output location
--cache-dir PATH    shared download cache (manipulation only)
--dry-run           print effective config; no downloads, training, or evaluation
```

Configuration defaults preserve the recorded seeds, data sizes, optimizer settings, evaluation matrices, and task budgets. Only operational paths were made portable. The CPU recipes refuse to overwrite an existing method/run output. The GPU recipe creates a unique timestamped subdirectory and prints its path. Reference data under `results/` cannot be selected as an output target.

## Navigation

```bash
bash run_navigation.sh --output-dir outputs/navigation-study
uv run --locked python scripts/analyze_navigation.py --results-root outputs --output-dir outputs/analysis-new
```

The analysis command expects `navigation-study/episodes.jsonl` and `navigation-study/gate_results.json` under `--results-root`. It writes `navigation_paired_analysis.json` to `--output-dir`. The default command, without arguments, analyzes the bundled reference snapshot.

## CPU Push-T pilot

### Reproduce selection with the released auxiliary weights

```bash
bash run.sh --config configs/pusht/world_model_a.json
bash run.sh --config configs/pusht/world_model_b.json
bash run.sh --config configs/pusht/direct_scorer.json
```

These use the included `results/pusht-pilot/*.pt` checkpoints, verified against the hashes in the method configurations. Outputs are `outputs/pusht-pilot/world_model_a/`, `world_model/`, and `direct_scorer/`. The internal key `world_model` denotes direct multi-step model B.

### Retrain auxiliaries and evaluate all methods

```bash
bash run.sh --config configs/pusht/baseline.json --output-dir outputs/pilot-retrained
bash run.sh --config configs/pusht/world_model_a.json --output-dir outputs/pilot-retrained --auxiliary-dir outputs/pilot-retrained
bash run.sh --config configs/pusht/world_model_b.json --output-dir outputs/pilot-retrained --auxiliary-dir outputs/pilot-retrained
bash run.sh --config configs/pusht/direct_scorer.json --output-dir outputs/pilot-retrained --auxiliary-dir outputs/pilot-retrained
```

The baseline command trains all three auxiliary models before policy evaluation. The selectors use the same newly trained checkpoints when `--auxiliary-dir` is supplied. Their actual hashes are logged. Retrained bytes are not required to match the released hashes, because training and serialization can differ across platforms. Without this flag the released hashes remain mandatory.

For reanalysis of new pilot runs, use a common output root named `pusht-pilot`, then pass its parent to `scripts/analyze_pusht.py --results-root`. The default script analyzes the included reference outcomes.

## Extended manipulation: capacity, horizon, and physics

Linux:

```bash
bash run_gpu.sh --output-dir outputs/gpu-horizon-study
```

Windows PowerShell:

```powershell
.\run_gpu.ps1 --output-dir outputs/gpu-horizon-study
```

Direct command on either platform:

```bash
uv run --project gpu_extension --locked python -m gpu_extension.study --output-dir outputs/gpu-horizon-study
```

The command downloads/validates inputs, builds train/validation examples, trains 18 auxiliary networks, measures deployment latency, evaluates 54 learned arms plus policy, runs oracle diagnostics, and fits/evaluates episode gates. It uses the full locked configuration and refuses a silent CPU fallback. All controller weights remain frozen during evaluation; gate training includes shifted validation outcomes, as detailed in the protocol.

To reanalyze a completed run on CPU, substitute its printed directory:

```bash
uv run --locked python scripts/analyze_gpu.py --results-dir outputs/gpu-horizon-study/RUN-DIRECTORY --output-dir outputs/analysis-new/gpu --check-reference
```

This accepts either the raw `episodes.csv` produced by a new run or the compressed reference CSV. It checks equality of all reported group statistics, paired bootstrap intervals, and gate results against that run's saved summary. Peak GPU memory is not compared during CPU reanalysis.

## Reference validation and figures

```bash
uv run --locked python scripts/verify_results.py
uv run --locked python scripts/analyze_navigation.py
uv run --locked python scripts/analyze_pusht.py
uv run --locked python scripts/analyze_gpu.py --check-reference
uv run --locked python figs/make_gpu_paper_figure.py
```

`verify_results.py` validates the snapshot hashes, complete episode keys, published success rates, pilot checkpoint hashes, figure data, and vendored source hashes. The manifest covers **reference inputs**, not generated figure binaries, so figure regeneration does not invalidate it. Do not regenerate the manifest to hide an unexplained mismatch.

The figure script uses the reference summaries in `figs/data/`. To plot a new run, deliberately update those inputs in a separate working copy after checking its protocol and outputs. Do not replace reference outcomes with new runs while presenting the reference results.

For figure regeneration, install Arial, Helvetica, Liberation Sans, or TeX Gyre Heros and inspect the figure audit. Matching the original Arial face is needed for matching typography. Font availability can change regenerated figure bytes or layout. The committed vector figures are the reference rendering.

## What is and is not included

Included: all experiment implementations, committed dependency locks, public asset URLs and hashes, compact episode records, aggregate statistics, pilot auxiliary checkpoints, protocols, and figure code. The manuscript is not included.

Downloaded on demand: the upstream policy checkpoint and demonstration dataset.

Recreated by running the GPU study: its training arrays, 18 model checkpoints, and large per-decision trace. They are not part of this compact release. A successful snapshot check establishes consistency of the bundled evidence, not convergence of every model or universal necessity of world models.
