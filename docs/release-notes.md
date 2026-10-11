# Public release notes

This repository starts with a clean public history. It packages the experiment implementation and reference data from the completed study. The manuscript and its LaTeX support files are excluded from the repository and its history.

## Packaging changes

- Portable cache/output locations and explicit command-line configuration, with dry-run support.
- Separate released-checkpoint and retrained-checkpoint paths for pilot selection.
- Protection against accidental replacement of reference results or earlier CPU outputs.
- CPU-only reanalysis, snapshot validation, CLI tests, and GitHub Actions.
- Removal of private development history, internal review notes, host-specific paths, administrative run metadata, and redundant figure/report exports.
- MIT license for original code, citation metadata, and preserved third-party notices.

Model architectures, losses, sampling logic, simulation, random seeds, data splits, evaluation matrices, and statistical estimators are unchanged. Reference measurement arrays are carried over from the completed study. The public configuration and manifest hashes differ where operational paths or metadata were changed.

## Validation scope

Release checks validate reference integrity, load the pilot checkpoints, exercise CLI behavior, run CPU simulator correctness tests, and recompute all three statistical analyses. CUDA-dependent tests are skipped on a CPU host and should be run on CUDA before claiming full GPU test coverage. The multi-hour training/evaluation studies were not rerun to package this release.

The initial packaging validation passed all six reproduction tests and five CPU simulator checks; four CUDA-dependent checks were skipped. Navigation, pilot, and GPU summary reanalysis matched the reference exactly. Figure code executed, but the packaging host lacked a publication font, so the original Arial vector assets are retained and the test renders are not published.
