# Reference results

These are the compact reference outputs for the study. New runs and reanalysis write to `outputs/`.

| Directory | Contents |
| --- | --- |
| `navigation-study/` | Per-episode navigation records, validation-fitted gates, training metadata, and summaries |
| `pusht-pilot/` | Four pilot methods, raw episode records, shared auxiliary checkpoints, and training metadata |
| `gpu-horizon-study/` | Compressed main episodes, oracle/ranking diagnostics, gates, timing measurements, and training metadata |
| `world-model-study/` | Paired navigation and pilot analysis summaries |

`manifest.json` contains SHA-256 hashes of every reference result file and the two figure-data summaries. Configurations retain the experimental settings with portable output/cache paths. Administrative run identifiers and machine-specific paths were omitted from this public package. Measured episode outcomes, summary metrics, and checkpoint bytes are preserved.

Run `uv run --locked python scripts/verify_results.py` to verify consistency. See [the reproduction guide](../docs/reproduction.md) for reanalysis and full experiment commands. Repeated arms and seeds are paired evaluations, not independent tasks.
