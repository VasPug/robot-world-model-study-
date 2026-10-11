"""Recompute paired intervals and episode gates from saved GPU-study episodes on CPU."""
import argparse
import ast
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpu_extension.batched import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'results/gpu-horizon-study')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/analysis/gpu')
    parser.add_argument('--check-reference', action='store_true', help='Require equality to the input summary')
    args = parser.parse_args()
    source, output = args.results_dir.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents:
        parser.error('Output must be separate from input results')
    output.mkdir(parents=True, exist_ok=True)
    compressed = source/'episodes.csv.gz'
    with (gzip.open(compressed, 'rt', newline='') if compressed.exists()
          else (source/'episodes.csv').open(newline='')) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if value == '': row[key] = None
            elif key not in ('kind', 'condition', 'split'): row[key] = ast.literal_eval(value)
    cfg = json.loads((source/'config.json').read_text())
    oracle = json.loads((source/'oracle_episodes.json').read_text())
    analyze(rows, cfg, output, oracle)
    if args.check_reference:
        original = json.loads((source/'summary.json').read_text())
        actual = json.loads((output/'summary.json').read_text())
        for key in ('groups', 'gates', 'oracle_validation_groups', 'primary_contrast', 'test_episodes', 'validation_episodes'):
            assert original[key] == actual[key], f'Reference mismatch: {key}'
        print('All paired intervals, gate results, and aggregate outcomes match the reference exactly.')


if __name__ == '__main__':
    main()
