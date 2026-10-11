"""Verify the released reference snapshot with the Python standard library only."""
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = read(RESULTS / 'manifest.json')
    for name, expected in manifest['files'].items():
        assert sha(ROOT / name) == expected, f'Snapshot checksum mismatch: {name}'
    gpu = RESULTS / 'gpu-horizon-study'
    config, summary = read(gpu / 'config.json'), read(gpu / 'summary.json')
    with gzip.open(gpu / 'episodes.csv.gz', 'rt', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for field in ['width', 'seed', 'horizon', 'env_seed', 'success', 'steps']:
            row[field] = int(row[field])
    fields = ['kind', 'width', 'seed', 'horizon', 'split', 'env_seed', 'condition']
    keys = {tuple(r[k] for k in fields) for r in rows}
    arms = [('policy', 0, 0, 8)] + list(itertools.product(
        ['world_model', 'direct_scorer'], config['evaluation_widths'],
        config['training_seeds'], config['horizons']))
    expected = {a + (split, start+i, condition)
                for a in arms for split, start, count in [('test', 820000, 100), ('validation', 810000, 12)]
                for i in range(count) for condition in config['conditions']}
    assert len(rows) == len(keys) == 24640 and keys == expected
    assert sum(r['split'] == 'test' for r in rows) == summary['test_episodes'] == 22000
    assert sum(r['split'] == 'validation' for r in rows) == summary['validation_episodes'] == 2640
    for group in summary['groups']:
        rr = [r for r in rows if r['split'] == 'test' and
              all(r[k] == group[k] for k in ['kind', 'width', 'horizon', 'condition'])]
        assert len(rr) == group['episodes']
        assert math.isclose(sum(r['success'] for r in rr)/len(rr), group['success_rate'], abs_tol=1e-12)
    oracle = read(gpu / 'oracle_episodes.json')
    oracle_keys = {(r['horizon'], r['env_seed'], r['condition']) for r in oracle}
    assert len(oracle) == len(oracle_keys) == 48
    assert oracle_keys == set(itertools.product(config['horizons'], range(810000, 810004), config['conditions']))
    assert read(ROOT / 'figs/data/gpu_summary.json') == summary

    nav = RESULTS / 'navigation-study'
    rows = [json.loads(line) for line in (nav / 'episodes.jsonl').read_text().splitlines()]
    cfg = read(nav / 'config.json')
    fields = ['training_seed', 'scenario', 'shots', 'method', 'split', 'reset_seed']
    keys = {tuple(r[k] for k in fields) for r in rows}
    arms = itertools.product(cfg['training_seeds'], cfg['scenarios'], cfg['adaptation_episodes'],
                             ['baseline', 'world_model_a', 'world_model_b', 'direct_scorer'])
    expected = {a + (split, start+i) for a in arms
                for split, start, count in [('validation', 400000, 12), ('test', 500000, 20)] for i in range(count)}
    assert len(rows) == len(keys) == 4608 and keys == expected
    for group in read(nav / 'summary.json'):
        rr = [r for r in rows if r['split'] == 'test' and
              all(r[k] == group[k] for k in ['scenario', 'shots', 'method'])]
        assert len(rr) == group['n']
        assert math.isclose(sum(r['success'] for r in rr)/len(rr), group['success'], abs_tol=1e-12)
    assert read(ROOT / 'figs/data/navigation_summary.json') == read(nav / 'summary.json')

    pilot = RESULTS / 'pusht-pilot'
    for method in ['baseline', 'world_model_a', 'world_model', 'direct_scorer']:
        rows = read(pilot / method / 'episodes.json')
        keys = {(r['env_seed'], r['condition']) for r in rows}
        assert len(rows) == len(keys) == 24
        assert keys == set(itertools.product(range(610000, 610012), ['standard', 'shifted_start']))
        assert math.isclose(sum(r['success'] for r in rows)/len(rows),
                            read(pilot / method / 'summary.json')['success_rate'], abs_tol=1e-12)
        if method != 'baseline':
            config_name = 'world_model_b' if method == 'world_model' else method
            cfg = read(ROOT / 'configs/pusht' / (config_name + '.json'))
            assert sha(pilot / (method + '.pt')) == cfg['expected_auxiliary_sha256']
    for name, expected in read(ROOT / 'vendor/provenance.json')['files'].items():
        assert sha(ROOT / 'vendor' / name) == expected, f'Upstream source changed: {name}'
    return {'status': 'passed', 'checksummed_files': len(manifest['files']),
            'navigation_episodes': 4608, 'pilot_episodes': 96,
            'extended_manipulation_episodes': 24640, 'oracle_diagnostics': 48}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
