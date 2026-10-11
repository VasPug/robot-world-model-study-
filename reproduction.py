"""Portable command-line configuration shared by the experiment entry points."""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent


def load_config(default, description, argv=None, auxiliary=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=Path, default=REPO / default,
                        help='JSON configuration (default: %(default)s)')
    parser.add_argument('--output-dir', type=Path, help='Output directory; default comes from config')
    parser.add_argument('--cache-dir', type=Path, help='Downloaded dataset/checkpoint cache')
    parser.add_argument('--dry-run', action='store_true', help='Print effective configuration and exit without running')
    if auxiliary:
        parser.add_argument('--auxiliary-dir', type=Path,
                            help='Use retrained pilot weights from this directory instead of released weights')
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    for field, override in [('artifact_root', args.output_dir), ('input_cache', args.cache_dir)]:
        if field not in cfg:
            if override is not None:
                parser.error(f'{field} does not apply to this experiment')
            continue
        path = override.expanduser().resolve() if override is not None else (REPO / cfg[field]).resolve()
        cfg[field] = str(path)
    output = Path(cfg['artifact_root'])
    reference = REPO / 'results'
    if output == reference or reference in output.parents or output in reference.parents:
        parser.error('Output must not overwrite the repository or reference results')
    if auxiliary and cfg['variant'] != 'baseline':
        if args.auxiliary_dir is not None:
            cfg['auxiliary_dir'] = str(args.auxiliary_dir.expanduser().resolve())
            # Retraining can change tensor values and serialization bytes across hosts.
            # Record the actual hash at execution instead of claiming reference identity.
            cfg.pop('expected_auxiliary_sha256', None)
        else:
            cfg['auxiliary_dir'] = str(REPO / 'results/pusht-pilot')
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        parser.exit()
    return cfg


def require_empty(path):
    """Refuse to silently replace an earlier run."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'{path} is not empty. Choose a new --output-dir.')
