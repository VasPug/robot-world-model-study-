"""Tests for the public CLI and reference checkpoint wiring; no experiment launches."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from reproduction import REPO, load_config, require_empty


class ReproductionTests(unittest.TestCase):
    def test_defaults_resolve_from_repo_and_pin_released_weights(self):
        cfg = load_config('configs/pusht/world_model_b.json', 'test', [], auxiliary=True)
        self.assertEqual(Path(cfg['input_cache']), REPO/'.cache/pusht')
        self.assertEqual(Path(cfg['auxiliary_dir']), REPO/'results/pusht-pilot')
        self.assertIn('expected_auxiliary_sha256', cfg)

    def test_explicit_retrained_weights_record_new_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = load_config('configs/pusht/world_model_a.json', 'test',
                              ['--auxiliary-dir', directory], auxiliary=True)
            self.assertEqual(cfg['auxiliary_dir'], directory)
            self.assertNotIn('expected_auxiliary_sha256', cfg)

    def test_reference_results_cannot_be_output(self):
        for directory in (REPO, REPO/'results', REPO/'results/navigation-study'):
            with self.subTest(directory=directory), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    load_config('navigation_config.json', 'test', ['--output-dir', str(directory)])

    def test_existing_runs_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            require_empty(path)
            (path/'summary.json').write_text('{}')
            with self.assertRaises(FileExistsError): require_empty(path)

    def test_entry_points_accept_dry_run_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'new-output'
            for script in ('navigation.py', 'experiment.py', 'gpu_extension/study.py'):
                with self.subTest(script=script):
                    result = subprocess.run([sys.executable, str(REPO/script), '--dry-run',
                                             '--output-dir', str(output)], cwd=directory,
                                            text=True, capture_output=True, check=True)
                    config = json.loads(result.stdout)
                    self.assertEqual(config['artifact_root'], str(output))
                    self.assertFalse(output.exists())

    def test_all_released_auxiliary_models_load(self):
        from experiment import load_aux
        for method in ('world_model_a', 'world_model', 'direct_scorer'):
            model = load_aux(REPO/'results/pusht-pilot', method)
            self.assertFalse(model.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
