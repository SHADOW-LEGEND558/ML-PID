"""Accepted-row targets and replacement sampling, without expensive PID searches."""
import contextlib
import io
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_gen import DatasetGenerator


class DatasetCountTests(unittest.TestCase):
    def test_failed_plants_are_replaced_with_new_samples(self):
        seen = []
        def worker(plant, starts):
            seen.append(plant)
            return None if len(seen) in (1, 4) else {'K': plant.K}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.csv'
            with patch('data_gen.ProcessPoolExecutor', ThreadPoolExecutor), \
                 patch('data_gen._process_single_plant', worker), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = DatasetGenerator(n_samples=3, output_path=str(path)).run(n_jobs=1)
            self.assertEqual(len(result), 3)
            self.assertEqual(len(path.read_text().splitlines()), 4)
            self.assertEqual(len(seen), 5)
            self.assertEqual(len(set(seen)), 5)

    def test_attempt_exhaustion_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.csv'
            path.write_text('original data')
            with patch('data_gen.ProcessPoolExecutor', ThreadPoolExecutor), \
                 patch('data_gen._process_single_plant', return_value=None) as worker, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, '0/3 valid plants after 5 attempts'):
                    DatasetGenerator(n_samples=3, max_attempts=5, output_path=str(path)).run(n_jobs=1)
            self.assertEqual(worker.call_count, 5)
            self.assertEqual(path.read_text(), 'original data')

    def test_replacement_sampling_and_results_are_reproducible(self):
        def run_once(directory, name):
            calls = [0]
            def worker(plant, starts):
                calls[0] += 1
                return None if calls[0] == 2 else {'K': plant.K, 'tau': plant.tau}
            with patch('data_gen.ProcessPoolExecutor', ThreadPoolExecutor), \
                 patch('data_gen._process_single_plant', worker), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return DatasetGenerator(n_samples=3, seed=42,
                    output_path=str(Path(directory) / name)).run(n_jobs=1)
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(run_once(directory, 'a.csv').equals(run_once(directory, 'b.csv')))


if __name__ == '__main__':
    unittest.main()
