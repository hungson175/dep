import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.jevbench_flash import (
    PRICE_INPUT, PRICE_OUTPUT, budget_reserve, measured_cost,
    run_public, validate_cap,
)
from deepseek_flash import BenchmarkResult


class BudgetTests(unittest.TestCase):
    def test_peak_reserve(self):
        self.assertAlmostEqual(budget_reserve(), .3000012)
        self.assertEqual(validate_cap(20), 20)
        for cap in [21, 0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): validate_cap(cap)

    def test_billed_failures_count_and_cache_conservative(self):
        r = BenchmarkResult(ok=False, usage={'input_tokens': 123, 'output_tokens': 1})
        self.assertAlmostEqual(measured_cost(r), (123 * PRICE_INPUT + PRICE_OUTPUT) / 1e6)
        r.usage.update(prompt_cache_hit_tokens=23, prompt_cache_miss_tokens=100)
        self.assertAlmostEqual(measured_cost(r), (100 * .3 + 23 * .006 + 1.2) / 1e6)

    def test_unknown_usage_not_zero(self):
        for usage in [{}, {'input_tokens': None, 'output_tokens': 1},
                      {'input_tokens': -1, 'output_tokens': 1},
                      {'input_tokens': True, 'output_tokens': 1}]:
            self.assertIsNone(measured_cost(BenchmarkResult(usage=usage)))

    def test_bad_cache_counts_do_not_discount(self):
        r = BenchmarkResult(usage={'input_tokens': 10, 'output_tokens': 1,
                                  'prompt_cache_hit_tokens': 999, 'prompt_cache_miss_tokens': 1})
        self.assertAlmostEqual(measured_cost(r), (10 * .3 + 1.2) / 1e6)


class CacheTests(unittest.TestCase):
    def test_pinned_download_and_tamper_detection(self):
        from benchmarks.jevbench_flash import SOURCE, FILES, PIN, blob_sha, fetch_source, activate_source
        contents = {name: (SOURCE / name).read_bytes() for name in FILES}
        tree = {'tree': [{'path': name, 'sha': blob_sha(data), 'type': 'blob'}
                         for name, data in contents.items()]}
        def fake_get(url, **kwargs):
            if '/git/trees/' in url:
                data = json.dumps(tree).encode()
            else:
                self.assertIn(PIN, url)
                name = url.split(PIN + '/')[1]
                data = contents[name]
            return io.BytesIO(data)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'upstream'
            with patch('urllib.request.urlopen', side_effect=fake_get) as http:
                fetch_source(source)
            self.assertEqual(http.call_count, 1 + len(FILES))
            before = list(sys.path)
            try:
                # Other adapter tests may already import the canonical cache.
                # Isolate this test of a different, temporary cache copy.
                with patch.dict(sys.modules):
                    for name in list(sys.modules):
                        if name == 'jevbench' or name.startswith('jevbench.'):
                            sys.modules.pop(name)
                    activate_source(source)
                    (source / FILES[0]).write_text('tampered')
                    with self.assertRaisesRegex(ValueError, 'source changed'): activate_source(source)
            finally:
                sys.path[:] = before

    def test_download_hash_mismatch_writes_nothing(self):
        from benchmarks.jevbench_flash import FILES, fetch_source
        tree = {'tree': [{'path': name, 'sha': 'bad', 'type': 'blob'} for name in FILES]}
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            with patch('urllib.request.urlopen', side_effect=[io.BytesIO(json.dumps(tree).encode()),
                    io.BytesIO(b'bad-content')]):
                with self.assertRaisesRegex(ValueError, 'blob hash mismatch'): fetch_source(source)
            self.assertFalse(source.exists())

    def test_wrong_manifest_and_import_collision(self):
        from benchmarks.jevbench_flash import SOURCE, activate_source, fetch_source
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / 'source_manifest.json').write_text('{"pin":"bad","blobs":{}}')
            with self.assertRaisesRegex(ValueError, 'manifest mismatch'): activate_source(source)
        fake = unittest.mock.Mock(__file__='/tmp/different/jevbench/__init__.py')
        with patch.dict(sys.modules, {'jevbench': fake}):
            with self.assertRaisesRegex(ValueError, 'different jevbench'): activate_source(SOURCE)
        with patch('benchmarks.jevbench_flash.activate_source') as activate:
            fetch_source(SOURCE)
            activate.assert_called_once()


class CliTests(unittest.TestCase):
    def test_dry_run_and_limit_are_offline(self):
        from benchmarks.jevbench_flash import main
        with patch('urllib.request.urlopen', side_effect=AssertionError('unexpected network')):
            self.assertEqual(main(['dry-run', '--limit', '2']), 0)
            self.assertEqual(main(['dry-run', '--limit', '2', '--missing-policy', 'error']), 0)
        with self.assertRaises(SystemExit): main(['dry-run', '--limit', '0'])
        with patch('benchmarks.jevbench_flash.fetch_source') as fetch:
            self.assertEqual(main(['fetch']), 0)
            fetch.assert_called_once()

    def test_run_cli_passes_client_and_honors_incomplete_status(self):
        from benchmarks.jevbench_flash import main
        with patch('benchmarks.jevbench_flash.DeepSeekFlashClient.from_env') as client, \
                patch('benchmarks.jevbench_flash.run_public', return_value={'complete': False, 'accuracy': None}) as run, \
                patch('jevbench.budget.Ledger') as ledger:
            ledger.return_value.charged = 0
            self.assertEqual(main(['run', '--limit', '1', '--output', '/tmp/test_cli_only']), 2)
            self.assertEqual(len(run.call_args.args[0]), 1)
            client.assert_called_once()
        with self.assertRaises(ValueError): main(['run', '--cap-usd', '21'])

    def test_dataset_validity(self):
        from benchmarks.jevbench_flash import load_public_tasks
        tasks = load_public_tasks()
        self.assertEqual(len(tasks), 231)
        with patch('jevbench.tasks.load_jsonl', return_value=[tasks[0]]):
            with self.assertRaisesRegex(ValueError, 'duplicate'): load_public_tasks()


class RunTests(unittest.TestCase):
    def setUp(self):
        from benchmarks.jevbench_flash import activate_source
        activate_source()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def task(self, n=1):
        from jevbench.tasks import Task
        return Task(id=f'test-{n}', family='intent', state='hello',
                    question={'type': 'choice', 'instructions': 'Pick', 'criteria': {'a': 'A', 'b': 'B'}},
                    labels=['a', 'b'], expected='a', split='public')

    def test_mocked_e2e_records_metrics_and_no_overwrite(self):
        from jevbench.budget import Ledger
        tasks = [self.task(), self.task(2)]
        adapter = unittest.mock.Mock()
        adapter.missing_policy = 'zero'
        adapter.run.side_effect = [
            BenchmarkResult(ok=True, probs={'a': .8, 'b': .2}, latency_s=.1,
                            usage={'input_tokens': 100, 'output_tokens': 1}),
            BenchmarkResult(error_kind='distribution', error='missing candidate', status=200,
                            usage={'input_tokens': 100, 'output_tokens': 1}),
        ]
        ledger = Ledger(self.root / 'ledger.jsonl', 20)
        out = self.root / 'run'
        summary = run_public(tasks, adapter, ledger, out)
        self.assertEqual(summary['n_attempted'], 2)
        self.assertEqual(summary['n_correct'], 1)
        self.assertEqual(summary['accuracy'], .5)
        self.assertEqual(summary['scope'], 'public_authored_legacy_not_official_v1.6')
        self.assertTrue(summary['complete'])
        manifest = json.loads((out / 'manifest.json').read_text())
        self.assertEqual(manifest['settings']['missing_policy'], 'zero')
        self.assertEqual(len((out / 'records.jsonl').read_text().splitlines()), 2)
        self.assertAlmostEqual(ledger.charged, 2 * (100 * .3 + 1.2) / 1e6)
        with self.assertRaises(FileExistsError): run_public(tasks, adapter, ledger, out)
        self.assertEqual(adapter.run.call_count, 2)

    def test_budget_stops_before_inference(self):
        from jevbench.budget import Ledger
        adapter = unittest.mock.Mock()
        summary = run_public([self.task()], adapter, Ledger(self.root / 'ledger.jsonl', .1), self.root / 'run')
        adapter.run.assert_not_called()
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['stop_reason'], 'budget')

    def test_zero_fill_http_to_record_seam(self):
        from deepseek_flash import DeepSeekFlashClient, JevBenchAdapter
        from jevbench.budget import Ledger
        response = {'model': 'deepseek-flash',
                    'usage': {'prompt_tokens': 20, 'completion_tokens': 1},
                    'choices': [{'logprobs': {'content': [
                        {'top_logprobs': [{'token': '1', 'logprob': 0.0}]}]}}]}
        adapter = JevBenchAdapter(DeepSeekFlashClient('TEST_ONLY'))
        out = self.root / 'zero_fill'
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as http:
            summary = run_public([self.task()], adapter,
                                 Ledger(self.root / 'ledger.jsonl', 20), out)
        self.assertEqual(http.call_count, 1)
        self.assertEqual(summary['zero_filled_decisions'], 1)
        self.assertEqual(summary['missing_policy'], 'zero')
        self.assertEqual(summary['accuracy'], 1.0)
        row = json.loads((out / 'records.jsonl').read_text())
        self.assertEqual(row['probs'], {'a': 1.0, 'b': 0.0})
        self.assertEqual(row['missing_options'], ['b'])
        self.assertEqual(row['missing_policy'], 'zero')

    def test_access_and_transport_stops(self):
        from jevbench.budget import Ledger
        for status, expected_calls in [(401, 1), (403, 1), (429, 1), (500, 3)]:
            adapter = unittest.mock.Mock()
            adapter.run.return_value = BenchmarkResult(status=status, error_kind='transport')
            out = self.root / f'run-{status}'
            summary = run_public([self.task(n) for n in range(5)], adapter,
                                 Ledger(self.root / f'ledger-{status}.jsonl', 20), out)
            self.assertEqual(adapter.run.call_count, expected_calls)
            self.assertFalse(summary['complete'])

    def test_semantic_failures_do_not_trigger_transport_stop(self):
        from jevbench.budget import Ledger
        adapter = unittest.mock.Mock()
        adapter.run.return_value = BenchmarkResult(status=200, error_kind='distribution',
            usage={'input_tokens': 100, 'output_tokens': 1})
        summary = run_public([self.task(n) for n in range(4)], adapter,
                             Ledger(self.root / 'ledger.jsonl', 20), self.root / 'run')
        self.assertEqual(adapter.run.call_count, 4)
        self.assertTrue(summary['complete'])

    def test_exception_preserves_unknown_bill(self):
        from jevbench.budget import Ledger
        adapter = unittest.mock.Mock()
        adapter.run.side_effect = TimeoutError('PRIVATE_ERROR')
        ledger = Ledger(self.root / 'ledger.jsonl', 20)
        summary = run_public([self.task()], adapter, ledger, self.root / 'run')
        self.assertAlmostEqual(ledger.charged, budget_reserve())
        self.assertNotIn('PRIVATE_ERROR', (self.root / 'run' / 'records.jsonl').read_text())


if __name__ == '__main__':
    unittest.main()
