import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.jevbench_flash import load_public_tasks
from benchmarks.jevbench_bonsai_native import aggregate, build_jobs, main, run_native


def result(task, **changes):
    row = {'type': 'result', 'task_id': task.id, 'ok': True,
           'probs': {k: float(k == str(task.expected)) for k in task.labels},
           'candidate_ids': {k: i for i, k in enumerate(task.labels)},
           'prompt_tokens': 100, 'tokenize_ms': 1., 'reset_ms': 2.,
           'prefill_ms': 10., 'sample_ms': 3., 'decision_ms': 13.,
           'local_total_ms': 16., 'perf_prompt_ms': 10., 'perf_eval_ms': 0.}
    row.update(changes)
    return row


class NativeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tasks = load_public_tasks()[:2]
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def test_jobs_reuse_exact_prompts_without_gold_or_credentials(self):
        from benchmarks.jevbench_bonsai import BonsaiAdapter, load_recipe
        jobs = build_jobs(self.tasks)
        adapter = BonsaiAdapter(load_recipe())
        for task, job in zip(self.tasks, jobs):
            prompt, labels, rename = adapter.prepare(task)
            self.assertEqual(set(job), {'task_id', 'prompt', 'labels'})
            self.assertEqual(job['prompt'], prompt)
            self.assertEqual(job['labels'], {rename[k]: v for k, v in labels.items()})
            task.expected, task.provenance = 'DO_NOT_LEAK', {'secret': 'DO_NOT_LEAK'}
        with patch('urllib.request.urlopen', side_effect=AssertionError('HTTP forbidden')):
            self.assertEqual(build_jobs(self.tasks), jobs)
        self.assertNotIn('DO_NOT_LEAK', json.dumps(jobs))

    def test_native_scoring_and_timing_are_not_http_roundtrip(self):
        metadata = {'type': 'metadata', 'warmup_count': 2, 'load_ms': 9000.}
        rows = [result(self.tasks[0]), result(self.tasks[1], prefill_ms=30.,
                 decision_ms=33., local_total_ms=36., perf_prompt_ms=30.)]
        summary, records = aggregate(self.tasks, [metadata, *rows])
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['n_correct'], 2)
        self.assertAlmostEqual(summary['latency']['p50_s'], .026)
        self.assertAlmostEqual(summary['native_components']['decision']['p50_s'], .023)
        self.assertAlmostEqual(summary['native_components']['prefill']['p50_s'], .02)
        self.assertAlmostEqual(summary['native_components']['local_total']['p50_s'], .026)
        self.assertEqual(summary['native_metadata']['load_ms'], 9000.)
        self.assertEqual(summary['transport'], 'in_process_libllama_no_http')
        self.assertIsNone(summary['price_per_1000_decisions_usd'])
        self.assertIsNone(summary['official_score'])
        self.assertTrue(all(r['status_code'] is None for r in records))

    def test_empty_partial_and_failed_results_are_not_a_complete_success(self):
        metadata = {'type': 'metadata'}
        summary, _ = aggregate(self.tasks, [metadata])
        self.assertFalse(summary['complete'])
        self.assertIsNone(summary['latency']['p50_s'])
        failed = {'type': 'result', 'task_id': self.tasks[1].id, 'ok': False,
                  'error_kind': 'distribution'}
        summary, records = aggregate(self.tasks, [metadata, result(self.tasks[0]), failed])
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['n_valid'], 1)
        self.assertEqual(summary['accuracy'], .5)
        self.assertEqual(summary['native_components']['decision']['n'], 1)
        self.assertEqual(records[1]['error_kind'], 'distribution')

    def test_untrusted_native_output_fails_closed(self):
        metadata = {'type': 'metadata'}
        cases = [[], [result(self.tasks[0])], [metadata, metadata],
                 [metadata, {'type': 'wrong'}],
                 [metadata, result(self.tasks[0]), result(self.tasks[0])],
                 [metadata, result(self.tasks[0], task_id='unknown')]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                aggregate(self.tasks, rows)
        for changes in [{'ok': 1}, {'probs': {}},
                        {'probs': {k: math.nan for k in self.tasks[0].labels}},
                        {'probs': {k: 2. for k in self.tasks[0].labels}},
                        {'prefill_ms': -1}, {'prefill_ms': math.inf},
                        {'decision_ms': 999}, {'local_total_ms': 1},
                        {'local_wall_ms': -1}, {'local_wall_ms': 1},
                        {'prompt_tokens': 0}, {'prompt_tokens': True}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                aggregate(self.tasks, [metadata, result(self.tasks[0], **changes)])

    def test_wall_time_is_measured_separately_from_component_sum(self):
        summary, records = aggregate(self.tasks[:1], [
            {'type': 'metadata', 'model_load_ms': 9000},
            result(self.tasks[0], local_wall_ms=17.)])
        self.assertEqual(records[0]['latency_s'], .017)
        self.assertEqual(summary['native_components']['local_total']['p50_s'], .016)

    def fake_process(self, command, **kwargs):
        self.assertEqual(command[0], str(self.root / 'native'))
        self.assertNotIn('DEEPSEEK_API_KEY', kwargs['env'])
        self.assertNotIn('ANTHROPIC_API_KEY', kwargs['env'])
        input_path = Path(command[command.index('--input') + 1])
        output = Path(command[command.index('--output') + 1])
        jobs = [json.loads(s) for s in input_path.read_text().splitlines()]
        self.assertEqual(len(jobs), 2)
        rows = [{'type': 'metadata', 'warmup_count': 2}, *map(result, self.tasks)]
        output.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return subprocess.CompletedProcess(command, 0)

    def test_mock_process_full_seam_and_no_retry_or_output_overwrite(self):
        binary = self.root / 'native'; binary.write_bytes(b'fake executable')
        model = self.root / 'model.gguf'; model.write_bytes(b'fake gguf')
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'DO_NOT_LEAK', 'ANTHROPIC_API_KEY': 'DO_NOT_LEAK'}), \
                patch('subprocess.run', side_effect=self.fake_process) as process, \
                patch('urllib.request.urlopen', side_effect=AssertionError('HTTP forbidden')):
            summary = run_native(self.tasks, binary, model, self.root / 'run')
            self.assertTrue(summary['complete'])
            self.assertEqual(process.call_count, 1)
            with self.assertRaises(FileExistsError):
                run_native(self.tasks, binary, model, self.root / 'run')
        manifest = json.loads((self.root / 'run/manifest.json').read_text())
        self.assertEqual(manifest['transport'], 'in_process_libllama_no_http')
        self.assertFalse(manifest['cross_case_kv_cache'])
        self.assertEqual(manifest['warmup_count'], 2)
        self.assertEqual(len(manifest['model_sha256']), 64)
        self.assertNotIn('DO_NOT_LEAK', (self.root / 'run/manifest.json').read_text())
        self.assertTrue((self.root / 'run/records.jsonl').exists())

    def test_native_process_failure_or_timeout_is_not_retried(self):
        binary = self.root / 'native'; binary.write_bytes(b'native')
        model = self.root / 'model.gguf'; model.write_bytes(b'model')
        for index, behavior in enumerate([subprocess.CompletedProcess([], 4),
                                         subprocess.TimeoutExpired([], 1)]):
            kwargs = {'side_effect': behavior} if isinstance(behavior, Exception) else {'return_value': behavior}
            with patch('subprocess.run', **kwargs) as process:
                with self.assertRaises(RuntimeError):
                    run_native(self.tasks, binary, model, self.root / f'failure_{index}')
                self.assertEqual(process.call_count, 1)
            self.assertTrue((self.root / f'failure_{index}/process_status.json').exists())

    def test_cli_dry_run_is_offline_and_run_dispatch_is_explicit(self):
        with patch('subprocess.run', side_effect=AssertionError('native execution')), \
                patch('sys.stdout', new=io.StringIO()) as stream:
            self.assertEqual(main(['dry-run']), 0)
            self.assertIn('231', stream.getvalue())
        stub = {'complete': True, 'n_correct': 1, 'accuracy': .5, 'latency': {}, 'native_components': {}}
        with patch('benchmarks.jevbench_bonsai_native.run_native', return_value=stub) as run, \
                patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(main(['run', '--binary', '/native', '--model', '/model', '--output', '/new']), 0)
            self.assertEqual(run.call_count, 1)
        stub['complete'] = False
        with patch('benchmarks.jevbench_bonsai_native.run_native', return_value=stub), \
                patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(main(['run', '--binary', '/native', '--model', '/model', '--output', '/new']), 1)
        with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            main(['run'])


if __name__ == '__main__': unittest.main()
