import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from benchmarks.jevbench_flash import load_public_tasks
from benchmarks.jevbench_bonsai import (
    BonsaiAdapter, load_recipe, main, run_public, verify_model,
)


def task(kind='choice', number=1):
    from jevbench.tasks import Task
    q = {'type': kind, 'instructions': 'Pick?'}
    if kind == 'choice':
        q['criteria'], labels, gold = {'a': 'Alpha', 'b': 'Beta'}, ['a', 'b'], 'a'
    elif kind == 'score':
        q['criteria'], labels, gold = ['Low', 'High'], ['0', '1'], 1
    else:
        q['criteria'], labels, gold = {'true': 'match', 'false': 'not match'}, ['no', 'yes'], 'yes'
    return Task(f'test-{number}', 'intent', 'hello 中文', q, labels, gold, 'public')


class RecipeTests(unittest.TestCase):
    def test_safe_ast_load_has_no_dotenv_or_network_side_effect(self):
        import builtins
        real_open = builtins.open
        opened = []
        def guarded_open(path, *args, **kwargs):
            opened.append(str(path))
            if str(path).endswith('.env'):
                raise AssertionError('dotenv read')
            return real_open(path, *args, **kwargs)
        with patch.dict(os.environ, {'DEP_API_KEY': 'UNRELATED_SECRET', 'DEP_LLAMA_URL': 'https://wrong'}, clear=False), \
                patch('builtins.open', side_effect=guarded_open), \
                patch('urllib.request.urlopen', side_effect=AssertionError('network')):
            recipe = load_recipe()
            self.assertEqual(os.environ['DEP_API_KEY'], 'UNRELATED_SECRET')
            self.assertEqual(os.environ['DEP_LLAMA_URL'], 'https://wrong')
        self.assertEqual(recipe.KEY, '')
        self.assertEqual(recipe.URL, 'http://127.0.0.1:17095')
        self.assertEqual(opened, [])

    def test_live_load_reads_only_authorized_key_and_restores_environment(self):
        with patch.dict(os.environ, {'DEP_API_KEY': 'UNRELATED_SECRET', 'DEP_API_KEY_FILE': '/wrong'}), \
                patch('os.path.exists', return_value=True), \
                patch('builtins.open', return_value=io.StringIO('LOCAL_TEST_KEY\n')) as opened:
            recipe = load_recipe(live=True)
            self.assertEqual(os.environ['DEP_API_KEY'], 'UNRELATED_SECRET')
            self.assertEqual(os.environ['DEP_API_KEY_FILE'], '/wrong')
        self.assertEqual(recipe.KEY, 'LOCAL_TEST_KEY')
        self.assertEqual(opened.call_args.args[0], '/home/hungson175/.claude/.secrets/bonsai-apikey.txt')

    def test_unexpected_source_structure_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'other.py'
            source.write_text('x = 1\n')
            with self.assertRaisesRegex(ValueError, 'dotenv'):
                load_recipe(source=source)

    def test_server_identity_preflight_is_metadata_only(self):
        recipe = load_recipe()
        recipe.KEY = 'NOT_FOR_ARTIFACTS'
        with patch('urllib.request.urlopen', return_value=io.BytesIO(b'{"data":[{"id":"bonsai-2-27b"}]}')) as http:
            identity = verify_model(recipe)
        self.assertEqual(identity['aliases'], ['bonsai-2-27b'])
        self.assertEqual(http.call_args.args[0].full_url, 'http://127.0.0.1:17095/v1/models')
        self.assertNotIn('NOT_FOR_ARTIFACTS', json.dumps(identity))
        with patch('urllib.request.urlopen', return_value=io.BytesIO(b'{"data":[{"id":"wrong"}]}')):
            with self.assertRaisesRegex(ValueError, 'identity'): verify_model(recipe)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        load_public_tasks()
        self.recipe = load_recipe()
        self.adapter = BonsaiAdapter(self.recipe)

    def test_prepare_uses_original_primitives_without_gold_leakage(self):
        for kind in ['choice', 'noul', 'score']:
            t = task(kind)
            original = self.recipe.decide
            with patch.object(self.recipe, '_post', side_effect=AssertionError('network')):
                prompt, labels, rename = self.adapter.prepare(t)
                t.expected, t.provenance = 'HIDDEN_GOLD', {'rationale': 'HIDDEN_REASON'}
                self.assertEqual(self.adapter.prepare(t), (prompt, labels, rename))
            self.assertIs(self.recipe.decide, original)
            self.assertNotIn('HIDDEN', prompt)
            self.assertIn('<think></think>Answer:', prompt)
            self.assertIn('hello 中文', prompt)
            self.assertEqual(set(rename.values()), set(t.labels))
            if kind == 'noul': self.assertEqual(rename, {'true': 'yes', 'false': 'no'})
            if kind == 'score': self.assertEqual(rename, {'1': '0', '2': '1'})

    def fake_http(self, url, **kwargs):
        body = json.loads(url.data)
        if url.full_url.endswith('/tokenize'):
            token = body['content'].strip()
            ids = {'1': 101, '2': 102, 'Yes': 101, 'No': 102}
            return io.BytesIO(json.dumps({'tokens': [ids[token]]}).encode())
        self.assertTrue(url.full_url.endswith('/completion'))
        self.assertEqual(body['n_predict'], 1)
        self.assertEqual(body['temperature'], 1.0)
        self.assertTrue(body['post_sampling_probs'])
        self.assertEqual(body['logit_bias'], [[101, 50.0], [102, 50.0]])
        return io.BytesIO(json.dumps({'model': 'bonsai-2-27b', 'tokens_evaluated': 42,
            'tokens_predicted': 1, 'timings': {'predicted_n': 1},
            'completion_probabilities': [{'top_probs': [{'id': 101, 'prob': .7}, {'id': 102, 'prob': .3}]}]}).encode())

    def test_http_tokenization_completion_and_canonical_mapping_seam(self):
        self.recipe.KEY = 'SECRET_AUTH'
        with patch('urllib.request.urlopen', side_effect=self.fake_http) as http:
            result = self.adapter.run(task())
        self.assertTrue(result.ok)
        self.assertEqual(result.probs, {'a': .7, 'b': .3})
        self.assertEqual(result.candidate_mass, 1.0)
        self.assertEqual(result.usage, {'input_tokens': 42, 'output_tokens': 1})
        self.assertEqual(http.call_count, 3)
        self.assertEqual(len(result.exchanges), 3)
        self.assertNotIn('SECRET_AUTH', json.dumps(result.exchanges))
        self.assertEqual(result.status, 200)
        with patch('urllib.request.urlopen', side_effect=self.fake_http):
            noul = self.adapter.run(task('noul'))
            score = self.adapter.run(task('score'))
        self.assertEqual(noul.probs, {'yes': .7, 'no': .3})
        self.assertEqual(score.probs, {'0': .7, '1': .3})

    def test_input_errors_are_offline(self):
        for changed in [{'type': 'unknown'}, {'type': 'choice', 'criteria': {}},
                        {'type': 'score', 'criteria': []}]:
            t = task(); t.question = changed
            with patch('urllib.request.urlopen', side_effect=AssertionError('network')):
                self.assertEqual(self.adapter.run(t).error_kind, 'input')
        t = task(); t.labels = ['mismatch']
        with self.assertRaises(ValueError): self.adapter.prepare(t)

    def test_distribution_failure_retains_raw_evidence_and_no_retry(self):
        bad = {'completion_probabilities': [{'top_probs': [{'id': 101, 'prob': 1.0}]}]}
        with patch.object(self.recipe, 'token_id', side_effect=[101, 102]), \
                patch.object(self.recipe, '_post', return_value=bad) as post:
            r = self.adapter.run(task())
        self.assertFalse(r.ok)
        self.assertEqual(r.error_kind, 'distribution')
        self.assertEqual(r.missing_options, ['b'])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(r.status, 200)
        self.assertEqual(r.exchanges[0]['response'], bad)
        for top in [[], [{'id': 101, 'prob': 0}, {'id': 102, 'prob': 0}],
                    [{'id': 101, 'prob': -1}, {'id': 102, 'prob': 2}],
                    [{'id': 101, 'prob': float('nan')}, {'id': 102, 'prob': .5}]]:
            response = {'completion_probabilities': [{'top_probs': top}]}
            with patch.object(self.recipe, 'token_id', side_effect=[101, 102]), \
                    patch.object(self.recipe, '_post', return_value=response):
                self.assertEqual(self.adapter.run(task()).error_kind, 'distribution')

    def test_transport_errors_sanitized_without_retries(self):
        for error, code in [(TimeoutError('SECRET_DETAIL'), None),
                            (HTTPError('http://local', 401, 'SECRET_DETAIL', {}, None), 401)]:
            with patch.object(self.recipe, '_post', side_effect=error) as post:
                r = self.adapter.run(task())
            self.assertEqual(r.error_kind, 'transport')
            self.assertEqual(r.status, code)
            self.assertEqual(post.call_count, 1)
            self.assertNotIn('SECRET_DETAIL', json.dumps(r.exchanges))
            self.assertNotIn('SECRET_DETAIL', r.error)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        load_public_tasks()
        self.recipe = load_recipe()
        self.adapter = BonsaiAdapter(self.recipe)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self): self.tmp.cleanup()

    def test_mocked_end_to_end_retains_all_failures_and_hashes(self):
        from benchmarks.jevbench_bonsai import Result
        results = [Result(ok=True, probs={'a': .8, 'b': .2}, latency_s=.1, candidate_mass=1),
                   Result(error_kind='distribution', error='bad', latency_s=.2)]
        with patch.object(self.adapter, 'run', side_effect=results) as run:
            summary = run_public([task(), task(number=2)], self.adapter, self.root / 'run')
        self.assertEqual(run.call_count, 2)
        self.assertEqual(summary['n_correct'], 1)
        self.assertEqual(summary['accuracy'], .5)
        self.assertEqual(summary['schema_validity'], .5)
        self.assertIsNone(summary['price_per_1000_decisions_usd'])
        self.assertIsNone(summary['official_score'])
        self.assertTrue(summary['complete'])
        manifest = json.loads((self.root / 'run' / 'manifest.json').read_text())
        self.assertEqual(manifest['settings']['logit_bias'], 50.0)
        self.assertEqual(len(manifest['task_ids']), 2)
        self.assertEqual(len(manifest['dataset_hash']), 64)
        rows = [json.loads(x) for x in (self.root / 'run' / 'records.jsonl').read_text().splitlines()]
        self.assertTrue(all(len(r['task_sha256']) == 64 for r in rows))
        self.assertTrue(all(r['cost_usd'] is None for r in rows))
        self.assertEqual(len(list((self.root / 'run' / 'raw').glob('*.json'))), 2)
        with self.assertRaises(FileExistsError): run_public([task()], self.adapter, self.root / 'run')

    def test_stop_on_access_or_three_consecutive_transport_failures(self):
        from benchmarks.jevbench_bonsai import Result
        for status, count in [(401, 1), (403, 1), (429, 1), (500, 3)]:
            with patch.object(self.adapter, 'run', return_value=Result(error_kind='transport', status=status)) as run:
                summary = run_public([task(number=i) for i in range(5)], self.adapter, self.root / str(status))
            self.assertEqual(run.call_count, count)
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['stop_reason'], 'access_or_transport')

    def test_transport_counter_resets_and_unexpected_exception_is_safe(self):
        from benchmarks.jevbench_bonsai import Result
        with patch.object(self.adapter, 'run', side_effect=[TimeoutError('PRIVATE'),
            Result(error_kind='distribution'), Result(error_kind='transport'),
            Result(ok=True, probs={'a': .8, 'b': .2})]):
            summary = run_public([task(number=i) for i in range(4)], self.adapter, self.root / 'resets')
        self.assertTrue(summary['complete'])
        self.assertNotIn('PRIVATE', (self.root / 'resets' / 'records.jsonl').read_text())

    def test_offline_cli_and_bounded_run_cli(self):
        with patch('urllib.request.urlopen', side_effect=AssertionError('network')):
            self.assertEqual(main(['dry-run']), 0)
            self.assertEqual(main(['dry-run', '--limit', '2']), 0)
        with self.assertRaises(SystemExit): main(['dry-run', '--limit', '0'])
        with patch('benchmarks.jevbench_bonsai.load_recipe', return_value=self.recipe) as load, \
                patch('benchmarks.jevbench_bonsai.verify_model', return_value={'aliases': ['bonsai-2-27b']}) as verify, \
                patch('benchmarks.jevbench_bonsai.run_public', return_value={'complete': False, 'accuracy': None}) as run:
            self.assertEqual(main(['run', '--limit', '1', '--output', str(self.root / 'cli')]), 2)
            self.assertEqual(len(run.call_args.args[0]), 1)
            load.assert_called_once_with(live=True)
            verify.assert_called_once()


if __name__ == '__main__': unittest.main()
