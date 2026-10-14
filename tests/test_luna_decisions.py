import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from benchmarks.jevbench_flash import load_public_tasks


class DecisionsTests(unittest.TestCase):
    def test_native_three_type_mapping_and_no_gold(self):
        from dep_luna_decisions import DecisionsAdapter, DecisionsClient
        adapter = DecisionsAdapter(DecisionsClient('TEST_ONLY'))
        for kind in ('choice', 'noul', 'score'):
            task = next(t for t in load_public_tasks() if t.question['type'] == kind)
            body = adapter.build(task)
            self.assertEqual(body['model'], 'gpt-6-luna')
            old = json.dumps(body)
            task.expected = 'HIDDEN_GOLD'
            self.assertEqual(json.dumps(adapter.build(task)), old)
            question = body['questions'][0]
            self.assertEqual(question['name'], 'decision')
            answer = {'name': 'decision', 'type': question['type']}
            if kind == 'noul': answer['probability'] = .75
            else: answer['probabilities'] = [{'value': int(k) if kind == 'score' else k,
                                             'probability': 1 / len(task.labels)} for k in task.labels]
            response = {'model': 'gpt-6-luna', 'answers': [answer],
                        'usage': {'input_tokens': 100, 'output_tokens': 0,
                                  'output_tokens_details': {'reasoning_tokens': 0}}}
            with patch.object(adapter.client, 'post', return_value=response): result = adapter.run(task)
            self.assertTrue(result.ok)
            self.assertEqual(set(result.probs), set(task.labels))
            self.assertEqual(result.probs_source, 'native_decisions_api')
            self.assertEqual(result.usage['output_tokens'], 0)
        self.assertEqual(result.probs, {k: 1 / len(task.labels) for k in task.labels})

    def test_distribution_failures_and_transport(self):
        from dep_luna_decisions import DecisionsAdapter, DecisionsClient
        adapter = DecisionsAdapter(DecisionsClient('TEST_ONLY'));task = load_public_tasks()[0]
        valid = {'model': 'gpt-6-luna', 'answers': [{'type': 'choice', 'name': 'decision',
                 'probabilities': [{'value': k, 'probability': 1 / len(task.labels)} for k in task.labels]}],
                 'usage': {'input_tokens': 100, 'output_tokens': 0,
                           'output_tokens_details': {'reasoning_tokens': 0}}}
        variants = []
        for key, value in [('model', 'wrong'), ('answers', []), ('answers', [{'type': 'refusal'}]),
                           ('usage', {'output_tokens': 1})]:
            variants.append(dict(valid, **{key: value}))
        for p in [[], [{'value': 'extra', 'probability': 1}],
                  [{'value': k, 'probability': -1} for k in task.labels],
                  [{'value': k, 'probability': True} for k in task.labels],
                  [{'value': k, 'probability': 0} for k in task.labels],
                  valid['answers'][0]['probabilities'] * 2]:
            variants.append(dict(valid, answers=[{'type': 'choice', 'name': 'decision', 'probabilities': p}]))
        for response in variants:
            with patch.object(adapter.client, 'post', return_value=response): self.assertFalse(adapter.run(task).ok)
        with patch.object(adapter.client, 'post', side_effect=HTTPError('url', 429, '', {}, None)):
            self.assertEqual(adapter.run(task).status, 429)
        with patch.object(adapter.client, 'post', side_effect=URLError('offline')):
            self.assertFalse(adapter.run(task).ok)

    def test_selective_credentials_and_http(self):
        from dep_luna_decisions import DecisionsClient, read_key
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'env';p.write_text('OTHER=secret\nOPENAI_API_KEY="TEST_ONLY" # comment\n')
            self.assertEqual(read_key(p), 'TEST_ONLY')
            for value in ['OTHER=x', 'OPENAI_API_KEY=', 'OPENAI_API_KEY=a\nOPENAI_API_KEY=b']:
                p.write_text(value)
                with self.assertRaises(ValueError): read_key(p)
        for key in ('', None, '  '):
            with self.assertRaises(ValueError): DecisionsClient(key)
        client = DecisionsClient('TEST_ONLY')
        self.assertNotIn('TEST_ONLY', repr(client))
        with patch('dep_luna_decisions.urllib.request.urlopen', return_value=io.BytesIO(b'{"model":"gpt-6-luna"}')) as call:
            self.assertEqual(client.post({})['model'], 'gpt-6-luna')
            req = call.call_args.args[0]
            self.assertEqual(req.full_url, 'https://api.openai.com/v1/decisions')
            self.assertNotIn('Authorization', req.headers)
            self.assertEqual(req.unredirected_hdrs['Authorization'], 'Bearer TEST_ONLY')

    def test_runner_cost_scoring_stop_and_cap(self):
        from benchmarks.jevbench_luna_decisions import run
        from deepseek_flash import BenchmarkResult
        tasks = load_public_tasks()[:3]
        class Fake:
            def run(self, task):
                return BenchmarkResult(ok=True, probs={k: 1 / len(task.labels) for k in task.labels},
                    model='gpt-6-luna', status=200, probs_source='native_decisions_api',
                    usage={'input_tokens': 100, 'output_tokens': 0,
                           'output_tokens_details': {'reasoning_tokens': 0}})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run(tasks, Fake(), root / 'ok')
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['calibration_n'], 3)
            self.assertAlmostEqual(summary['cost_usd'], .00003)
            self.assertIsNone(summary['official_score'])
            with patch.object(Fake, 'run', return_value=BenchmarkResult(status=401)):
                self.assertEqual(run(tasks, Fake(), root / 'blocked')['n_attempted'], 1)
            with patch.object(Fake, 'run', return_value=BenchmarkResult()):
                self.assertEqual(run(tasks, Fake(), root / 'transport')['n_attempted'], 3)
            with patch.object(Fake, 'run', return_value=BenchmarkResult(status=200, error_kind='distribution')):
                self.assertTrue(run(load_public_tasks()[:4], Fake(), root / 'refusals')['complete'])
            self.assertEqual(run(tasks, Fake(), root / 'cap', cap=.001)['n_attempted'], 0)

    def test_cli(self):
        from benchmarks.jevbench_luna_decisions import main
        with patch('sys.argv', ['run', '--limit', '0']):
            with self.assertRaises(SystemExit): main()
        for complete in (False, True):
            with patch('sys.argv', ['run']), patch('benchmarks.jevbench_luna_decisions.read_key', return_value='TEST_ONLY'), patch(
                'benchmarks.jevbench_luna_decisions.run', return_value={'complete': complete, 'accuracy': .5, 'cost_usd': 0}):
                self.assertEqual(main(), 0 if complete else 2)

    def test_real_35_option_vcr_replay(self):
        import vcr
        from types import SimpleNamespace
        from dep_luna_decisions import DecisionsAdapter, DecisionsClient
        task = SimpleNamespace(state='The target category is option_17.',
            question={'type': 'choice', 'instructions': 'Select the category explicitly named in the input.',
                      'criteria': {f'option_{i}': f'Target category option_{i}.' for i in range(1, 36)}},
            labels=[f'option_{i}' for i in range(1, 36)])
        with vcr.use_cassette(str(Path(__file__).parent / 'fixtures/luna_decisions_35.yaml'),
            record_mode='none', match_on=['method', 'scheme', 'host', 'port', 'path', 'query', 'body'],
            filter_headers=['authorization']):
            result = DecisionsAdapter(DecisionsClient('TEST_ONLY')).run(task)
        self.assertTrue(result.ok)
        self.assertEqual(len(result.probs), 35)
        self.assertAlmostEqual(sum(result.probs.values()), 1)
        self.assertEqual(max(result.probs, key=result.probs.get), 'option_17')


if __name__ == '__main__': unittest.main()
