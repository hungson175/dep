import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from benchmarks.jevbench_flash import load_public_tasks

FIXTURE = Path(__file__).parent / 'fixtures/luna_subscription.yaml'


class LunaSubscriptionTests(unittest.TestCase):
    def test_request_all_types_no_gold_or_fake_probabilities(self):
        from dep_luna import LunaAdapter, SubscriptionClient
        adapter = LunaAdapter(SubscriptionClient())
        for kind in ('choice', 'noul', 'score'):
            task = next(t for t in load_public_tasks() if t.question['type'] == kind)
            body, labels, rename = adapter.build(task)
            self.assertEqual(body['model'], 'gpt-6-luna')
            self.assertEqual(body['reasoning'], {'effort': 'none'})
            for forbidden in ('top_logprobs', 'max_tokens', 'max_output_tokens'):
                self.assertNotIn(forbidden, body)
            old = json.dumps(body)
            task.expected = 'HIDDEN_GOLD'
            self.assertEqual(json.dumps(adapter.build(task)[0]), old)
            self.assertEqual(set(rename.values()), set(task.labels))
            mark = next(iter(labels.values()))
            with patch.object(adapter.client, 'post', return_value=events(mark)):
                result = adapter.run(task)
            self.assertTrue(result.ok)
            self.assertIsNone(result.probs)
            self.assertEqual(result.label, rename[next(iter(labels))])
            self.assertEqual(result.probs_source, 'label_only_no_calibrated_distribution')

    def test_fails_closed_on_reasoning_model_tools_missing_events(self):
        from dep_luna import LunaAdapter, SubscriptionClient
        adapter = LunaAdapter(SubscriptionClient())
        task = load_public_tasks()[0]
        variants = [events('unknown'), events('1 extra'), events('1', reasoning=1),
                    events('1', model='gpt-5.6-luna'), [], events('1', status='incomplete'),
                    events('1', item_type='function_call'), events('1', reasoning=None)]
        for response in variants:
            with patch.object(adapter.client, 'post', return_value=response):
                result = adapter.run(task)
            self.assertFalse(result.ok)
            self.assertIsNone(result.probs)
        with patch.object(adapter.client, 'post', side_effect=HTTPError('url', 429, 'x', {}, None)):
            self.assertEqual(adapter.run(task).status, 429)
        with patch.object(adapter.client, 'post', side_effect=URLError('offline')):
            self.assertFalse(adapter.run(task).ok)

    def test_selective_auth_and_vcr_transport_replay(self):
        import vcr
        from dep_luna import SubscriptionClient, LunaAdapter
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'auth.json'
            p.write_text(json.dumps({'auth_mode': 'chatgpt', 'tokens': {
                'access_token': 'TEST_ONLY', 'account_id': 'TEST_ACCOUNT'}}))
            client = SubscriptionClient(p)
            adapter = LunaAdapter(client)
            with vcr.use_cassette(str(FIXTURE), record_mode='none',
                                 match_on=['method', 'scheme', 'host', 'port', 'path', 'query', 'body'],
                                 filter_headers=['authorization', 'chatgpt-account-id']):
                result = adapter.run(load_public_tasks()[0])
            self.assertTrue(result.ok)
            self.assertEqual(result.label, 'track_order')
            self.assertEqual(result.usage['reasoning_tokens'], 0)
            self.assertNotIn('TEST_ONLY', json.dumps(result.raw))
            p.write_text('{"auth_mode":"apikey"}')
            with self.assertRaises(ValueError): client.post({})
            p.write_text('{"auth_mode":"chatgpt","tokens":{}}')
            with self.assertRaises(ValueError): client.post({})

    def test_full_runner_scoring_and_stop(self):
        from dep_luna import LunaAdapter, SubscriptionClient
        from benchmarks.jevbench_luna import run
        adapter = LunaAdapter(SubscriptionClient())
        tasks = load_public_tasks()[:3]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(adapter.client, 'post', return_value=events('1')):
                summary = run(tasks, adapter, root / 'ok')
            self.assertTrue(summary['complete'])
            self.assertIsNone(summary['brier_mean'])
            self.assertIsNone(summary['ece'])
            self.assertIsNone(summary['official_score'])
            self.assertIsNone(summary['cost_usd'])
            with patch.object(adapter.client, 'post', side_effect=HTTPError('url', 401, 'x', {}, None)):
                summary = run(tasks, adapter, root / 'blocked')
            self.assertEqual(summary['n_attempted'], 1)
            self.assertFalse(summary['complete'])
            with patch.object(adapter.client, 'post', side_effect=URLError('offline')):
                summary = run(tasks, adapter, root / 'transport')
            self.assertEqual(summary['n_attempted'], 3)
            with self.assertRaises(ValueError): run(tasks * 100, adapter, root / 'oversized')

    def test_cli_paths(self):
        from benchmarks.jevbench_luna import main
        with patch('sys.argv', ['run', '--limit', '0']):
            with self.assertRaises(SystemExit): main()
        for complete in (True, False):
            with patch('sys.argv', ['run', '--limit', '1']), patch(
                'benchmarks.jevbench_luna.run', return_value={'complete': complete, 'accuracy': 1}):
                self.assertEqual(main(), 0 if complete else 2)


def events(text, reasoning=0, model='gpt-6-luna', status='completed', item_type='message'):
    return [{'type': 'response.output_item.done', 'item': {
        'type': item_type, 'content': [{'type': 'output_text', 'text': text}]}},
        {'type': 'response.completed', 'response': {'model': model, 'status': status,
         'usage': {'input_tokens': 20, 'output_tokens': 5,
                   'output_tokens_details': {'reasoning_tokens': reasoning}}}}]


if __name__ == '__main__': unittest.main()
