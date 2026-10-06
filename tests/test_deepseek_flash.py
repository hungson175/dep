import importlib
import json
import math
import os
import subprocess
import sys
import unittest
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

from deepseek_flash import (
    DeepSeekFlashClient, DistributionError, JevBenchAdapter,
    build_request, candidate_probs, confidence, marks,
)


def response(tokens=(('1', .8), ('2', .2)), content='1'):
    return {
        'model': 'deepseek-flash',
        'choices': [{'message': {'content': content}, 'logprobs': {
            'content': [{'top_logprobs': [
                {'token': t, 'logprob': math.log(p)} for t, p in tokens
            ]}]
        }}],
        'usage': {'prompt_tokens': 123, 'completion_tokens': 1,
                  'prompt_cache_hit_tokens': 23, 'prompt_cache_miss_tokens': 100},
    }


def task(qtype='choice', criteria=None, labels=None):
    if criteria is None:
        criteria = {'ignore': 'not relevant', 'today': 'act today'}
    return SimpleNamespace(state='Xin chào', question={
        'type': qtype, 'instructions': 'How urgent?', 'criteria': criteria,
    }, labels=labels or ['ignore', 'today'], expected='GOLD_MUST_NOT_LEAK',
        provenance={'rationale': 'RATIONALE_MUST_NOT_LEAK'})


class PromptTests(unittest.TestCase):
    def test_notebook_choice_payload(self):
        body, labels = build_request('Xin chào', task().question)
        self.assertEqual(body, {
            'model': 'deepseek-flash', 'messages': [{'role': 'user', 'content':
                'Text:\nXin chào\n\nQuestion: How urgent?\nOptions:\n'
                '1. ignore -- not relevant\n2. today -- act today\n'
                'Reply with exactly one character: 1, 2.'}],
            'max_tokens': 1, 'temperature': 1.0,
            'thinking': {'type': 'disabled'}, 'logprobs': True, 'top_logprobs': 20,
        })
        self.assertEqual(labels, {'ignore': '1', 'today': '2'})

    def test_noul_and_score_match_notebook(self):
        body, labels = build_request('text', {'type': 'noul', 'instructions': 'True?',
            'criteria': {'true': 'allowed', 'false': 'blocked'}})
        self.assertEqual(labels, {'true': 'Yes', 'false': 'No'})
        self.assertIn('true means: allowed\nfalse means: blocked', body['messages'][0]['content'])
        body, labels = build_request('text', {'type': 'score', 'instructions': 'Severity?',
                                            'criteria': ['calm', 'angry']})
        self.assertEqual(labels, {'1': '1', '2': '2'})
        self.assertIn('Scale:\n1. calm\n2. angry', body['messages'][0]['content'])

    def test_structured_state_none_and_list_criteria(self):
        body, _ = build_request({'text': 'Tiếng Việt'}, {'type': 'choice',
            'instructions': 'Pick', 'criteria': ['a', 'b']})
        self.assertIn('{"text": "Tiếng Việt"}', body['messages'][0]['content'])
        body, _ = build_request([], {'type': 'choice', 'instructions': 'Pick',
                                    'criteria': {'a': None, 'b': 'B'}})
        self.assertIn('1. a -- None', body['messages'][0]['content'])
        _, labels = build_request('', {'type': 'noul', 'instructions': 'True?'})
        self.assertEqual(len(labels), 2)

    def test_boundaries(self):
        self.assertEqual(marks(10)[-1], 'A')
        self.assertEqual(len(marks(35)), 35)
        for n in [0, 36, -1, True, 1.5]:
            with self.assertRaises(ValueError): marks(n)
        for q in [{}, {'type': 'other', 'instructions': 'x'},
                  {'type': 'choice', 'instructions': '', 'criteria': ['a']},
                  {'type': 'choice', 'instructions': 'x', 'criteria': []},
                  {'type': 'score', 'instructions': 'x', 'criteria': {'x': 'x'}},
                  {'type': 'noul', 'instructions': 'x', 'criteria': {'true': 'x'}},
                  {'type': 'choice', 'instructions': 'x', 'criteria': ['a', 'a']}]:
            with self.assertRaises(ValueError): build_request('text', q)


class DistributionTests(unittest.TestCase):
    def test_variants_summed_and_non_candidate_ignored(self):
        report = [{'token': t, 'logprob': math.log(p)} for t, p in
                  [('Yes', .2), (' yes', .3), ('NO', .1), ('Maybe', .4)]]
        p, mass = candidate_probs(report, {'true': 'Yes', 'false': 'No'})
        self.assertAlmostEqual(p['true'], 5 / 6)
        self.assertAlmostEqual(mass, .6)
        self.assertEqual(confidence(p), .6667)

    def test_underflow_not_confused_with_missing(self):
        p, mass = candidate_probs([{'token': '1', 'logprob': -1000},
                                  {'token': '2', 'logprob': -1001}], {'a': '1', 'b': '2'})
        self.assertAlmostEqual(p['a'], 1 / (1 + math.exp(-1)))
        self.assertEqual(mass, 0.0)

    def test_invalid_fail_closed(self):
        bad_reports = [[], [{'token': '1', 'logprob': -.1}],
            [{'token': '1', 'logprob': 1}, {'token': '2', 'logprob': -.2}],
            [{'token': '1', 'logprob': float('nan')}, {'token': '2', 'logprob': -.2}],
            [{'token': '1', 'logprob': -9999}, {'token': '2', 'logprob': -.2}],
            [{'token': '1'}, {'token': '2', 'logprob': -.2}],
            [{'token': '1', 'logprob': -.1}, {'token': '1', 'logprob': -.1}],
            [{'token': None, 'logprob': -.1}], {}, [None]]
        for report in bad_reports:
            with self.assertRaises(DistributionError):
                candidate_probs(report, {'a': '1', 'b': '2'})
        with self.assertRaises(DistributionError):
            candidate_probs([], {'a': 'Yes', 'b': ' yes'})
        with self.assertRaises(DistributionError): candidate_probs([], {})
        self.assertEqual(confidence({'one': 1.0}), 1.0)


class ClientTests(unittest.TestCase):
    def test_import_has_no_env_reads_or_calls(self):
        code = ('from unittest.mock import patch; '
                'p=patch("builtins.open",side_effect=AssertionError("file read")); '
                'p.start(); import deepseek_flash')
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_from_env_no_dotenv_and_key_redacted(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError): DeepSeekFlashClient.from_env()
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'TEST_ONLY'}):
            client = DeepSeekFlashClient.from_env()
        self.assertNotIn('TEST_ONLY', repr(client))
        for url in ['http://api.deepseek.com', 'https://evil.example', 'https://api.deepseek.com/evil']:
            with self.assertRaises(ValueError): DeepSeekFlashClient('TEST_ONLY', endpoint=url)
        with self.assertRaises(ValueError): DeepSeekFlashClient('')

    def test_http_shape_and_no_retry(self):
        client = DeepSeekFlashClient('TEST_ONLY')
        body, _ = build_request('x', task().question)
        fake = unittest.mock.MagicMock()
        fake.__enter__.return_value.read.return_value = json.dumps(response()).encode()
        with patch('urllib.request.urlopen', return_value=fake) as post:
            self.assertEqual(client.post(body)['usage']['prompt_tokens'], 123)
        req = post.call_args.args[0]
        self.assertEqual(req.full_url, 'https://api.deepseek.com/chat/completions')
        self.assertEqual(json.loads(req.data), body)
        with patch('urllib.request.urlopen', side_effect=urllib.error.HTTPError(
                req.full_url, 401, 'SECRET_ECHO', {}, None)) as post:
            with self.assertRaisesRegex(RuntimeError, 'HTTP 401') as error: client.post(body)
        self.assertNotIn('SECRET_ECHO', str(error.exception))
        self.assertEqual(post.call_count, 1)

    def test_notebook_one_liners(self):
        client = DeepSeekFlashClient('TEST_ONLY')
        with patch.object(client, 'post', return_value=response((('Yes', .9), ('No', .1)))):
            prob, conf = client.noul('x', 'True?')
            self.assertAlmostEqual(prob, .9)
            self.assertEqual(conf, .8)
        with patch.object(client, 'post', return_value=response()):
            p, conf = client.choice('x', 'Pick?', {'a': 'A', 'b': 'B'})
            self.assertAlmostEqual(p['a'], .8)
            self.assertEqual(conf, .6)
            ev, p, conf = client.score('x', 'Rate?', ['low', 'high'])
            self.assertAlmostEqual(ev, 1.2)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.client = DeepSeekFlashClient('TEST_ONLY')
        self.adapter = JevBenchAdapter(self.client)

    def test_no_answer_leak_and_canonical_keys(self):
        with patch.object(self.client, 'post', return_value=response()) as post:
            r = self.adapter.run(task())
        self.assertTrue(r.ok)
        self.assertEqual(set(r.probs), {'ignore', 'today'})
        request = json.dumps(post.call_args.args[0])
        self.assertNotIn('GOLD_MUST_NOT_LEAK', request)
        self.assertNotIn('RATIONALE_MUST_NOT_LEAK', request)
        self.assertEqual(r.probs_source, 'native_logprobs_candidate_renormalized')
        self.assertEqual(r.usage['input_tokens'], 123)

    def test_noul_and_zero_based_score(self):
        with patch.object(self.client, 'post', return_value=response((('Yes', .9), ('No', .1)))):
            r = self.adapter.run(task('noul', {'true': 'T', 'false': 'F'}, ['no', 'yes']))
        self.assertTrue(r.ok)
        self.assertAlmostEqual(r.probs['yes'], .9)
        with patch.object(self.client, 'post', return_value=response()):
            r = self.adapter.run(task('score', ['low', 'high'], ['0', '1']))
        self.assertEqual(set(r.probs), {'0', '1'})
        self.assertAlmostEqual(r.probs['0'], .8)

    def test_parse_failure_keeps_billed_usage_and_raw(self):
        out = response((('1', .8), ('Other', .2)))
        with patch.object(self.client, 'post', return_value=out): r = self.adapter.run(task())
        self.assertFalse(r.ok)
        self.assertEqual(r.error_kind, 'distribution')
        self.assertEqual(r.usage['input_tokens'], 123)
        self.assertEqual(r.raw, out)

    def test_malformed_response_network_and_label_mismatch(self):
        for out in [{}, {'choices': []}, {'choices': [{'logprobs': None}]},
                    {'choices': [{'logprobs': {'content': [None]}}]}]:
            with patch.object(self.client, 'post', return_value=out): r = self.adapter.run(task())
            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, 'distribution')
        with patch.object(self.client, 'post', side_effect=TimeoutError('SECRET_ECHO')):
            r = self.adapter.run(task())
        self.assertEqual(r.error_kind, 'transport')
        self.assertNotIn('SECRET_ECHO', r.error)
        with patch.object(self.client, 'post') as post:
            r = self.adapter.run(task(labels=['unexpected']))
            post.assert_not_called()
        self.assertEqual(r.error_kind, 'input')


class NotebookTests(unittest.TestCase):
    def test_run_all_is_offline_and_uses_canonical_payload(self):
        from pathlib import Path
        nb = json.loads(Path('learn/deepseek-flash.ipynb').read_text())
        cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertNotIn('import minijev', '\n'.join(cells))
        scope = {}
        with patch.dict(os.environ, {}, clear=True), patch('urllib.request.urlopen',
                side_effect=AssertionError('Notebook must not call API on Run All')):
            for code in cells:
                exec(compile(code, 'deepseek-flash.ipynb', 'exec'), scope)
        self.assertEqual(scope['req_body']['max_tokens'], 1)
        self.assertEqual(scope['req_body']['thinking'], {'type': 'disabled'})
        for c in nb['cells']:
            if c['cell_type'] == 'code':
                self.assertEqual(c['outputs'], [])
                self.assertIsNone(c['execution_count'])


if __name__ == '__main__':
    unittest.main()
