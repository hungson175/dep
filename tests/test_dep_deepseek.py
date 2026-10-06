import io
import json
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from dep_deepseek import DeepSeek, APIError, DistributionError, TransportError


def response(pairs, input_tokens=20, output_tokens=1):
    return {'model': 'deepseek-flash', 'usage': {
        'prompt_tokens': input_tokens, 'completion_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens},
        'choices': [{'logprobs': {'content': [{'top_logprobs': [
            {'token': token, 'logprob': math.log(prob)} for token, prob in pairs
        ]}]}}]}


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.dep = DeepSeek(api_key='TEST_ONLY')

    def test_choice_hides_transport_and_fills_missing_zero(self):
        with patch('deepseek_flash.DeepSeekFlashClient.post', return_value=response([('2', .9), ('Other', .1)])) as post:
            out = self.dep.choice('Xin chào', 'Route?', {'sales': 'buy', 'support': 'help', 'other': 'unknown'})
        self.assertEqual(out['type'], 'choice')
        self.assertEqual(out['choice'], 'support')
        self.assertEqual(out['probabilities'], {'sales': 0.0, 'support': 1.0, 'other': 0.0})
        self.assertEqual(out['diagnostics']['missing_options'], ['sales', 'other'])
        self.assertEqual(post.call_count, 1)

    def test_predict_all_types_and_aggregate_usage(self):
        qs = {
            'urgent': {'type': 'noul', 'instructions': 'Urgent?'},
            'route': {'type': 'choice', 'instructions': 'Route?', 'criteria': ['sales', 'support']},
            'mood': {'type': 'score', 'instructions': 'Mood?', 'criteria': ['calm', 'angry']},
        }
        with patch('deepseek_flash.DeepSeekFlashClient.post', side_effect=[
                response([('Yes', .8), ('No', .2)]), response([('1', .2), ('2', .8)]),
                response([('1', .25), ('2', .75)])]):
            out = self.dep.predict({'message': 'Help!'}, qs)
        self.assertEqual(set(out['answers']), set(qs))
        self.assertAlmostEqual(out['answers']['urgent']['noul'], .8)
        self.assertEqual(out['answers']['route']['choice'], 'support')
        mood = out['answers']['mood']
        self.assertAlmostEqual(mood['score'], 1.75)
        self.assertEqual(mood['legend'], {'1': 'calm', '2': 'angry'})
        self.assertAlmostEqual(mood['probabilities']['1'], .25)
        self.assertAlmostEqual(mood['probabilities']['2'], .75)
        self.assertEqual(out['usage'], {'input_tokens': 60, 'output_tokens': 3, 'total_tokens': 63})
        self.assertEqual(out['model'], 'deepseek-flash')

    def test_convenience_noul_score(self):
        with patch('deepseek_flash.DeepSeekFlashClient.post', return_value=response([('No', .9)])):
            out = self.dep.noul('x', 'Allowed?', {'true': 'allowed', 'false': 'blocked'})
        self.assertEqual(out['noul'], 0.0)
        self.assertEqual(out['probabilities'], {'true': 0.0, 'false': 1.0})
        with patch('deepseek_flash.DeepSeekFlashClient.post', return_value=response([('2', .9)])):
            out = self.dep.score('x', 'Mood?', ['calm', 'angry'])
        self.assertEqual(out['score'], 2.0)
        self.assertEqual(out['probabilities']['1'], 0.0)

    def test_validate_whole_request_before_billing(self):
        bad = [None, [], {}, {'': {'type': 'noul', 'instructions': 'x'}},
               {'q': None}, {'q': {'type': 'score', 'instructions': 'Rate?', 'criteria': []}},
               {'first': {'type': 'noul', 'instructions': 'x'}, 'bad': {'type': 'nope'}}]
        for qs in bad:
            with patch('deepseek_flash.DeepSeekFlashClient.post') as post:
                with self.assertRaises(ValueError): self.dep.predict('x', qs)
                post.assert_not_called()
        with patch('deepseek_flash.DeepSeekFlashClient.post') as post:
            with self.assertRaises(ValueError): self.dep.predict({'x': float('nan')},
                {'q': {'type': 'noul', 'instructions': 'x'}})
            post.assert_not_called()

    def test_env_constructor_no_secret_in_repr(self):
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'TEST_ENV'}):
            dep = DeepSeek()
        self.assertNotIn('TEST_ENV', repr(dep))
        self.assertNotIn('TEST_ONLY', repr(self.dep))
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError): DeepSeek()
        with self.assertRaises(ValueError): DeepSeek(api_key='TEST_ONLY', missing_policy='bad')
        for timeout in [0, -1, float('nan'), True]:
            with self.assertRaises(ValueError): DeepSeek(api_key='TEST_ONLY', timeout=timeout)

    def test_strict_and_all_candidates_missing_errors(self):
        out = response([('1', .9)])
        with patch('deepseek_flash.DeepSeekFlashClient.post', return_value=out):
            with self.assertRaises(DistributionError):
                DeepSeek(api_key='TEST_ONLY', missing_policy='error').choice('x', 'Pick?', ['a', 'b'])
        with patch('deepseek_flash.DeepSeekFlashClient.post', return_value=response([('Other', .9)])):
            with self.assertRaises(DistributionError): self.dep.noul('x', 'True?')

    def test_unknown_usage_and_malformed_response(self):
        out = response([('Yes', .9)])
        out.pop('usage')
        with patch('deepseek_flash.DeepSeekFlashClient.post', return_value=out):
            result = self.dep.predict('x', {'q': {'type': 'noul', 'instructions': 'x'}})
        self.assertEqual(result['usage'], {'input_tokens': None, 'output_tokens': None, 'total_tokens': None})
        out['usage'] = 'malformed'
        with patch('deepseek_flash.DeepSeekFlashClient.post', return_value=out):
            with self.assertRaises(DistributionError): self.dep.noul('x', 'True?')
        for malformed in [False, 0, [], '']:
            out['usage'] = malformed
            with patch('deepseek_flash.DeepSeekFlashClient.post', return_value=out):
                with self.assertRaises(DistributionError): self.dep.noul('x', 'True?')

    def test_packaged_style_http_seam_without_live_api(self):
        out = response([('1', .7), ('2', .3)])
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(out).encode())) as http:
            answer = self.dep.choice('hello', 'Pick?', ['a', 'b'])
        self.assertEqual(answer['choice'], 'a')
        request = http.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertEqual(body['max_tokens'], 1)
        self.assertNotIn('TEST_ONLY', json.dumps(body))
        with patch('deepseek_flash.DeepSeekFlashClient.post', side_effect=APIError(429)) as post:
            with self.assertRaises(APIError): self.dep.noul('x', 'True?')
        self.assertEqual(post.call_count, 1)

    def test_keyword_convenience_and_transport_errors(self):
        with patch('deepseek_flash.DeepSeekFlashClient.post', return_value=response([('1', .9)])):
            self.assertEqual(self.dep.choice('x', 'Pick?', options=['a', 'b'])['choice'], 'a')
            self.assertEqual(self.dep.score('x', 'Rate?', levels=['low', 'high'])['score'], 1.0)
        with patch('urllib.request.urlopen', side_effect=TimeoutError('PRIVATE_ERROR_ECHO')):
            with self.assertRaises(TransportError) as error: self.dep.noul('x', 'True?')
        self.assertNotIn('PRIVATE_ERROR_ECHO', str(error.exception))
        with patch('urllib.request.urlopen', return_value=io.BytesIO(b'not-json')):
            with self.assertRaises(DistributionError): self.dep.noul('x', 'True?')
        with patch('urllib.request.urlopen', return_value=io.BytesIO(b'\xff')):
            with self.assertRaises(DistributionError): self.dep.noul('x', 'True?')

    def test_authorization_is_not_forwarded_on_redirect(self):
        import urllib.request
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response([('Yes', .9)])).encode())) as http:
            self.dep.noul('x', 'True?')
        request = http.call_args.args[0]
        self.assertEqual(request.get_header('Authorization'), 'Bearer TEST_ONLY')
        redirected = urllib.request.HTTPRedirectHandler().redirect_request(
            request, None, 302, 'Found', {}, 'https://untrusted.invalid/')
        self.assertIsNone(redirected.get_header('Authorization'))

    def test_recorded_missing_option_response_replay(self):
        fixture = json.loads(Path('tests/fixtures/deepseek_missing_options.json').read_text())
        with patch('deepseek_flash.DeepSeekFlashClient.post', return_value=fixture['response']):
            answer = self.dep.choice('Offline replay', 'Pick?', fixture['options'])
        self.assertEqual(answer['diagnostics']['missing_options'], fixture['missing_options'])
        for name in fixture['missing_options']:
            self.assertEqual(answer['probabilities'][name], 0.0)
        self.assertAlmostEqual(sum(answer['probabilities'].values()), 1.0)


if __name__ == '__main__':
    unittest.main()
