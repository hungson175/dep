import os
import unittest
from unittest.mock import Mock, patch

from pathlib import Path
from fastapi.testclient import TestClient
# Legacy minijev auto-loads .env; never allow that side effect in tests.
_exists = os.path.exists
with patch('os.path.exists', side_effect=lambda p: False if Path(p).name == '.env' else _exists(p)), \
        patch.dict(os.environ, {'DEP_API_KEY': 'TEST_ONLY'}):
    import app
from deepseek_flash import APIError, DistributionError
from deepseek_site import BudgetUnavailable


class ModelRouteTests(unittest.TestCase):
    def setUp(self):
        app._hits.clear()
        self.body = {'state': 'hello', 'questions': {'route': {
            'type': 'choice', 'instructions': 'Pick?', 'criteria': {'a': 'A', 'b': 'B'}}}}
        self.site = Mock(available=True)
        self.site.budget.snapshot.return_value = {'cap_usd': 20, 'charged_usd': 0, 'remaining_usd': 20}
        self.patch = patch.object(app, '_deepseek', self.site); self.patch.start()
        self.client = TestClient(app.app); self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None); self.patch.stop()

    def test_bonsai_default_legacy_and_explicit(self):
        for model in [None, 'bonsai']:
            body = dict(self.body)
            if model: body['model'] = model
            with patch('minijev.choice', return_value=({'a': .8, 'b': .2}, .6)), \
                 patch('minijev.model_seconds', return_value=.1):
                response = self.client.post('/api/decide', json=body)
            self.assertEqual(response.status_code, 200)
            self.assertIn('bonsai', response.json()['model'])
            self.assertEqual(response.json()['cost_usd'], 0)
            self.assertEqual(response.json()['answers']['route']['choice'], 'a')
        self.site.run.assert_not_called()

    def test_deepseek_library_route_and_reported_cost(self):
        answer = {'type': 'choice', 'choice': 'a', 'confidence': 1.,
                  'probabilities': {'a': 1., 'b': 0.}, 'diagnostics': {'missing_options': ['b']}}
        self.site.run.return_value = (answer, {'input_tokens': 100, 'output_tokens': 1}, .0000312)
        with patch('minijev.choice', side_effect=AssertionError('wrong model')):
            response = self.client.post('/api/decide', json={**self.body, 'model': 'deepseek-flash'})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['model'], 'deepseek-flash')
        self.assertEqual(result['answers']['route']['probabilities']['b'], 0)
        self.assertAlmostEqual(result['cost_usd'], .0000312)
        self.assertEqual(result['usage']['input_tokens'], 100)
        self.assertEqual(self.site.run.call_count, 1)

    def test_validation_before_any_paid_call(self):
        bad = {'type': 'choice', 'instructions': 'Pick?', 'criteria': {'a': 'only one'}}
        response = self.client.post('/api/decide', json={**self.body, 'model': 'deepseek-flash',
            'questions': {**self.body['questions'], 'invalid': bad}})
        self.assertEqual(response.status_code, 400)
        self.site.run.assert_not_called()
        self.assertEqual(self.client.post('/api/decide', json={**self.body, 'model': 'unknown'}).status_code, 422)

    def test_unavailable_budget_and_provider_errors(self):
        self.site.available = False
        self.assertEqual(self.client.post('/api/decide', json={**self.body, 'model': 'deepseek-flash'}).status_code, 503)
        self.site.run.assert_not_called()
        self.site.available = True
        for error, status in [(BudgetUnavailable('cap'), 503), (APIError(429), 502),
                              (DistributionError('bad report'), 502)]:
            self.site.run.side_effect = error
            self.assertEqual(self.client.post('/api/decide', json={**self.body, 'model': 'deepseek-flash'}).status_code, status)

    def test_stats_and_stress_do_not_enable_paid_burst(self):
        stats = self.client.get('/api/stats').json()
        self.assertEqual(stats['default_model'], 'bonsai')
        self.assertTrue(stats['models']['deepseek-flash']['available'])
        self.assertEqual(stats['models']['deepseek-flash']['budget']['cap_usd'], 20)
        response = self.client.post('/api/stress', json={**self.body, 'model': 'deepseek-flash', 'password': 'test'})
        self.assertEqual(response.status_code, 400)
        self.site.run.assert_not_called()

    def test_bonsai_noul_score_and_deepseek_score_shapes(self):
        questions = {'yesno': {'type': 'noul', 'instructions': 'Allowed?',
                     'criteria': {'true': 'allowed', 'false': 'blocked'}},
                     'score': {'type': 'score', 'instructions': 'Rate?', 'criteria': ['low', 'high']}}
        with patch('minijev.noul', return_value=(.8, .6)), \
                patch('minijev.score', return_value=(1.7, {'1': .3, '2': .7}, .4)):
            response = self.client.post('/api/decide', json={'state': 'x', 'questions': questions})
        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(response.json()['answers']['score']['score'], 1.7)
        self.assertAlmostEqual(response.json()['answers']['yesno']['noul'], .8)
        self.site.run.return_value = ({'type': 'score', 'score': 2., 'confidence': 1.,
            'probabilities': {'1': 0., '2': 1.}}, {'input_tokens': None, 'output_tokens': None}, None)
        response = self.client.post('/api/decide', json={'model': 'deepseek-flash', 'state': 'x',
            'questions': {'score': questions['score']}})
        self.assertEqual(response.status_code, 200)
        answer = response.json()['answers']['score']
        self.assertEqual(answer['distribution'], {'1': 0., '2': 1.})
        self.assertEqual(answer['levels'], 2)
        self.assertIsNone(response.json()['cost_usd'])

    def test_body_limits_empty_questions_and_validation_boundaries(self):
        self.assertEqual(self.client.post('/api/decide', json={**self.body, 'questions': {}}).status_code, 400)
        self.assertEqual(self.client.post('/api/decide', json={**self.body,
            'questions': {str(i): self.body['questions']['route'] for i in range(5)}}).status_code, 400)
        for question in [{'type': 'noul', 'instructions': '   '},
                         {'type': 'noul', 'instructions': 'x', 'criteria': {'true': 'x'}},
                         {'type': 'score', 'instructions': 'x', 'criteria': ['one']},
                         {'type': 'score', 'instructions': 'x', 'criteria': ['x' * 201, 'high']}]:
            self.assertEqual(self.client.post('/api/decide', json={'state': 'x',
                             'questions': {'q': question}}).status_code, 400)
        response = self.client.post('/api/decide', content=b'{}', headers={'content-length': '300000'})
        self.assertEqual(response.status_code, 413)

    def test_queue_and_rate_limits_preserve_bonsai(self):
        with patch.dict(app.stats, {'waiting': app.MAX_WAITING}):
            self.assertEqual(self.client.post('/api/decide', json=self.body).status_code, 503)
        with patch.dict(app._hits, {'testclient': __import__('collections').deque([__import__('time').time()] * 6)}):
            self.assertEqual(self.client.post('/api/decide', json=self.body).status_code, 429)

    def test_bonsai_stress_gates_and_success_without_inference(self):
        import hashlib
        body = {**self.body, 'password': 'TEST_ONLY'}
        with patch.object(app, 'STRESS_HASH', ''):
            self.assertEqual(self.client.post('/api/stress', json=body).status_code, 404)
        with patch.object(app, 'STRESS_HASH', hashlib.sha256(b'TEST_ONLY').hexdigest()), \
                patch.dict(app._stress_last, {'t': 0}), \
                patch('minijev.choice', return_value=({'a': .8, 'b': .2}, .6)):
            self.assertEqual(self.client.post('/api/stress', json={**body, 'password': 'wrong'}).status_code, 401)
            response = self.client.post('/api/stress', json=body)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['ok'], 20)
            self.assertEqual(self.client.post('/api/stress', json=body).status_code, 429)

    def test_health_budget_failure_and_generic_provider_error(self):
        with patch('minijev.token_id', return_value=123):
            self.assertTrue(self.client.get('/api/health').json()['ok'])
        with patch('minijev.token_id', side_effect=RuntimeError('private')):
            self.assertEqual(self.client.get('/api/health').status_code, 503)
        self.site.budget.snapshot.side_effect = BudgetUnavailable('corrupt')
        self.assertTrue(self.client.get('/api/stats').json()['models']['deepseek-flash']['budget']['unavailable'])
        for error, status in [(TimeoutError('timeout'), 504), (Exception('PRIVATE_ERROR'), 502)]:
            self.site.run.side_effect = error
            response = self.client.post('/api/decide', json={**self.body, 'model': 'deepseek-flash'})
            self.assertEqual(response.status_code, status)
            self.assertNotIn('PRIVATE_ERROR', response.text)


if __name__ == '__main__': unittest.main()
