import os
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
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


if __name__ == '__main__': unittest.main()
