import concurrent.futures
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deepseek_site import SiteBudget, BudgetUnavailable, DeepSeekSite, RESERVE_USD


class SiteBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'ledger.jsonl'
        self.budget = SiteBudget(self.path, .6)

    def tearDown(self): self.tmp.cleanup()

    def test_reservation_settlement_and_restart(self):
        rid = self.budget.reserve(.3)
        self.assertAlmostEqual(SiteBudget(self.path, .6).snapshot()['charged_usd'], .3)
        self.budget.settle(rid, .01)
        self.assertAlmostEqual(self.budget.snapshot()['charged_usd'], .01)
        with self.assertRaises(BudgetUnavailable): self.budget.settle(rid, .01)
        with self.assertRaises(BudgetUnavailable): self.budget.settle('missing', 0)
        self.assertNotIn('api_key', self.path.read_text())

    def test_concurrent_calls_cannot_over_reserve(self):
        def reserve(_):
            try: return SiteBudget(self.path, .6).reserve(.3)
            except BudgetUnavailable: return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(reserve, range(8)))
        self.assertEqual(sum(i is not None for i in ids), 2)
        self.assertAlmostEqual(self.budget.snapshot()['charged_usd'], .6)
        with self.assertRaises(BudgetUnavailable): SiteBudget(self.path, .1).snapshot()

    def test_bad_inputs_and_corrupt_evidence_fail_closed(self):
        for cap in [0, 21, float('nan'), True]:
            with self.assertRaises(ValueError): SiteBudget(self.path, cap)
        for cost in [0, -1, float('inf'), True]:
            with self.assertRaises(ValueError): self.budget.reserve(cost)
        rid = self.budget.reserve(.3)
        for cost in [.4, -1, float('nan'), True]:
            with self.assertRaises((ValueError, BudgetUnavailable)): self.budget.settle(rid, cost)
        self.path.write_text('corrupt\n')
        with self.assertRaises(BudgetUnavailable): self.budget.reserve(.1)
        self.assertEqual(self.path.read_text(), 'corrupt\n')


class DeepSeekSiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.budget = SiteBudget(Path(self.tmp.name) / 'ledger.jsonl', 20)
        self.site = DeepSeekSite(self.budget, key='TEST_ONLY')
        self.question = {'type': 'choice', 'instructions': 'Pick?', 'criteria': ['a', 'b']}

    def tearDown(self): self.tmp.cleanup()

    def test_library_wrapper_bills_known_usage(self):
        result = {'answers': {'decision': {'type': 'choice', 'choice': 'a',
                   'probabilities': {'a': 1., 'b': 0.}, 'confidence': 1.,
                   'diagnostics': {'missing_options': ['b']}}},
                  'usage': {'input_tokens': 100, 'output_tokens': 1, 'total_tokens': 101}}
        with patch('deepseek_site.DeepSeek.predict', return_value=result) as predict:
            answer, usage, cost = self.site.run('hello', self.question)
        self.assertEqual(predict.call_count, 1)
        self.assertEqual(answer['probabilities']['b'], 0)
        self.assertAlmostEqual(cost, .0000312)
        self.assertAlmostEqual(self.budget.snapshot()['charged_usd'], cost)
        self.assertNotIn('TEST_ONLY', repr(self.site))

    def test_unknown_usage_or_failed_call_retains_reserve(self):
        out = {'answers': {'decision': {'type': 'noul', 'noul': 1}}, 'usage': {'input_tokens': None}}
        with patch('deepseek_site.DeepSeek.predict', return_value=out):
            self.assertIsNone(self.site.run('x', self.question)[2])
        with patch('deepseek_site.DeepSeek.predict', side_effect=TimeoutError('private')):
            with self.assertRaises(TimeoutError): self.site.run('x', self.question)
        self.assertAlmostEqual(self.budget.snapshot()['charged_usd'], 2 * RESERVE_USD)

    def test_unavailable_invalid_and_exhausted_do_not_call_model(self):
        with patch('deepseek_site.DeepSeek.predict') as predict:
            with self.assertRaises(RuntimeError): DeepSeekSite(self.budget, key='').run('x', self.question)
            with self.assertRaises(ValueError): self.site.run('x', {'type': 'bad'})
            exhausted = DeepSeekSite(SiteBudget(Path(self.tmp.name) / 'small.jsonl', .01), key='TEST_ONLY')
            with self.assertRaises(BudgetUnavailable): exhausted.run('x', self.question)
            predict.assert_not_called()
        self.assertEqual(self.budget.snapshot()['charged_usd'], 0)


if __name__ == '__main__': unittest.main()
