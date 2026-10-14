import io
import json
import math
import string
import unittest
from unittest.mock import patch
from deepseek_flash import build_request, marks, candidate_probs, missing_options, DeepSeekFlashClient, DistributionError
from dep_deepseek import DeepSeek

class Letters52Tests(unittest.TestCase):
    def test_distinct_boundaries_and_no_silent_legacy_change(self):
        self.assertEqual(marks(52, marker_scheme='letters52'), list(string.ascii_lowercase + string.ascii_uppercase))
        self.assertEqual(marks(49, marker_scheme='letters52')[-1], 'W')
        self.assertEqual(marks(2), ['1', '2'])
        for n in [0, 53, True, 1.1]:
            with self.assertRaises(ValueError): marks(n, marker_scheme='letters52')
        with self.assertRaises(ValueError): DeepSeekFlashClient('TEST', marker_scheme='bad')

    def test_cache_prefix_question_last_case_explicit(self):
        options = {f'category_{i}': f'Category {i}' for i in range(49)}
        q = {'type':'choice', 'instructions':'TARGET_QUERY_ONE', 'criteria':options}
        body, labels = build_request('STATIC_CARDS', q, marker_scheme='letters52')
        text = body['messages'][0]['content']
        self.assertEqual(labels['category_26'], 'A')
        self.assertTrue(text.endswith('Question: TARGET_QUERY_ONE'))
        self.assertLess(text.index('STATIC_CARDS'), text.index('Options:'))
        self.assertIn('case-sensitive', text)
        self.assertIn('a and A are different', text)
        q['instructions']='TARGET_QUERY_TWO'
        text2=build_request('STATIC_CARDS', q, marker_scheme='letters52')[0]['messages'][0]['content']
        self.assertEqual(text.split('Question: ')[0], text2.split('Question: ')[0])
        self.assertEqual((body['max_tokens'], body['top_logprobs']), (1,20))

    def test_no_case_merge_no_whitespace_merge_and_missing_detection(self):
        labels={'lower':'a','upper':'A','last':'Z'}
        report=[{'token':t,'logprob':math.log(p)} for t,p in [('a',.1),('A',.6),(' a',.2),(' Z',.1)]]
        p,mass=candidate_probs(report,labels)
        self.assertAlmostEqual(p['lower'],1/7)
        self.assertAlmostEqual(p['upper'],6/7)
        self.assertEqual(p['last'],0)
        self.assertAlmostEqual(mass,.7)
        out={'choices':[{'logprobs':{'content':[{'top_logprobs':report}]}}]}
        self.assertEqual(missing_options(out,labels),['last'])
        with self.assertRaises(DistributionError):candidate_probs(report,labels,missing_policy='error')

    def test_library_transport_49_and_usage_cache_diagnostics(self):
        report=[{'token':t,'logprob':math.log(.045)} for t in string.ascii_lowercase[:19]]
        report.append({'token':'A','logprob':math.log(.1)})
        out={'model':'deepseek-flash','usage':{'prompt_tokens':100,'completion_tokens':1,'total_tokens':101,
             'prompt_cache_hit_tokens':80,'prompt_cache_miss_tokens':20},
             'choices':[{'logprobs':{'content':[{'top_logprobs':report}]}}]}
        with patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(out).encode())) as http:
            result=DeepSeek(api_key='TEST',marker_scheme='letters52').predict('PUBLIC_CARDS',{
              'route':{'type':'choice','instructions':'QUERY_LAST','criteria':[f'c{i}' for i in range(49)]}})
        answer=result['answers']['route']
        self.assertEqual(answer['choice'],'c26')
        self.assertEqual(len(answer['diagnostics']['missing_options']),29)
        self.assertEqual(answer['diagnostics']['reported_options_count'],20)
        self.assertEqual(answer['diagnostics']['reported_ranking'][0],'c26')
        self.assertEqual(len(answer['diagnostics']['reported_logprobs']),20)
        self.assertTrue(answer['diagnostics']['top10_supported'])
        self.assertEqual(result['usage']['prompt_cache_hit_tokens'],80)
        self.assertEqual(result['usage']['prompt_cache_miss_tokens'],20)
        self.assertEqual(http.call_count,1)
        self.assertTrue(json.loads(http.call_args.args[0].data)['messages'][0]['content'].endswith('Question: QUERY_LAST'))

    def test_short_report_not_top10_complete(self):
        out={'usage':{},'choices':[{'logprobs':{'content':[{'top_logprobs':[{'token':'a','logprob':-.1}]}]}}]}
        with patch('deepseek_flash.DeepSeekFlashClient.post',return_value=out):
            result=DeepSeek(api_key='TEST',marker_scheme='letters52').choice('x','Pick',['c'+str(i) for i in range(49)])
        self.assertFalse(result['diagnostics']['top10_supported'])
        self.assertEqual(result['diagnostics']['reported_options_count'],1)

class Letters52EvidenceTests(unittest.TestCase):
    def test_published_tokenizer_evidence_distinct_bare_tokens(self):
        from pathlib import Path
        evidence=json.loads(Path('tests/fixtures/deepseek_v4_letters52_markers.json').read_text())
        rows=evidence['markers']
        self.assertEqual([r['marker'] for r in rows],list(string.ascii_lowercase+string.ascii_uppercase))
        self.assertEqual(len({r['token_id'] for r in rows}),52)
        self.assertTrue(all(r['marker']==r['token'] for r in rows))
        self.assertTrue(all(r['space_prefix_ids'] != [r['token_id']] for r in rows))
        self.assertEqual(len(evidence['tokenizer_json_sha256']),64)

    def test_public_live_vcr_replay_case_and_cache_fields(self):
        import vcr
        options={f'category_{i}':f'Owns questions explicitly naming category_{i}.' for i in range(49)}
        cards='\n'.join(f'Public synthetic card category_{i}: owns questions explicitly naming category_{i}. No other ownership.' for i in range(49))
        with vcr.use_cassette('tests/fixtures/deepseek_letters52_public.yaml',record_mode='none',filter_headers=['authorization']):
            dep=DeepSeek(api_key='TEST_ONLY',marker_scheme='letters52')
            for target in ['category_0','category_26']:
                result=dep.predict(cards,{'route':{'type':'choice','criteria':options,
                    'instructions':f'Select the category explicitly named here: {target}.'}})
                self.assertEqual(result['answers']['route']['choice'],target)
                self.assertEqual(len(result['answers']['route']['probabilities']),49)
                self.assertIsInstance(result['usage']['prompt_cache_hit_tokens'],int)
                self.assertIsInstance(result['usage']['prompt_cache_miss_tokens'],int)

    def test_new_mode_score_env_adapter_and_unknown_cache(self):
        from deepseek_flash import JevBenchAdapter
        from types import SimpleNamespace
        with patch.dict('os.environ',{'DEEPSEEK_API_KEY':'TEST_ONLY'}):
            client=DeepSeekFlashClient.from_env(marker_scheme='letters52')
        task=SimpleNamespace(state='Public state',question={'type':'score','instructions':'Rate?','criteria':['low','high']},labels=['0','1'])
        out={'model':'deepseek-flash','usage':{},'choices':[{'logprobs':{'content':[{
             'top_logprobs':[{'token':'a','logprob':math.log(.2)},{'token':'b','logprob':math.log(.8)}]}]}}]}
        with patch.object(client,'post',return_value=out):
            result=JevBenchAdapter(client).run(task)
        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.probs['1'],.8)
        with patch('deepseek_flash.DeepSeekFlashClient.post',return_value=out):
            result=DeepSeek(api_key='TEST_ONLY',marker_scheme='letters52').predict('x',{'r':task.question})
        self.assertIsNone(result['usage']['prompt_cache_hit_tokens'])
        self.assertIsNone(result['usage']['prompt_cache_miss_tokens'])
