import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from benchmarks.jevbench_flash import load_public_tasks

class OriginalJevTests(unittest.TestCase):
 def test_selective_credentials(self):
  from benchmarks.jevbench_jev import read_key
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'env';p.write_text('UNRELATED=secret\nexport TYPESAFE_API_KEY="TEST_ONLY" # comment\n')
   self.assertEqual(read_key(p),'TEST_ONLY')
   for text in ['X=y','TYPESAFE_API_KEY=','TYPESAFE_API_KEY=a\nTYPESAFE_API_KEY=b']:
    p.write_text(text)
    with self.assertRaises(ValueError):read_key(p)
 def test_original_three_type_mapping_without_gold(self):
  from benchmarks.jevbench_jev import adapter_class
  cls=adapter_class();adapter=cls(model='jev-1.13.0',key_env='MOCK_JEV_KEY')
  for kind in ['choice','noul','score']:
   t=next(t for t in load_public_tasks() if t.question['type']==kind)
   request=adapter.build_request(t);old=request.copy();t.expected='HIDDEN_GOLD'
   self.assertEqual(adapter.build_request(t),old)
   self.assertNotIn('HIDDEN_GOLD',json.dumps(request))
   if kind=='choice':ans={'type':kind,'choice':t.labels[0],'probabilities':{k:1/len(t.labels) for k in t.labels}}
   elif kind=='noul':ans={'type':kind,'noul':.8}
   else:ans={'type':kind,'probabilities':{k:1/len(t.labels) for k in t.labels}}
   with patch.dict('os.environ',{'MOCK_JEV_KEY':'TEST_ONLY'}),patch('jevbench.adapters.typesafe.http_post_json',return_value=(200,{'model':'jev-1.13.0','answers':{'decision':ans},'usage':{'input_tokens':100,'output_tokens':0}},.2)) as http:
    result=adapter.run(t)
   self.assertTrue(result.ok);self.assertEqual(set(result.probs),set(t.labels));self.assertEqual(http.call_count,1)
 def test_offline_run_uses_existing_scoring_and_separate_budget(self):
  from benchmarks.jevbench_jev import run,adapter_class
  adapter_class()
  from jevbench.adapters.base import DecisionResult
  tasks=load_public_tasks()[:3]
  class Fake:
   name='typesafe';price_input_per_m=.042;price_output_per_m=0
   def reserve_estimate(self,t):return .0042
   def run(self,t):return DecisionResult('typesafe',True,probs={k:float(k==str(t.expected)) for k in t.labels},model='jev-1.13.0',probs_source='native',raw={'model':'jev-1.13.0'},usage={'input_tokens':100,'output_tokens':0})
  with tempfile.TemporaryDirectory() as tmp:
   summary=run(tasks,Fake(),Path(tmp)/'run')
   self.assertEqual(summary['n_correct'],3);self.assertEqual(summary['n_attempted'],3)
   self.assertIsNone(summary['official_score']);self.assertEqual(summary['per_type']['choice']['n_correct'],3)
   manifest=json.loads((Path(tmp)/'run/manifest.json').read_text());self.assertTrue(manifest['no_retries'])
   self.assertNotIn('TEST_ONLY',json.dumps(manifest))
if __name__=='__main__':unittest.main()
