"""Original Jev adapter with unchanged pinned public tasks/scorer; no import I/O."""
import hashlib,json,os,re,shlex,sys,types
from pathlib import Path
from benchmarks.jevbench_flash import PIN,SOURCE,SCOPE,blob_sha,activate_source,write_json
MODEL='jev-1.13.0'
PRICE_INPUT=.042

def read_key(path):
 values=[]
 for line in Path(path).read_text().splitlines():
  m=re.match(r'^\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)$',line)
  if m:
   parts=shlex.split(m[1],comments=True)
   if len(parts)!=1:raise ValueError('Missing or malformed TYPESAFE_API_KEY')
   values.append(parts[0])
 if len(values)!=1 or not values[0]:raise ValueError('Expected exactly one nonempty TYPESAFE_API_KEY')
 return values[0]

def adapter_class():
 activate_source()
 manifest=json.loads((SOURCE/'typesafe_source_manifest.json').read_text())
 expected={'jevbench/adapters/base.py','jevbench/adapters/typesafe.py','jevbench/runner.py'}
 if set(manifest)!=expected:raise ValueError('Original adapter manifest mismatch')
 for name,sha in manifest.items():
  if blob_sha((SOURCE/name).read_bytes())!=sha:raise ValueError('Original adapter source changed')
 # Namespace only: avoid importing unrelated optional adapters/dependencies.
 if 'jevbench.adapters' not in sys.modules:
  package=types.ModuleType('jevbench.adapters');package.__path__=[str(SOURCE/'jevbench/adapters')]
  sys.modules['jevbench.adapters']=package
  import jevbench
  jevbench.adapters=package
 from jevbench.adapters.typesafe import TypeSafeAdapter
 return TypeSafeAdapter

def run(tasks,adapter,output):
 adapter_class()
 from jevbench.runner import Runner
 from jevbench.budget import Ledger
 from jevbench.tasks import dataset_hash
 from jevbench.summarize import summarize
 output=Path(output);output.mkdir(parents=True,exist_ok=False,mode=0o700)
 write_json(output/'manifest.json',{'scope':SCOPE,'upstream_pin':PIN,'dataset_hash':dataset_hash(tasks),
  'planned':len(tasks),'requested_model':MODEL,'endpoint':'https://api.typesafe.ai/v1/systemone',
  'adapter':'original_upstream_typesafe','no_retries':True,'cap_usd':20,
  'price_usd_per_million':{'input':PRICE_INPUT,'output':0},'price_checked':'2026-10-06',
  'pricing_source':'https://docs.typesafe.ai/models','latency_basis':'provider_api_round_trip_wall',
  'source_git_blobs':json.loads((SOURCE/'typesafe_source_manifest.json').read_text()),
  'orchestration_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 ledger=Ledger(output/'ledger.jsonl',cap_usd=20)
 runner=Runner(adapter,ledger,raw_dir=output/'raw',default_reserve_usd=.0042)
 records=runner.run_all(tasks,results_path=output/'records.jsonl',delay_s=.05)
 summary=summarize(tasks,records,ledger.charged)
 summary.update(scope=SCOPE,upstream_pin=PIN,official_score=None,official_rank=None,
  latency_basis='provider_api_round_trip_wall',no_retries=True,
  price_note='Usage times published input tariff; not an invoice. Failed/unknown calls retain reservations.',
  records_sha256=hashlib.sha256((output/'records.jsonl').read_bytes()).hexdigest())
 summary['per_type']={}
 # Upstream records do not include question_type; use canonical task IDs.
 for kind in sorted({t.question['type'] for t in tasks}):
  typed=[t for t in tasks if t.question['type']==kind];ids={t.id for t in typed}
  summary['per_type'][kind]=summarize(typed,[r for r in records if r['task_id'] in ids])
 write_json(output/'summary.json',summary)
 return summary
