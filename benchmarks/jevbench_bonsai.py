"""Serial Bonsai-Dep diagnostic on pinned legacy public JevBench tasks.

The existing minijev recipe is executed unchanged except its eager dotenv call
is removed from the in-memory AST. No provider retry, text fallback or repair.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import time
import types
import urllib.request

from benchmarks.jevbench_flash import PIN, ROOT, SCOPE, load_public_tasks, write_json

MODEL = 'bonsai-2-27b'
URL = 'http://127.0.0.1:17095'
KEY_FILE = '/home/hungson175/.claude/.secrets/bonsai-apikey.txt'


def load_recipe(*, live=False, source=ROOT / 'minijev.py'):
    """Load the original recipe without reading dotenv or unrelated credentials.

    Only a live run reads the specifically approved local llama-server key file.
    The controlled environment is restored immediately after module execution.
    """
    source = Path(source)
    tree = ast.parse(source.read_text(), filename=str(source))
    calls = [n for n in tree.body if isinstance(n, ast.Expr)
             and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
             and n.value.func.id == '_load_dotenv']
    if len(calls) != 1 or calls[0].value.args or calls[0].value.keywords:
        raise ValueError('expected exactly one simple top-level dotenv call')
    tree.body.remove(calls[0])
    recipe = types.ModuleType('bonsai_minijev_runtime')
    recipe.__file__ = str(source)
    controlled = {'DEP_LLAMA_URL': URL, 'DEP_API_KEY': '',
                  'DEP_API_KEY_FILE': KEY_FILE if live else ''}
    previous = {k: os.environ.get(k) for k in controlled}
    try:
        os.environ.update(controlled)
        exec(compile(tree, str(source), 'exec'), recipe.__dict__)
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
    if live and not recipe.KEY:
        raise ValueError('authorized Bonsai key is unavailable')
    return recipe


def verify_model(recipe):
    """GET metadata before inference; return only non-secret identity fields."""
    headers = {'Authorization': f'Bearer {recipe.KEY}'} if recipe.KEY else {}
    request = urllib.request.Request(recipe.URL + '/v1/models', headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        metadata = json.load(response)
    aliases = [item['id'] for item in metadata['data']]
    if MODEL not in aliases:
        raise ValueError('Bonsai server identity mismatch')
    return {'aliases': aliases, 'verified_alias': MODEL}


@dataclass
class Result:
    ok: bool = False
    probs: dict | None = None
    probs_source: str = 'native_post_sampling_probs_candidate_renormalized'
    model: str = MODEL
    status: int | None = None
    error: str | None = None
    error_kind: str | None = None
    latency_s: float = 0.0
    model_seconds: float = 0.0
    usage: dict = field(default_factory=dict)
    exchanges: list = field(default_factory=list)
    candidate_mass: float | None = None
    missing_options: list = field(default_factory=list)


class BonsaiAdapter:
    """Call the existing prompt/bias/one-token recipe with canonical renaming."""
    name = 'bonsai_dep_minijev'

    def __init__(self, recipe):
        self.recipe = recipe

    def prepare(self, task):
        # Capture the original primitives' exact prompt without any tokenization.
        # Only state/question/labels are inspected, never expected or provenance.
        q = task.question
        kind, instructions = q['type'], q.get('instructions', '')
        criteria = q.get('criteria')
        if kind not in ('noul', 'choice', 'score'):
            raise ValueError('unknown question type')
        if kind in ('choice', 'score') and not criteria:
            raise ValueError('empty criteria')
        captured = []
        def capture(prompt, labels):
            captured.append((prompt, labels))
            return {k: 1 / len(labels) for k in labels}
        original = self.recipe.decide
        self.recipe.decide = capture
        try:
            getattr(self.recipe, kind)(task.state, instructions, criteria)
        finally:
            self.recipe.decide = original
        prompt, labels = captured[0]
        rename = {k: k for k in labels}
        if kind == 'noul': rename = {'true': 'yes', 'false': 'no'}
        elif kind == 'score': rename = {k: str(int(k) - 1) for k in labels}
        if set(rename.values()) != set(task.labels):
            raise ValueError('canonical labels do not match criteria')
        return prompt, labels, rename

    def run(self, task):
        result = Result()
        try:
            prompt, labels, rename = self.prepare(task)
        except (ValueError, KeyError, TypeError, IndexError):
            result.error_kind, result.error = 'input', 'invalid task specification'
            return result
        original = self.recipe._post
        transport_failure = False
        def record_post(path, body):
            nonlocal transport_failure
            exchange = {'path': path, 'request': body}
            result.exchanges.append(exchange)
            try:
                exchange['response'] = original(path, body)
                exchange['status'] = 200
                result.status = 200
                return exchange['response']
            except Exception as error:
                transport_failure = True
                result.status = getattr(error, 'code', None)
                exchange.update(error=type(error).__name__, status=result.status)
                raise
        self.recipe._post = record_post
        self.recipe.model_seconds_reset()
        start = time.perf_counter()
        try:
            native = self.recipe.decide(prompt, labels)
            if (set(native) != set(labels) or
                any(isinstance(p, bool) or not isinstance(p, (int, float)) or
                    not math.isfinite(p) or not 0 <= p <= 1 for p in native.values()) or
                abs(sum(native.values()) - 1) > 1e-6):
                raise ValueError('invalid native distribution')
            result.probs, result.ok = {rename[k]: p for k, p in native.items()}, True
        except Exception as error:
            # Exception messages can expose server content or credentials; types only.
            result.error_kind = 'transport' if transport_failure else 'distribution'
            result.error = type(error).__name__
        finally:
            result.latency_s = time.perf_counter() - start
            result.model_seconds = self.recipe.model_seconds()
            self.recipe._post = original
        completed = [e for e in result.exchanges if e['path'] == '/completion' and 'response' in e]
        if completed:
            exchange = completed[-1]
            response = exchange['response']
            if isinstance(response, dict):
                result.usage = {'input_tokens': response.get('tokens_evaluated'),
                                'output_tokens': response.get('tokens_predicted')}
                try:
                    top = {item['id']: item['prob'] for item in
                           response['completion_probabilities'][0]['top_probs']}
                    ids = [pair[0] for pair in exchange['request']['logit_bias']]
                    result.missing_options = [rename[k] for k, i in zip(labels, ids) if i not in top]
                    mass = sum(top[i] for i in ids if i in top)
                    if isinstance(mass, (int, float)) and math.isfinite(mass):
                        result.candidate_mass = mass
                except (KeyError, IndexError, TypeError):
                    pass  # Missing/malformed diagnostics never rescue a failed result.
        return result


def _json_safe(value):
    """Preserve malformed numerical evidence as strings, not invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value): return repr(value)
    if isinstance(value, dict): return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list): return [_json_safe(v) for v in value]
    return value


def run_public(tasks, adapter, output, *, identity=None):
    from jevbench.scoring import score_task
    from jevbench.summarize import summarize
    from jevbench.tasks import dataset_hash

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    raw_dir = output / 'raw'
    raw_dir.mkdir(mode=0o700)
    write_json(output / 'manifest.json', {
        'scope': SCOPE, 'upstream_pin': PIN, 'dataset_hash': dataset_hash(tasks),
        'planned': len(tasks), 'task_ids': [t.id for t in tasks], 'recipe': 'minijev.py',
        'identity': identity, 'model': MODEL, 'transport': 'loopback_llama.cpp',
        'implementation_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ['minijev.py', 'benchmarks/jevbench_bonsai.py']},
        'settings': {'n_predict': 1, 'temperature': 1.0, 'n_probs': 'min(40,max(20,2*N))',
                     'post_sampling_probs': True, 'logit_bias': 50.0,
                     'top_k': 0, 'top_p': 1.0, 'min_p': 0.0, 'typical_p': 1.0,
                     'top_n_sigma': -1.0, 'missing_policy': 'error'},
        'serial': True, 'no_retries': True, 'dotenv_loaded': False,
        'provider_api_dollar_spend': 0, 'inference_cost_usd': None,
        'cost_note': 'Local inference: electricity/hardware costs not measured.',
    })
    rows, consecutive_transport, stop_reason = [], 0, None
    with (output / 'records.jsonl').open('x', encoding='utf-8') as stream:
        for t in tasks:
            try:
                result = adapter.run(t)
            except Exception as error:
                result = Result(error_kind='transport', error=type(error).__name__)
            evidence = {'task_id': t.id, 'exchanges': result.exchanges}
            write_json(raw_dir / (hashlib.sha256(t.id.encode()).hexdigest() + '.json'), _json_safe(evidence))
            scored = score_task(result.probs or {}, t)
            record = {
                'task_id': t.id, 'task_sha256': hashlib.sha256(t.to_json().encode()).hexdigest(),
                'family': t.family, 'split': t.split, 'question_type': t.question['type'],
                'ok': result.ok, **scored, 'model': result.model,
                'probs_source': result.probs_source, 'error': result.error,
                'error_kind': result.error_kind, 'status_code': result.status,
                'latency_s': result.latency_s, 'model_seconds': result.model_seconds,
                'usage': result.usage, 'candidate_mass': result.candidate_mass,
                'missing_options': result.missing_options, 'missing_policy': 'error',
                'cost_usd': None, 'cost_basis': 'local_hardware_and_electricity_unmeasured',
            }
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            rows.append(record)
            print(f"{len(rows)}/{len(tasks)} {t.id}: {'valid' if scored['valid'] else 'failed'}", flush=True)
            consecutive_transport = consecutive_transport + 1 if result.error_kind == 'transport' else 0
            if result.status in (401, 403, 429) or consecutive_transport >= 3:
                stop_reason = 'access_or_transport'
                break
    summary = summarize(tasks, rows)
    summary.update(scope=SCOPE, upstream_pin=PIN, stop_reason=stop_reason,
                   official_score=None, official_rank=None, provider_api_dollar_spend=0,
                   inference_cost_usd=None, distribution_failures=sum(r['error_kind'] == 'distribution' for r in rows),
                   transport_failures=sum(r['error_kind'] == 'transport' for r in rows),
                   missing_candidate_failures=sum(bool(r['missing_options']) for r in rows),
                   calibration_note='Native biased-candidate probabilities; calibration measured on this public set only.',
                   latency_note='Serial shared-server HTTP round trip, including initial uncached tokenization; not exclusive-GPU throughput.',
                   cost_note='No provider API billing; electricity/hardware cost not measured.')
    summary['per_type'] = {
        kind: summarize([t for t in tasks if t.question['type'] == kind],
                        [r for r in rows if r['question_type'] == kind])
        for kind in sorted({t.question['type'] for t in tasks})
    }
    masses = [r['candidate_mass'] for r in rows if r['candidate_mass'] is not None]
    summary['candidate_mass'] = {'n': len(masses), 'min': min(masses) if masses else None,
                                 'max': max(masses) if masses else None,
                                 'mean': sum(masses) / len(masses) if masses else None}
    summary['completion_calls'] = sum(e['path'] == '/completion' for r in rows
        for e in json.loads((raw_dir / (hashlib.sha256(r['task_id'].encode()).hexdigest() + '.json')).read_text())['exchanges'])
    write_json(output / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['dry-run', 'run'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    tasks = load_public_tasks()
    if args.limit is not None:
        if args.limit < 1: parser.error('--limit must be positive')
        tasks = tasks[:args.limit]
    if args.action == 'dry-run':
        adapter = BonsaiAdapter(load_recipe())
        for t in tasks: adapter.prepare(t)
        print(json.dumps({'scope': SCOPE, 'planned': len(tasks), 'calls_sent': 0, 'upstream_pin': PIN}))
        return 0
    recipe = load_recipe(live=True)
    identity = verify_model(recipe)
    output = args.output or ROOT / 'benchmark_runs' / time.strftime('bonsai_dep_public_%Y%m%dT%H%M%S%z')
    summary = run_public(tasks, BonsaiAdapter(recipe), output, identity=identity)
    print(json.dumps({'output': str(output), 'complete': summary['complete'], 'accuracy': summary['accuracy']}))
    return 0 if summary['complete'] else 2


if __name__ == '__main__': raise SystemExit(main())
