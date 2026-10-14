"""Same pinned 231-case scorer; owner subscription, labels only, no retries."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from benchmarks.jevbench_flash import PIN, SCOPE, ROOT, load_public_tasks, write_json
from dep_luna import MODEL, ENDPOINT, SubscriptionClient, LunaAdapter


def run(tasks, adapter, output):
    if not 1 <= len(tasks) <= 231:
        raise ValueError('Run must contain 1..231 public cases')
    # Also verify pinned source before loading the canonical scorer.
    load_public_tasks()
    from jevbench.tasks import dataset_hash
    from jevbench.scoring import score_label
    from jevbench.summarize import summarize
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    raw = output / 'raw'
    raw.mkdir(mode=0o700)
    write_json(output / 'manifest.json', {
        'scope': SCOPE, 'upstream_pin': PIN, 'dataset_hash': dataset_hash(tasks),
        'planned': len(tasks), 'requested_model': MODEL, 'endpoint': ENDPOINT,
        'authentication': 'existing_owner_codex_chatgpt_subscription',
        'settings': {'reasoning_effort': 'none', 'no_retries': True,
                     'probabilities': 'unavailable_full_candidate_distribution',
                     'sampling': 'one_answer_per_case_backend_default_temperature'},
        'latency_basis': 'subscription_responses_round_trip_wall_including_local_parsing',
        'implementation_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ['dep_luna.py', 'benchmarks/jevbench_luna.py']}})
    rows, failures, stop = [], 0, None
    with (output / 'records.jsonl').open('x') as stream:
        for task in tasks:
            result = adapter.run(task)
            write_json(raw / (hashlib.sha256(task.id.encode()).hexdigest() + '.json'),
                       {'request': result.request_body, 'events': result.raw, 'status': result.status})
            scored = score_label(result.label if result.ok else None, task)
            schema_error = scored.pop('error', None)
            row = dict(task_id=task.id, family=task.family, split=task.split,
                       question_type=task.question['type'], ok=result.ok, model=result.model,
                       **scored, probs_source=result.probs_source, usage=result.usage,
                       latency_s=result.latency_s, status_code=result.status,
                       error=result.error, schema_error=schema_error, cost_usd=None)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            rows.append(row)
            print(f'{len(rows)}/{len(tasks)} {task.id}: {"valid" if result.ok else "failed"}', flush=True)
            failures = 0 if result.ok else failures + 1
            if result.status in (401, 403, 429) or failures >= 3:
                stop = 'access_rate_or_consecutive_failure'
                break
    summary = summarize(tasks, rows)
    summary.update(scope=SCOPE, upstream_pin=PIN, model=MODEL, official_score=None,
                   official_rank=None, stop_reason=stop, cost_usd=None,
                   cost_note='Owner subscription quota; not metered API, not free compute.',
                   probabilities_available=False, reasoning_tokens=sum(
                       r['usage'].get('reasoning_tokens') or 0 for r in rows),
                   latency_basis='subscription_responses_round_trip_wall_including_local_parsing')
    summary['per_type'] = {kind: summarize([t for t in tasks if t.question['type'] == kind],
                         [r for r in rows if r['question_type'] == kind])
                         for kind in sorted({t.question['type'] for t in tasks})}
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, default=231)
    args = parser.parse_args()
    if not 1 <= args.limit <= 231:
        parser.error('--limit must be 1..231')
    output = args.output or ROOT / 'benchmark_runs' / time.strftime('luna6_subscription_%Y%m%dT%H%M%S')
    summary = run(load_public_tasks()[:args.limit], LunaAdapter(SubscriptionClient()), output)
    print(json.dumps({'output': str(output), 'complete': summary['complete'],
                      'accuracy': summary['accuracy']}))
    return 0 if summary['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
