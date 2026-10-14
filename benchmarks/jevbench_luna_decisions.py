"""Public-only native-distribution benchmark using Luna Decisions, with durable cap."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from benchmarks.jevbench_flash import PIN, SCOPE, ROOT, load_public_tasks, write_json, validate_cap
from dep_luna_decisions import MODEL, ENDPOINT, DecisionsAdapter, DecisionsClient, read_key

PRICE_INPUT = .1
RESERVE = .21


def run(tasks, adapter, output, cap=20):
    validate_cap(cap)
    load_public_tasks()
    from jevbench.budget import Ledger, BudgetExceeded
    from jevbench.tasks import dataset_hash
    from jevbench.scoring import score_task
    from jevbench.summarize import summarize
    output = Path(output);output.mkdir(parents=True, exist_ok=False, mode=0o700)
    raw = output / 'raw';raw.mkdir(mode=0o700)
    ledger = Ledger(output.parent / 'luna_decisions_ledger.jsonl', cap)
    write_json(output / 'manifest.json', {'scope': SCOPE, 'upstream_pin': PIN,
        'dataset_hash': dataset_hash(tasks), 'planned': len(tasks), 'requested_model': MODEL,
        'endpoint': ENDPOINT, 'native_primitives': {'noul': 'predicate', 'choice': 'choice', 'score': 'score'},
        'no_retries': True, 'no_zero_fill': True, 'cap_usd': cap,
        'price_input_usd_per_million': PRICE_INPUT, 'pricing_checked': '2026-10-15',
        'pricing_source': 'https://developers.openai.com/api/docs/guides/decisions',
        'latency_basis': 'paid_decisions_api_round_trip_wall',
        'implementation_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ['dep_luna_decisions.py', 'benchmarks/jevbench_luna_decisions.py']}})
    rows, failures, stop = [], 0, None
    with (output / 'records.jsonl').open('x') as stream:
        for task in tasks:
            try: reservation = ledger.reserve(RESERVE, {'task_id': task.id})
            except BudgetExceeded:
                stop = 'budget';break
            result = adapter.run(task)
            write_json(raw / (hashlib.sha256(task.id.encode()).hexdigest() + '.json'),
                       {'request': result.request_body, 'response': result.raw, 'status': result.status})
            tokens = result.usage.get('input_tokens')
            cost = tokens * PRICE_INPUT / 1e6 if type(tokens) is int and tokens >= 0 else None
            ledger.settle(reservation, cost if cost is not None else RESERVE)
            scored = score_task(result.probs or {}, task)
            schema_error = scored.pop('error', None)
            row = dict(task_id=task.id, family=task.family, split=task.split,
                       question_type=task.question['type'], ok=result.ok, **scored,
                       probs_as_returned=result.probs, probs_source=result.probs_source, model=result.model,
                       latency_s=result.latency_s, usage=result.usage, status_code=result.status,
                       error=result.error, schema_error=schema_error, cost_usd=cost,
                       cost_basis='standard_input_tariff_upper_bound_not_invoice')
            stream.write(json.dumps(row, allow_nan=False) + '\n');stream.flush();os.fsync(stream.fileno())
            rows.append(row)
            print(f'{len(rows)}/{len(tasks)} {task.id}: {"valid" if result.ok else "failed"}', flush=True)
            failures = failures + 1 if not result.ok and result.status != 200 else 0
            if result.status in (401, 403, 429) or failures >= 3:
                stop = 'access_rate_or_consecutive_failure';break
    summary = summarize(tasks, rows, ledger.charged)
    costs = [r['cost_usd'] for r in rows]
    summary.update(scope=SCOPE, upstream_pin=PIN, model=MODEL, official_score=None, official_rank=None,
                   stop_reason=stop, cost_usd=sum(costs) if all(c is not None for c in costs) else None,
                   latency_basis='paid_decisions_api_round_trip_wall',
                   cost_note='Usage times standard input tariff; cached input not discounted; not an invoice.')
    summary['per_type'] = {kind: summarize([t for t in tasks if t.question['type'] == kind],
                         [r for r in rows if r['question_type'] == kind])
                         for kind in sorted({t.question['type'] for t in tasks})}
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=231)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.limit <= 231: parser.error('--limit must be 1..231')
    output = args.output or ROOT / 'benchmark_runs' / time.strftime('luna6_decisions_%Y%m%dT%H%M%S')
    summary = run(load_public_tasks()[:args.limit], DecisionsAdapter(DecisionsClient(read_key(Path.home() / 'dev/.env'))), output)
    print(json.dumps({'output': str(output), 'complete': summary['complete'],
                      'accuracy': summary['accuracy'], 'cost_usd': summary['cost_usd']}))
    return 0 if summary['complete'] else 2


if __name__ == '__main__': raise SystemExit(main())
