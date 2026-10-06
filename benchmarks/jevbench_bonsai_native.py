"""Explicit Bonsai benchmark using an in-process libllama executable, not HTTP."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from benchmarks.jevbench_bonsai import BonsaiAdapter, MODEL, load_recipe
from benchmarks.jevbench_flash import PIN, ROOT, SCOPE, load_public_tasks, write_json

MODEL_PATH = Path('/home/hungson175/models/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf')
LIB_DIR = Path('/home/hungson175/dev/llama.cpp-prism/build-cuda/bin')
TIMES = ('tokenize', 'reset', 'prefill', 'sample', 'decision', 'local_total')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_jobs(tasks):
    adapter = BonsaiAdapter(load_recipe())
    jobs = []
    for task in tasks:
        prompt, labels, rename = adapter.prepare(task)
        jobs.append({'task_id': task.id, 'prompt': prompt,
                     'labels': {rename[k]: mark for k, mark in labels.items()}})
    return jobs


def _number(value, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError('invalid native numerical evidence')
    return value


def aggregate(tasks, output_rows):
    from jevbench.metrics import latency_summary
    from jevbench.scoring import score_task
    from jevbench.summarize import summarize

    if not output_rows or output_rows[0].get('type') != 'metadata':
        raise ValueError('missing native metadata')
    metadata = output_rows[0]
    by_id = {t.id: t for t in tasks}
    records, seen, times = [], set(), {key: [] for key in (*TIMES, 'local_wall')}
    for row in output_rows[1:]:
        task_id = row.get('task_id')
        if row.get('type') != 'result' or task_id not in by_id or task_id in seen:
            raise ValueError('invalid native task identity')
        seen.add(task_id)
        task = by_id[task_id]
        if not isinstance(row.get('ok'), bool):
            raise ValueError('invalid native success flag')
        values = {}
        probs = row.get('probs') if row['ok'] else {}
        if row['ok']:
            if not isinstance(probs, dict) or set(probs) != set(task.labels):
                raise ValueError('native label mismatch')
            if any(_number(p) > 1 for p in probs.values()) or abs(sum(probs.values()) - 1) > 1e-6:
                raise ValueError('invalid native probability distribution')
            tokens = row.get('prompt_tokens')
            if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
                raise ValueError('invalid native prompt length')
            values = {key: _number(row.get(key + '_ms')) for key in TIMES}
            if not math.isclose(values['decision'], values['prefill'] + values['sample'], abs_tol=.02):
                raise ValueError('native decision timer mismatch')
            if values['local_total'] + .02 < sum(values[k] for k in ('tokenize', 'reset', 'decision')):
                raise ValueError('native total timer undercounts components')
            values['local_wall'] = _number(row.get('local_wall_ms', values['local_total']))
            if values['local_wall'] + .02 < values['local_total']:
                raise ValueError('native wall timer undercounts components')
            for key in ('perf_prompt_ms', 'perf_eval_ms'):
                if key in row: _number(row[key])
            for key, value in values.items(): times[key].append(value / 1000)
        record = {'task_id': task_id, 'ok': row['ok'], **score_task(probs, task),
                  'model': MODEL, 'probs_source': 'native_libllama_biased_candidate_renormalized',
                  'latency_s': values.get('local_wall', None) / 1000 if values else None,
                  'timings_ms': values, 'prompt_tokens': row.get('prompt_tokens'),
                  'error_kind': row.get('error_kind'), 'status_code': None,
                  'cost_usd': None, 'cost_basis': 'hardware_and_electricity_unmeasured'}
        records.append(record)
    summary = summarize(tasks, records)
    summary.update(scope=SCOPE, upstream_pin=PIN, official_score=None, official_rank=None,
                   transport='in_process_libllama_no_http', native_metadata=metadata,
                   native_components={key: latency_summary(values) for key, values in times.items()},
                   latency_definition='native in-process decision wall time including tokenization, KV reset, synchronous prefill and probability extraction; excludes model load, context setup, warmups, HTTP, queue and disk I/O')
    summary['per_type'] = {kind: summarize([t for t in tasks if t.question['type'] == kind],
                        [r for r in records if by_id[r['task_id']].question['type'] == kind])
                        for kind in sorted({t.question['type'] for t in tasks})}
    return summary, records


def run_native(tasks, binary, model, output, *, ctx_size=4096, batch_size=2048,
               ubatch_size=512, threads=8, gpu_layers=99, warmup=2, lib_dir=LIB_DIR):
    from jevbench.tasks import dataset_hash

    binary, model, output, lib_dir = map(Path, (binary, model, output, lib_dir))
    jobs = build_jobs(tasks)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    input_path, raw_path = output / 'native_jobs.jsonl', output / 'native_output.jsonl'
    input_path.write_text(''.join(json.dumps(j, ensure_ascii=False) + '\n' for j in jobs))
    settings = {'ctx_size': ctx_size, 'batch_size': batch_size, 'ubatch_size': ubatch_size,
                'threads': threads, 'gpu_layers': gpu_layers, 'warmup': warmup}
    command = [str(binary.resolve()), '--model', str(model.resolve()), '--input', str(input_path.resolve()),
               '--output', str(raw_path.resolve())]
    for key, value in settings.items(): command.extend(['--' + key.replace('_', '-'), str(value)])
    write_json(output / 'manifest.json', {
        'scope': SCOPE, 'upstream_pin': PIN, 'dataset_hash': dataset_hash(tasks),
        'planned': len(tasks), 'task_ids': [t.id for t in tasks], 'settings': settings,
        'transport': 'in_process_libllama_no_http', 'cross_case_kv_cache': False,
        'warmup_count': warmup, 'no_retries': True, 'dotenv_loaded': False,
        'model_sha256': sha256_file(model), 'binary_sha256': sha256_file(binary),
        'jobs_sha256': sha256_file(input_path),
        'source_sha256': {name: sha256_file(ROOT / name) for name in
                          ('minijev.py', 'benchmarks/jevbench_bonsai.py',
                           'benchmarks/jevbench_bonsai_native.py', 'benchmarks/bonsai_native.cpp')},
        'libllama_sha256': sha256_file(lib_dir / 'libllama.so') if (lib_dir / 'libllama.so').exists() else None,
        'provider_api_dollar_spend': 0, 'inference_cost_usd': None,
    })
    environment = {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG', 'CUDA_VISIBLE_DEVICES') if k in os.environ}
    environment.update(LD_LIBRARY_PATH=str(lib_dir.resolve()), LC_ALL='C')
    try:
        with (output / 'native_stdout.log').open('x') as stdout, (output / 'native_stderr.log').open('x') as stderr:
            process = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr, timeout=540, check=False)
    except (subprocess.TimeoutExpired, OSError) as error:
        write_json(output / 'process_status.json', {'ok': False, 'error_kind': type(error).__name__, 'no_retry': True})
        raise RuntimeError('native benchmark process failed; no retry performed') from None
    write_json(output / 'process_status.json', {'ok': process.returncode == 0, 'returncode': process.returncode, 'no_retry': True})
    if process.returncode:
        raise RuntimeError('native benchmark exited unsuccessfully; no retry performed')
    output_rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    summary, records = aggregate(tasks, output_rows)
    (output / 'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    summary['native_output_sha256'] = sha256_file(raw_path)
    summary['records_sha256'] = sha256_file(output / 'records.jsonl')
    write_json(output / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('dry-run', 'run'))
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--model', type=Path, default=MODEL_PATH)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--lib-dir', type=Path, default=LIB_DIR)
    for flag, default in [('ctx-size', 4096), ('batch-size', 2048), ('ubatch-size', 512),
                          ('threads', 8), ('gpu-layers', 99), ('warmup', 2)]:
        parser.add_argument('--' + flag, type=int, default=default)
    args = parser.parse_args(argv)
    tasks = load_public_tasks()
    if args.command == 'dry-run':
        jobs = build_jobs(tasks)
        print(json.dumps({'planned': len(jobs), 'max_prompt_chars': max(len(j['prompt']) for j in jobs),
                          'transport': 'in_process_libllama_no_http', 'inference': False}))
        return 0
    if args.binary is None or args.output is None:
        parser.error('run requires --binary and a new --output directory')
    summary = run_native(tasks, args.binary, args.model, args.output, lib_dir=args.lib_dir,
                         **{k: getattr(args, k) for k in ('ctx_size', 'batch_size', 'ubatch_size', 'threads', 'gpu_layers', 'warmup')})
    print(json.dumps({key: summary[key] for key in ('complete', 'n_correct', 'accuracy', 'latency', 'native_components')}, allow_nan=False))
    return 0 if summary['complete'] else 1


if __name__ == '__main__': raise SystemExit(main())
