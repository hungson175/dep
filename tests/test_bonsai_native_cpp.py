"""Native harness validation/sampler tests: compile only, never load a model."""
import json
import gzip
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LLAMA = Path('/home/hungson175/dev/llama.cpp-prism')


class NativeCppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='bonsai-native-tests-')
        cls.path = Path(cls.tmp.name)
        cls.binary = cls.path / 'bonsai-native'
        cls.compile_command = ['g++', '-std=c++17', '-O2', '-Wall', '-Wextra',
            '-I' + str(LLAMA / 'include'), '-I' + str(LLAMA / 'ggml/include'),
            '-I' + str(LLAMA / 'vendor'), str(ROOT / 'benchmarks/bonsai_native.cpp'),
            '-L' + str(LLAMA / 'build-cuda/bin'),
            '-Wl,-rpath,' + str(LLAMA / 'build-cuda/bin'), '-lllama', '-lggml', '-lggml-base', '-o', str(cls.binary)]
        if os.environ.get('BONSAI_CPP_COVERAGE_DIR'):
            cls.compile_command[1:1] = ['--coverage']
            cls.compile_command[cls.compile_command.index('-O2')] = '-O0'
        p = subprocess.run(cls.compile_command, capture_output=True, text=True)
        if p.returncode:
            cls.tmp.cleanup()
            raise AssertionError('Native compile failed:\n' + p.stderr)

    @classmethod
    def tearDownClass(cls):
        coverage_dir = os.environ.get('BONSAI_CPP_COVERAGE_DIR')
        if coverage_dir:
            notes = next(cls.path.glob('*.gcno'))
            p = subprocess.run(['gcov', '--json-format', str(notes)], cwd=cls.path,
                               capture_output=True, text=True)
            if p.returncode: raise AssertionError(p.stderr)
            data = json.loads(gzip.decompress(next(cls.path.glob('*.gcov.json.gz')).read_bytes()))
            source = next(x for x in data['files'] if x['file'].endswith('/benchmarks/bonsai_native.cpp'))
            lines = source['lines']
            covered = sum(x['count'] > 0 for x in lines)
            report = {'source': 'benchmarks/bonsai_native.cpp', 'covered_lines': covered,
                      'executable_lines': len(lines), 'line_coverage': covered / len(lines),
                      'note': 'GPU/model/decode paths intentionally unexecuted in unit tests.',
                      'line_records': [{k:x[k] for k in ['line_number', 'count']} for x in lines],
                      'functions': [{k:f[k] for k in ['name', 'execution_count', 'blocks', 'blocks_executed', 'start_line', 'end_line']}
                                    for f in source['functions']]}
            target = Path(coverage_dir); target.mkdir(parents=True, exist_ok=True)
            (target / 'cpp_coverage.json').write_text(json.dumps(report, indent=2))
            print(f'Native C++ no-model line coverage: {covered}/{len(lines)} ({100*covered/len(lines):.1f}%)')
        cls.tmp.cleanup()

    def run_binary(self, *args):
        return subprocess.run([str(self.binary), *map(str, args)], capture_output=True, text=True,
                              timeout=20, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})

    def job(self, **changes):
        return {'task_id': 'test-中文', 'prompt': '<|im_start|>user\nHi',
                'labels': {'alpha': ' 1', 'beta': ' 2'}, **changes}

    def jobs_file(self, name, jobs):
        p = self.path / name
        p.write_text(''.join(json.dumps(j, ensure_ascii=False) + '\n' for j in jobs))
        return p

    def test_help_and_cli_bounds_no_model_loading(self):
        p = self.run_binary('--help')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('--model', p.stdout)
        self.assertIn('--warmup', p.stdout)
        for args in [[], ['--unknown'], ['--ctx-size', '0'], ['--batch-size', '-1'],
                     ['--ubatch-size', '0'], ['--threads', '0'], ['--gpu-layers', '-2'],
                     ['--warmup', '-1'], ['--threads', 'oops'], ['--threads']]:
            p = self.run_binary(*args)
            self.assertNotEqual(p.returncode, 0)
            self.assertNotIn('llama_model_load', p.stderr)

    def test_validate_only_checks_whole_batch_without_weights(self):
        p = self.jobs_file('valid.jsonl', [self.job(), self.job(task_id='other')])
        r = self.run_binary('--validate-only', '--input', p)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout), {'valid_jobs': 2, 'model_loaded': False})
        bad_jobs = [self.job(task_id=''), self.job(prompt=''), self.job(labels={}),
                    self.job(labels={'x': ''}), self.job(labels={'x': 1}),
                    self.job(labels={'x': ' 1', 'y': ' 1'}),
                    self.job(labels={str(i): ' ' + str(i) for i in range(36)}),
                    self.job(task_id=None), self.job(prompt=[]), []]
        for n, bad in enumerate(bad_jobs):
            q = self.jobs_file(f'bad-{n}.jsonl', [self.job(task_id='first'), bad])
            r = self.run_binary('--validate-only', '--input', q)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('invalid_jobs', r.stderr)
        q = self.jobs_file('duplicate.jsonl', [self.job(), self.job()])
        self.assertNotEqual(self.run_binary('--validate-only', '--input', q).returncode, 0)
        q = self.jobs_file('empty.jsonl', [])
        self.assertNotEqual(self.run_binary('--validate-only', '--input', q).returncode, 0)
        q.write_text('SECRET_INVALID_JSON{')
        r = self.run_binary('--validate-only', '--input', q)
        self.assertNotIn('SECRET_INVALID_JSON', r.stderr)
        self.assertNotEqual(self.run_binary('--validate-only', '--input', '/not-found').returncode, 0)

    def test_output_is_never_overwritten_and_invalid_model_fails_without_inference(self):
        jobs = self.jobs_file('output-jobs.jsonl', [self.job()])
        output = self.path / 'existing.jsonl'; output.write_text('KEEP_ME')
        r = self.run_binary('--model', '/missing-model.gguf', '--input', jobs, '--output', output)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(output.read_text(), 'KEEP_ME')
        output = self.path / 'invalid-model-result.jsonl'
        r = self.run_binary('--model', '/missing-model.gguf', '--input', jobs, '--output', output)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('invalid_model', r.stderr)
        self.assertFalse(output.exists())

    def probe(self, value):
        path = self.path / 'sampler-probe.json'
        path.write_text(json.dumps(value))
        p = self.run_binary('--sampler-probe', path)
        self.assertEqual(p.returncode, 0, p.stderr)
        return json.loads(p.stdout)

    def test_native_float_sampler_and_bias_candidate_normalization(self):
        r = self.probe({'logits': [0.0, 1.0, 0.0], 'candidate_ids': {'a': 0, 'b': 1}})
        self.assertTrue(r['ok'])
        self.assertAlmostEqual(r['probs']['a'], 1 / (1 + 2.718281828459045), places=6)
        self.assertAlmostEqual(sum(r['probs'].values()), 1)
        self.assertAlmostEqual(r['candidate_mass'], 1, places=6)
        self.assertEqual(r['candidate_ids'], {'a': 0, 'b': 1})
        one = self.probe({'logits': [0, 0], 'candidate_ids': {'only': 1}})
        self.assertEqual(one['probs'], {'only': 1.0})
        unicode = self.probe({'logits': [2, 2], 'candidate_ids': {'中文': 0, 'other': 1}})
        self.assertEqual(unicode['probs'], {'中文': .5, 'other': .5})

    def test_strict_missing_top_n_underflow_suppression_and_invalid_numbers(self):
        for value, kind in [
            ({'logits': [100] * 22 + [-100], 'candidate_ids': {'missing': 22}}, 'distribution'),
            ({'logits': [0, -1000], 'candidate_ids': {'a': 0, 'b': 1}}, 'distribution'),
            ({'logits': [0, 0], 'candidate_ids': {'a': 0, 'b': 1}, 'suppress_ids': [1]}, 'distribution'),
            ({'logits': [0, 0], 'candidate_ids': {'a': 0, 'b': 1}, 'suppress_ids': [0, 1]}, 'distribution'),
            ({'logits': [float('nan'), 0], 'candidate_ids': {'a': 0}}, 'input'),
            ({'logits': [float('inf'), 0], 'candidate_ids': {'a': 0}}, 'input'),
            ({'logits': [], 'candidate_ids': {'a': 0}}, 'input'),
            ({'logits': [0], 'candidate_ids': {'a': 2}}, 'input'),
            ({'logits': [0], 'candidate_ids': {'a': -1}}, 'input'),
            ({'logits': [0], 'candidate_ids': {}}, 'input'),
            ({'logits': [0], 'candidate_ids': {'a': 0, 'b': 0}}, 'input')]:
            with self.subTest(value=value):
                r = self.probe(value)
                self.assertFalse(r['ok'])
                self.assertEqual(r['error_kind'], kind)

    def test_source_no_http_and_no_extra_answer_forward_pass(self):
        source = (ROOT / 'benchmarks/bonsai_native.cpp').read_text()
        self.assertNotIn('curl', source)
        self.assertNotIn('DEEPSEEK_API_KEY', source)
        self.assertNotIn('.env', source)
        self.assertIn('llama_get_logits_ith', source)
        self.assertIn('llama_synchronize', source)


if __name__ == '__main__': unittest.main()
