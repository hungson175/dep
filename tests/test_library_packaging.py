import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
import unittest


class PackagingMetadataTests(unittest.TestCase):
    def test_explicit_minimal_package_contents(self):
        # Lightweight metadata check; a separately built wheel is also tested in isolation.
        text = Path('pyproject.toml').read_text()
        self.assertIn('name = "dep-deepseek"', text)
        self.assertIn('requires-python = ">=3.10"', text)
        self.assertIn('dependencies = []', text)
        self.assertIn('packages = ["dep_deepseek"]', text)
        self.assertIn('py-modules = ["deepseek_flash"]', text)
        self.assertIn('include-package-data = false', text)
        self.assertTrue(Path('docs/dep_deepseek.md').is_file())
        self.assertTrue(Path('LICENSE').is_file())
        self.assertIn('prune tests', Path('MANIFEST.in').read_text())
        for name in ['README.md', 'docs/dep_deepseek.md']:
            doc = Path(name).read_text()
            self.assertIn('releases/download/dep-deepseek-v0.1.0/', doc)
            self.assertIn('from dep_deepseek import DeepSeek', doc)

    def test_no_automatic_credentials_or_calls(self):
        import subprocess, sys
        code = ('from unittest.mock import patch; '
                'p=patch("builtins.open",side_effect=AssertionError("file read")); p.start(); '
                'import dep_deepseek; assert dep_deepseek.__version__ == "0.1.0"')
        subprocess.run([sys.executable, '-I', '-c',
                        'import sys; sys.path.insert(0, ' + repr(str(Path.cwd())) + '); ' + code], check=True)

    @unittest.skipUnless(os.environ.get('DEP_TEST_WHEEL'), 'Build a wheel and set DEP_TEST_WHEEL for isolated install E2E')
    def test_wheel_installed_outside_repository(self):
        wheel = Path(os.environ['DEP_TEST_WHEEL']).resolve()
        with zipfile.ZipFile(wheel) as z:
            files = set(z.namelist())
            self.assertIn('dep_deepseek/client.py', files)
            self.assertIn('deepseek_flash.py', files)
            for name in files:
                self.assertFalse(any(x in name for x in ['.env', 'benchmark_runs/', '.benchmark_cache/', 'app.py', 'minijev.py']))
            metadata = next(name for name in files if name.endswith('.dist-info/METADATA'))
            self.assertNotIn('Requires-Dist:', z.read(metadata).decode())
        with tempfile.TemporaryDirectory(prefix='dep_wheel_test_') as tmp:
            target = Path(tmp) / 'site'
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                            '--target', str(target), str(wheel)], check=True, capture_output=True)
            code = '''
import io,json,sys
from unittest.mock import patch
sys.path.insert(0, TARGET)
import dep_deepseek, deepseek_flash
assert dep_deepseek.__file__.startswith(TARGET)
assert deepseek_flash.__file__.startswith(TARGET)
assert dep_deepseek.__version__ == '0.1.0'
out={'choices':[{'logprobs':{'content':[{'top_logprobs':[{'token':'2','logprob':0.0}]}]}}]}
with patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(out).encode())) as post:
    result=dep_deepseek.DeepSeek(api_key='TEST_ONLY').choice('x','Pick?', ['a','b','c'])
assert result['choice']=='b'
assert result['probabilities']=={'a':0.0,'b':1.0,'c':0.0}
assert post.call_count==1
print('Isolated installed-wheel E2E PASS; no live API calls')
'''.replace('TARGET', repr(str(target)))
            subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp, check=True)


if __name__ == '__main__':
    unittest.main()
