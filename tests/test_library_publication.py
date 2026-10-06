import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.publish_library import build_publication, validate_artifacts

ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / 'dist/dep_deepseek-0.1.0-py3-none-any.whl'
SDIST = ROOT / 'dist/dep_deepseek-0.1.0.tar.gz'


@unittest.skipUnless(WHEEL.is_file() and SDIST.is_file(), 'Build the library wheel and source archive first')
class PublicationTests(unittest.TestCase):
    def test_expected_artifacts_and_deterministic_manifest(self):
        manifest = validate_artifacts(WHEEL, SDIST)
        self.assertEqual(manifest['version'], '0.1.0')
        self.assertEqual(manifest['package'], 'dep-deepseek')
        self.assertEqual(set(manifest['files']), {WHEEL.name, SDIST.name})
        self.assertTrue(all(len(v['sha256']) == 64 for v in manifest['files'].values()))

    def test_build_complete_static_release_without_network(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp, patch('urllib.request.urlopen', side_effect=AssertionError('network')):
            out = Path(tmp) / 'release'
            manifest = build_publication(WHEEL, SDIST, out)
            html = (out / 'library.html').read_text()
            self.assertNotIn('{{', html)
            self.assertIn('https://dep.hungson175.com/static/packages/' + WHEEL.name, html)
            self.assertIn('#sha256=' + manifest['files'][WHEEL.name]['sha256'], html)
            self.assertIn('from dep_deepseek import DeepSeek', html)
            self.assertIn('0%', html)
            self.assertIn('82.25%', html)
            self.assertIn('not an official', html)
            self.assertEqual((out / 'packages' / WHEEL.name).read_bytes(), WHEEL.read_bytes())
            self.assertEqual(json.loads((out / 'packages/release.json').read_text()), manifest)
            self.assertEqual(len((out / 'packages/SHA256SUMS.txt').read_text().splitlines()), 2)
            self.assertEqual(set(p.name for p in out.iterdir()), {'library.html', 'packages'})
            with self.assertRaises(FileExistsError): build_publication(WHEEL, SDIST, out)

    def test_missing_artifact_does_not_create_partial_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'release'
            with self.assertRaises(ValueError): build_publication(Path(tmp) / WHEEL.name, SDIST, out)
            self.assertFalse(out.exists())

    def test_wrong_filename_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrong = Path(tmp) / 'other.whl'
            wrong.write_bytes(WHEEL.read_bytes())
            with self.assertRaises(ValueError): validate_artifacts(wrong, SDIST)

    def test_bad_archive_and_unexpected_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / WHEEL.name
            wheel.write_bytes(b'invalid')
            with self.assertRaises(ValueError): validate_artifacts(wheel, SDIST)
            wheel.write_bytes(WHEEL.read_bytes())
            with zipfile.ZipFile(wheel, 'a') as z:
                z.writestr('.env', 'TEST_ONLY_NOT_A_REAL_SECRET')
            with self.assertRaises(ValueError): validate_artifacts(wheel, SDIST)

    def test_wrong_metadata_and_changed_code_rejected(self):
        for changed in ['metadata', 'code', 'license']:
            with tempfile.TemporaryDirectory() as tmp:
                wheel = Path(tmp) / WHEEL.name
                with zipfile.ZipFile(WHEEL) as original, zipfile.ZipFile(wheel, 'w') as z:
                    for name in original.namelist():
                        content = original.read(name)
                        if changed == 'metadata' and name.endswith('/METADATA'):
                            content = content.replace(b'Version: 0.1.0', b'Version: 9.9.9')
                        if changed == 'code' and name == 'deepseek_flash.py': content += b'\n# changed\n'
                        if changed == 'license' and name.endswith('/licenses/LICENSE'): content = b'changed license'
                        z.writestr(name, content)
                with self.assertRaises(ValueError): validate_artifacts(wheel, SDIST)

    def test_source_archive_tampering_rejected(self):
        import io, tarfile
        for changed in ['invalid', 'extra', 'code']:
            with tempfile.TemporaryDirectory() as tmp:
                sdist = Path(tmp) / SDIST.name
                if changed == 'invalid':
                    sdist.write_bytes(b'invalid')
                else:
                    with tarfile.open(SDIST) as original, tarfile.open(sdist, 'w:gz') as archive:
                        for member in original.getmembers():
                            data = original.extractfile(member).read() if member.isfile() else None
                            if changed == 'code' and member.name.endswith('/deepseek_flash.py'):
                                data += b'\n# changed\n'; member.size = len(data)
                            archive.addfile(member, io.BytesIO(data) if data is not None else None)
                        if changed == 'extra':
                            member = tarfile.TarInfo('dep_deepseek-0.1.0/.env'); member.size = 4
                            archive.addfile(member, io.BytesIO(b'FAKE'))
                with self.assertRaises(ValueError): validate_artifacts(WHEEL, sdist)

    def test_cli_and_bad_template(self):
        from scripts.publish_library import main
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'cli_release'
            self.assertEqual(main(['--output', str(out)]), 0)
            other = Path(tmp) / 'bad_release'
            with patch('scripts.publish_library.Path.read_text', return_value='{{UNKNOWN}}'):
                with self.assertRaises(ValueError): build_publication(WHEEL, SDIST, other)
            self.assertFalse(other.exists())


if __name__ == '__main__':
    unittest.main()
