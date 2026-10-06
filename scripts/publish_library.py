"""Build a minimal static library release. Does not deploy or read credentials."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

from dep_deepseek import __version__

ROOT = Path(__file__).resolve().parents[1]
WHEEL_NAME = f'dep_deepseek-{__version__}-py3-none-any.whl'
SDIST_NAME = f'dep_deepseek-{__version__}.tar.gz'
PUBLIC_BASE = 'https://dep.hungson175.com/static/packages/'
MODULES = ['deepseek_flash.py', 'dep_deepseek/__init__.py', 'dep_deepseek/client.py']


def validate_artifacts(wheel: Path, sdist: Path) -> dict:
    """Fail before publication if archives contain unrelated files or stale code."""
    wheel, sdist = Path(wheel), Path(sdist)
    for path, name in [(wheel, WHEEL_NAME), (sdist, SDIST_NAME)]:
        if path.name != name or not path.is_file() or path.stat().st_size > 1_000_000:
            raise ValueError('Missing, oversized, or unexpected library artifact')
    info = f'dep_deepseek-{__version__}.dist-info/'
    allowed_wheel = set(MODULES + [info + name for name in [
        'licenses/LICENSE', 'METADATA', 'WHEEL', 'top_level.txt', 'RECORD']])
    try:
        with zipfile.ZipFile(wheel) as archive:
            if (set(archive.namelist()) != allowed_wheel or len(archive.namelist()) != len(allowed_wheel)
                    or any(m.file_size > 1_000_000 for m in archive.infolist())):
                raise ValueError('Wheel contains missing or unexpected files')
            metadata = BytesParser().parsebytes(archive.read(info + 'METADATA'))
            if (metadata['Name'] != 'dep-deepseek' or metadata['Version'] != __version__
                    or metadata.get_all('Requires-Dist') or metadata['License-Expression'] != 'MIT'):
                raise ValueError('Unexpected package metadata or runtime dependencies')
            for name in MODULES:
                if archive.read(name) != (ROOT / name).read_bytes():
                    raise ValueError('Wheel code differs from tested source')
            if archive.read(info + 'licenses/LICENSE') != (ROOT / 'LICENSE').read_bytes():
                raise ValueError('Wheel license differs from source')
        prefix = f'dep_deepseek-{__version__}/'
        allowed_source = set(MODULES + ['LICENSE', 'MANIFEST.in', 'PKG-INFO', 'README.md', 'pyproject.toml', 'setup.cfg',
            'docs/dep_deepseek.md', 'dep_deepseek.egg-info/PKG-INFO',
            'dep_deepseek.egg-info/SOURCES.txt', 'dep_deepseek.egg-info/dependency_links.txt',
            'dep_deepseek.egg-info/top_level.txt'])
        with tarfile.open(sdist) as archive:
            members = [m for m in archive.getmembers() if not m.isdir()]
            if (any(not m.isfile() or m.size > 1_000_000 for m in members)
                    or {m.name for m in members} != {prefix + n for n in allowed_source}
                    or len(members) != len(allowed_source)):
                raise ValueError('Source archive contains missing or unexpected files')
            for name in MODULES + ['LICENSE', 'MANIFEST.in', 'pyproject.toml', 'docs/dep_deepseek.md']:
                if archive.extractfile(prefix + name).read() != (ROOT / name).read_bytes():
                    raise ValueError('Source archive differs from tested source')
    except (zipfile.BadZipFile, tarfile.TarError, KeyError, EOFError):
        raise ValueError('Invalid library archive') from None
    return {'package': 'dep-deepseek', 'version': __version__, 'license': 'MIT',
            'python': '>=3.10', 'runtime_dependencies': [], 'model': 'deepseek-flash',
            'missing_policy': 'zero', 'hosted_inference': False,
            'files': {p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                               'bytes': p.stat().st_size} for p in [wheel, sdist]}}


def build_publication(wheel: Path, sdist: Path, output: Path) -> dict:
    manifest = validate_artifacts(wheel, sdist)
    html = (ROOT / 'site/library.html').read_text()
    for marker, value in {
        '{{VERSION}}': __version__, '{{WHEEL_NAME}}': WHEEL_NAME, '{{SDIST_NAME}}': SDIST_NAME,
        '{{WHEEL_URL}}': PUBLIC_BASE + WHEEL_NAME,
        '{{WHEEL_SHA256}}': manifest['files'][WHEEL_NAME]['sha256'],
    }.items():
        html = html.replace(marker, value)
    if '{{' in html:
        raise ValueError('Unresolved publication template marker')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    packages = output / 'packages'
    packages.mkdir()
    for path in [Path(wheel), Path(sdist)]:
        shutil.copyfile(path, packages / path.name)
    (packages / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (packages / 'SHA256SUMS.txt').write_text(''.join(
        f'{item["sha256"]}  {name}\n' for name, item in manifest['files'].items()))
    (output / 'library.html').write_text(html)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, default=ROOT / 'dist' / WHEEL_NAME)
    parser.add_argument('--sdist', type=Path, default=ROOT / 'dist' / SDIST_NAME)
    parser.add_argument('--output', type=Path, required=True, help='New staging directory, not a live deployment')
    args = parser.parse_args(argv)
    manifest = build_publication(args.wheel, args.sdist, args.output)
    print(json.dumps({'staged': str(args.output), 'version': manifest['version']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
