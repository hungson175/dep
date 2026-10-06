import json
import os
from html.parser import HTMLParser
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'site/benchmarks'


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a': self.links += [v for k, v in attrs if k == 'href']


class BenchmarkPageTests(unittest.TestCase):
    def test_html_and_machine_readable_results_agree(self):
        data = json.loads((ROOT / 'results.json').read_text())
        html = (ROOT / 'index.html').read_text()
        self.assertEqual(data['planned'], 231)
        self.assertEqual(data['scope'], 'legacy_authored_public_not_official_rank')
        self.assertEqual(data['upstream_pin'], 'bb05a335bc809e61b20c0f745d25499a82b326fc')
        self.assertEqual({m['name'] for m in data['models']},
                         {'Bonsai-Dep', 'DeepSeek-Dep', 'Original Jev'})
        self.assertEqual(data['models'][0]['correct'], 188)
        self.assertEqual(data['models'][1]['correct'], 190)
        for model in data['models']:
            self.assertEqual(model['valid'], 231)
            self.assertAlmostEqual(model['accuracy'], model['correct'] / 231)
            self.assertEqual(sum(v['correct'] for v in model['per_type'].values()), model['correct'])
            self.assertEqual(sum(v['count'] for v in model['per_type'].values()), 231)
            self.assertIn(model['name'], html)
            self.assertIn(f"{model['accuracy'] * 100:.2f}%", html)
        self.assertEqual(data['deepseek_original_strict']['correct'], 178)
        self.assertEqual(data['deepseek_original_strict']['valid'], 218)
        self.assertIn('offline replay', html)
        self.assertIn('not an official', html)
        self.assertIn('hardware', html)
        self.assertIn('82.25%', html); self.assertIn('80.95%', html)
        self.assertIn('81.39%', html)

    def test_bonsai_latency_is_native_and_not_mixed_with_api_roundtrip(self):
        data = json.loads((ROOT / 'results.json').read_text())
        html = (ROOT / 'index.html').read_text()
        bonsai, deepseek, jev = data['models']
        self.assertEqual(jev['model'], 'jev-1.13.0')
        self.assertEqual(jev['latency_basis'], 'provider_api_round_trip_wall')
        self.assertEqual(jev['evidence'], 'fresh_original_typesafe_API_run')
        self.assertIn('original Jev report', html)
        self.assertEqual(bonsai['latency_basis'], 'native_inprocess_decision_wall')
        self.assertEqual(deepseek['latency_basis'], 'provider_http_round_trip')
        self.assertEqual(bonsai['native_components']['decision']['n'], 231)
        self.assertEqual(bonsai['p50_s'], bonsai['native_components']['local_wall']['p50_s'])
        self.assertEqual(bonsai['p95_s'], bonsai['native_components']['local_wall']['p95_s'])
        self.assertEqual(data['bonsai_original_http']['correct'], 187)
        alignment = data['official_protocol_alignment']
        self.assertEqual(alignment['version'], 'v1.6.1')
        self.assertEqual(alignment['status'], 'not_matched_missing_public_dataset_and_scorer')
        self.assertIsNone(data['official_score'])
        self.assertIn('O1S', html)
        self.assertIn('Native local decision', html)
        self.assertIn('Provider API round-trip', html)
        self.assertIn('not directly comparable', html)
        self.assertNotIn('<tr><td>HTTP round-trip p50</td>', html)

    def test_only_public_release_files_and_valid_links(self):
        self.assertTrue((ROOT / '.nojekyll').is_file())
        parser = Links(); parser.feed((ROOT / 'index.html').read_text())
        for link in parser.links:
            if '://' not in link and not link.startswith('#'):
                self.assertTrue((ROOT / link.split('#')[0]).is_file(), link)
        files = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
        self.assertEqual(files, {'index.html', 'results.json', '.nojekyll',
            'reports/bonsai_dep_public_20261006.md', 'reports/bonsai_dep_native_public_20261006.md',
            'reports/deepseek_flash_public_20261006.md',
            'reports/deepseek_flash_zero_fill_replay_20261006.md',
            'reports/jev_original_public_20261006.md'})

    @unittest.skipUnless(os.environ.get('DEP_BENCHMARK_PAGE_URL'), 'Explicit static-page browser check')
    def test_browser_metrics_mobile_and_evidence_links(self):
        from playwright.sync_api import sync_playwright, expect
        base = os.environ['DEP_BENCHMARK_PAGE_URL'].rstrip('/')
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(base + '/', wait_until='domcontentloaded')
            expect(page).to_have_title('Dep benchmark — Bonsai, DeepSeek & Jev')
            expect(page.get_by_text('81.39%', exact=True)).to_be_visible()
            expect(page.get_by_text('82.25%', exact=True)).to_be_visible()
            expect(page.get_by_text('Original Jev / jev-1.13.0', exact=True)).to_be_visible()
            for width in [390, 1280]:
                page.set_viewport_size({'width': width, 'height': 900})
                self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
            response = page.request.get(base + '/results.json')
            self.assertEqual(response.status, 200)
            self.assertEqual(response.json()['models'][0]['correct'], 188)
            self.assertEqual(page.request.get(base + '/reports/bonsai_dep_public_20261006.md').status, 200)
            self.assertEqual(page.request.get(base + '/reports/bonsai_dep_native_public_20261006.md').status, 200)
            browser.close()


if __name__ == '__main__': unittest.main()
