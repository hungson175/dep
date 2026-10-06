"""Explicit Playwright journey; staging uses fake providers, live uses real calls."""
import json
import os
import unittest


@unittest.skipUnless(os.environ.get('DEP_E2E_URL'), 'Set DEP_E2E_URL for explicit browser E2E')
class ModelUiE2E(unittest.TestCase):
    def test_library_page_and_real_wheel_download(self):
        import hashlib, subprocess, sys, tempfile
        from pathlib import Path
        from playwright.sync_api import sync_playwright, expect
        with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(accept_downloads=True)
            page.goto(os.environ['DEP_E2E_URL'].rstrip('/') + '/static/library.html')
            expect(page).to_have_title('dep DeepSeek — Python library')
            for width in [390, 1280]:
                page.set_viewport_size({'width': width, 'height': 900})
                self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
            with page.expect_download() as event:
                page.get_by_role('link', name='Download wheel ↓').click()
            download = event.value
            wheel = Path(tmp) / download.suggested_filename
            download.save_as(wheel)
            manifest = page.request.get(os.environ['DEP_E2E_URL'].rstrip('/') + '/static/packages/release.json').json()
            self.assertEqual(hashlib.sha256(wheel.read_bytes()).hexdigest(), manifest['files'][wheel.name]['sha256'])
            target = Path(tmp) / 'installed'
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                            '--target', str(target), str(wheel)], check=True, capture_output=True)
            subprocess.run([sys.executable, '-I', '-c', 'import sys; sys.path.insert(0,' + repr(str(target)) +
                '); from dep_deepseek import DeepSeek; assert "0.1.0" == __import__("dep_deepseek").__version__'],
                cwd=tmp, check=True)
            browser.close()

    def test_default_bonsai_then_explicit_deepseek(self):
        from playwright.sync_api import sync_playwright, expect
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            errors = []; sent = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.on('request', lambda r: sent.append(json.loads(r.post_data))
                    if r.url.endswith('/api/decide') and r.method == 'POST' else None)
            # The playground polls /api/stats; networkidle is not a readiness gate.
            page.goto(os.environ['DEP_E2E_URL'], wait_until='domcontentloaded')
            selector = page.get_by_label('Model', exact=True)
            expect(selector).to_have_value('bonsai')
            page.get_by_role('button', name='Run', exact=True).click()
            expect(page.locator('#code')).to_contain_text('HTTP 200', timeout=120000)
            first = json.loads(page.locator('#res').inner_text())
            self.assertIn('bonsai', first['model']); self.assertEqual(first['cost_usd'], 0)
            selector.select_option('deepseek-flash')
            expect(page.locator('#bz')).to_be_disabled()
            expect(page.locator('#costmeta')).to_contain_text('$20')
            self.assertEqual(json.loads(page.locator('#req').input_value())['model'], 'deepseek-flash')
            page.get_by_role('button', name='Run', exact=True).click()
            expect(page.locator('#code')).to_contain_text('HTTP 200', timeout=120000)
            second = json.loads(page.locator('#res').inner_text())
            self.assertEqual(second['model'], 'deepseek-flash')
            probs = second['answers']['route']['probabilities']
            self.assertEqual(set(probs), {'billing', 'technical', 'account', 'feedback'})
            self.assertAlmostEqual(sum(probs.values()), 1.)
            self.assertIsNotNone(second['cost_usd'])
            self.assertEqual([body['model'] for body in sent], ['bonsai', 'deepseek-flash'])
            selector.select_option('bonsai')
            expect(page.locator('#costmeta')).to_contain_text('no tokens billed')
            self.assertEqual(errors, [])
            context.close(); browser.close()


if __name__ == '__main__': unittest.main()
