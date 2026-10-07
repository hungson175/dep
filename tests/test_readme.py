import re,unittest
from pathlib import Path

class ReadmeTests(unittest.TestCase):
 def test_short_honest_overview_with_one_real_benchmark_image(self):
  text=Path('README.md').read_text()
  self.assertTrue(text.startswith('# Dép —'))
  self.assertLessEqual(len(text.splitlines()),60)
  self.assertLessEqual(len(text.split()),400)
  images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text)
  self.assertEqual(images,['docs/benchmark_comparison.png'])
  self.assertTrue(Path(images[0]).read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
  for statement in ['We do not know how its private model works',
                    'not an official leaderboard score or rank',
                    'https://hungson175.github.io/dep/', 'first answer-token position']:
   self.assertIn(statement,text)
  self.assertNotIn('prompt injection inert',text)
