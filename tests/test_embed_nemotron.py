import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TargetedNemotronEmbedTest(unittest.TestCase):
    def test_adds_results_without_changing_existing_corpus_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            out = root / 'results' / 'corpus'
            out.mkdir(parents=True)
            shutil.copy(ROOT / 'scripts' / 'embed_nemotron.py', root / 'scripts')
            dashboard = root / 'results' / 'dashboard.html'
            original = (ROOT / 'results' / 'dashboard.html').read_text()
            start = '\n// ---- nemotron results ----'
            end = '// ---- end nemotron results ----'
            if start in original:
                prefix, suffix = original.split(start, 1)
                _, suffix = suffix.split(end, 1)
                original = prefix + suffix
            dashboard.write_text(original)
            (out / 'brook.nemotron.json').write_text(json.dumps({'status': 'ok', 'text': 'New </script> transcript'}))
            (out / 'sing.nemotron.json').write_text(json.dumps({'status': 'error', 'error': 'failure'}))
            run = subprocess.run([sys.executable, str(root / 'scripts' / 'embed_nemotron.py')],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            changed = dashboard.read_text()
            before, overlay = changed.split('// ---- nemotron results ----', 1)
            overlay, after = overlay.split('// ---- end nemotron results ----', 1)
            self.assertEqual(before[:-1] + after, original)
            self.assertIn('"brook.wav": "New <\\/script> transcript"', overlay)
            self.assertNotIn('</script>', overlay)
            self.assertNotIn('"sing.mp3":', overlay)
            again = subprocess.run([sys.executable, str(root / 'scripts' / 'embed_nemotron.py')],
                                   capture_output=True, text=True)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual(dashboard.read_text(), changed)


if __name__ == '__main__':
    unittest.main()
