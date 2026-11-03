import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NemotronDashboardTest(unittest.TestCase):
    def test_missing_synthetic_reference_refuses_to_rewrite_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            corpus = root / 'results' / 'corpus'
            corpus.mkdir(parents=True)
            shutil.copy(ROOT / 'scripts' / 'embed_corpus_transcripts.py', root / 'scripts')
            dashboard = root / 'results' / 'dashboard.html'
            dashboard.write_text('existing dashboard')
            (corpus / 'tse3-two.json').write_text(json.dumps({
                'duration_s': 1.0,
                'segments': [{'start': 0, 'end': 1, 'speaker_id': 0, 'text': 'model output'}],
            }))
            run = subprocess.run([sys.executable, str(root / 'scripts' / 'embed_corpus_transcripts.py')],
                                 text=True, capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('missing synthetic reference', run.stderr)
            self.assertEqual(dashboard.read_text(), 'existing dashboard')

    def test_real_result_is_embedded_and_failed_result_is_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            corpus = root / 'results' / 'corpus'
            corpus.mkdir(parents=True)
            shutil.copy(ROOT / 'scripts' / 'embed_corpus_transcripts.py', root / 'scripts')
            dashboard = ROOT / 'results' / 'dashboard.html'
            shutil.copy(dashboard, root / 'results' / 'dashboard.html')
            for stem in ('good', 'bad'):
                (corpus / f'{stem}.json').write_text(json.dumps({
                    'duration_s': 1.0,
                    'segments': [{'start': 0, 'end': 1, 'speaker_id': None, 'text': 'Reference'}],
                }))
            (corpus / 'good.nemotron.json').write_text(json.dumps({'status': 'ok', 'text': 'A real transcript'}))
            (corpus / 'bad.nemotron.json').write_text(json.dumps({'status': 'error', 'error': 'failed'}))
            run = subprocess.run([sys.executable, str(root / 'scripts' / 'embed_corpus_transcripts.py')],
                                 text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            html = (root / 'results' / 'dashboard.html').read_text()
            block = html.split('// ---- corpus (embedded', 1)[1].split('// ---- end corpus ----', 1)[0]
            good = block.split('"good.wav":', 1)[1]
            bad = block.split('"bad.wav":', 1)[1].split('"good.wav":', 1)[0]
            self.assertIn('nemotron: "A real transcript"', good)
            self.assertIn('nemotron: null', bad)
            self.assertIn("corpusKey: 'nemotron'", html)


if __name__ == '__main__':
    unittest.main()
