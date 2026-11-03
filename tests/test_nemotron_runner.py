import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class NemotronRunnerTest(unittest.TestCase):
    def test_default_corpus_includes_synthetic_multispeaker_clips(self):
        spec = importlib.util.spec_from_file_location('nemotron_matrix', ROOT / 'scripts' / 'nemotron_matrix.py')
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        jobs = module.matrix_jobs(Path('/corpus'))
        self.assertEqual(len(jobs), 18)
        self.assertIn(('tse3-three', Path('/corpus/corpus/tse3-three.wav')), jobs)

    def test_resume_keeps_matching_existing_results_without_retranscribing(self):
        spec = importlib.util.spec_from_file_location('nemotron_matrix', ROOT / 'scripts' / 'nemotron_matrix.py')
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / 'sample.wav'
            audio.write_bytes(b'original audio')
            existing = root / 'sample.nemotron.json'
            existing.write_text(json.dumps({'status': 'ok', 'model_revision': 'abc',
                                            'runtime_revision': 'xyz',
                                            'audio_sha256': module.sha256(audio)}))
            remaining, skipped = module.select_jobs([('sample', audio)], root, 'abc', 'xyz', True)
            self.assertEqual(remaining, [])
            self.assertEqual(skipped, ['sample'])
            with self.assertRaises(ValueError):
                module.select_jobs([('sample', audio)], root, 'wrong-revision', 'xyz', True)

    def test_empty_transcript_is_recorded_as_failure(self):
        spec = importlib.util.spec_from_file_location('nemotron_matrix', ROOT / 'scripts' / 'nemotron_matrix.py')
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class EmptyModel:
            def generate(self, audio, **kwargs):
                return SimpleNamespace(text='', segments=[])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / 'empty.wav'
            with wave.open(str(audio), 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(bytes(32000))
            result = module.run_clip(EmptyModel(), audio, root / 'result.json', {},
                                     {'task': 'asr', 'word_timestamps': False})
            self.assertEqual(result['status'], 'error')
            self.assertIn('empty transcript', result['error'])
            self.assertNotIn('text', result)

    def test_clip_produces_result_and_refuses_to_replace_it(self):
        spec = importlib.util.spec_from_file_location('nemotron_matrix', ROOT / 'scripts' / 'nemotron_matrix.py')
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class TestModel:
            def generate(self, audio, **kwargs):
                with wave.open(str(audio), 'rb') as w:
                    self.assertions = (w.getnchannels(), w.getframerate(), kwargs['task'])
                return SimpleNamespace(text='hello there', segments=[])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            wav = p / 'input.wav'
            with wave.open(str(wav), 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(bytes(32000))
            model = TestModel()
            out = p / 'result.json'
            settings = {'task': 'saa', 'temperature': 0.0, 'word_timestamps': True,
                        'speculative': False, 'max_tokens': 8192}
            module.run_clip(model, wav, out, {'model_revision': 'test-revision'}, settings)
            value = json.loads(out.read_text())
            self.assertEqual(value['text'], 'hello there')
            self.assertEqual(value['status'], 'ok')
            self.assertEqual(value['duration_s'], 1.0)
            self.assertEqual(value['model_revision'], 'test-revision')
            self.assertEqual(value['settings'], settings)
            self.assertEqual(model.assertions, (1, 16000, 'saa'))
            with self.assertRaises(FileExistsError):
                module.run_clip(model, wav, out, {'model_revision': 'test-revision'}, settings)
            self.assertEqual(json.loads(out.read_text())['text'], 'hello there')

    def test_suffixed_outputs_are_independent_of_plain_results(self):
        spec = importlib.util.spec_from_file_location('nemotron_matrix', ROOT / 'scripts' / 'nemotron_matrix.py')
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / 'sample.wav'
            audio.write_bytes(b'original audio')
            existing = root / 'sample.nemotron.json'
            existing.write_text(json.dumps({'status': 'ok', 'model_revision': 'abc',
                                            'runtime_revision': 'xyz',
                                            'audio_sha256': module.sha256(audio)}))
            remaining, skipped = module.select_jobs([('sample', audio)], root, 'abc', 'xyz',
                                                    True, suffix='-saa')
            self.assertEqual(remaining, [('sample', audio)])
            self.assertEqual(skipped, [])


if __name__ == '__main__':
    unittest.main()
