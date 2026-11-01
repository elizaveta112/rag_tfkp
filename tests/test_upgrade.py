"""Preserving prior measurements and rejecting mixed or contradictory decisions."""
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upgrade_project import install, RELEASE
from decisions import to_public_answer


class UpgradeTests(unittest.TestCase):
    def archive(self, path, corrupt=False):
        data = {'predict.py': b'# NEW SOURCE', 'config.json': b'{}'}
        manifest = {'release': RELEASE, 'results_included': False,
                    'files_sha256': {n: hashlib.sha256(v).hexdigest() for n, v in data.items()}}
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('rag_solution/run_kit_manifest.json', json.dumps(manifest))
            for n, v in data.items():
                z.writestr('rag_solution/'+n, b'BROKEN' if corrupt else v)

    def test_upgrade_keeps_weights_and_preserves_old_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'rag_solution';root.mkdir()
            (root/'predict.py').write_text('# OLD SOURCE')
            (root/'experiments.csv').write_text('OLD MEASUREMENTS')
            (root/'assets').mkdir();(root/'assets/model.gguf').write_bytes(b'EXISTING WEIGHTS')
            (root/'.venv').mkdir();(root/'.venv/marker').write_text('EXISTING ENVIRONMENT')
            archive = Path(folder)/'source.zip';self.archive(archive)
            with redirect_stdout(io.StringIO()):
                previous = install(archive, root)
            self.assertEqual((root/'assets/model.gguf').read_bytes(), b'EXISTING WEIGHTS')
            self.assertEqual((root/'.venv/marker').read_text(), 'EXISTING ENVIRONMENT')
            self.assertEqual((previous/'experiments.csv').read_text(), 'OLD MEASUREMENTS')
            self.assertFalse((root/'experiments.csv').exists())
            with zipfile.ZipFile(previous/'previous_source.zip') as z:
                self.assertEqual(z.read('predict.py'), b'# OLD SOURCE')

    def test_broken_archive_leaves_existing_project_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'rag_solution';root.mkdir()
            (root/'predict.py').write_text('# OLD SOURCE')
            (root/'experiments.csv').write_text('OLD MEASUREMENTS')
            archive = Path(folder)/'source.zip';self.archive(archive, True)
            with self.assertRaises(ValueError):
                install(archive, root)
            self.assertEqual((root/'predict.py').read_text(), '# OLD SOURCE')
            self.assertEqual((root/'experiments.csv').read_text(), 'OLD MEASUREMENTS')

    def test_refusal_cannot_hide_selected_options(self):
        question = {'id': 'new-id', 'options': [{'id': k} for k in ('choice-q', 'choice-z', 'choice-a', 'choice-r')]}
        model = {'id': question['id'], 'checks': [], 'rationale': 'TEST FIXTURE',
                 'abstain': True, 'citations': []}
        self.assertEqual(to_public_answer(model, question)['answer_ids'], [])
        model['checks'] = [{'id': 'choice-z', 'reason': 'TEST FIXTURE', 'correct': True}]
        with self.assertRaises(ValueError):
            to_public_answer(model, question)
        model['abstain'] = False
        with self.assertRaises(ValueError):
            to_public_answer(model, question)  # The other IDs must be checked too.


if __name__ == '__main__':
    unittest.main()
