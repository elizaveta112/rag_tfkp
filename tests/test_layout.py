"""Migration preserves the checkout and refuses to overwrite local edits."""
import sys
from pathlib import Path
for folder in ('src','scripts'):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/folder))
import contextlib
import io
import json
import tempfile
import unittest
from common import ROOT,sha256
from install_layout import install


class LayoutTests(unittest.TestCase):
    def original_checkout(self, folder):
        # Test from the exact original source kit retained in the migration record.
        target=Path(folder)/'checkout';target.mkdir()
        legacy=json.loads((ROOT/'artifacts/layout_migration.json').read_text())['legacy_files_sha256']
        # A real original file is recoverable from the unchanged source QA.
        (target/'data').mkdir();(target/'data/tfkp_15_qa.md').write_bytes((ROOT/'data/tfkp_15_qa.md').read_bytes())
        (target/'.git').mkdir();(target/'.git/marker').write_text('keep history')
        (target/'my_notes.txt').write_text('keep unrelated file')
        return target

    def test_migration_and_idempotence_preserve_git_and_user_files(self):
        with tempfile.TemporaryDirectory() as folder:
            target=self.original_checkout(folder)
            with contextlib.redirect_stdout(io.StringIO()):
                backup=install(target)
                second=install(target)
            self.assertTrue(backup.is_file());self.assertIsNone(second)
            self.assertEqual((target/'.git/marker').read_text(),'keep history')
            self.assertEqual((target/'my_notes.txt').read_text(),'keep unrelated file')
            self.assertTrue((target/'src/predict.py').exists())
            self.assertFalse((target/'common.py').exists())

    def test_modified_legacy_file_blocks_all_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            target=self.original_checkout(folder)
            (target/'common.py').write_text('# local edits')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError,'Local edits detected'):
                install(target)
            self.assertEqual((target/'common.py').read_text(),'# local edits')
            self.assertFalse((target/'run.py').exists())

    def test_dry_run_does_not_change_files(self):
        with tempfile.TemporaryDirectory() as folder:
            target=self.original_checkout(folder)
            with contextlib.redirect_stdout(io.StringIO()):install(target,dry_run=True)
            self.assertFalse((target/'run.py').exists())
            self.assertEqual(list(Path(folder).glob('*.zip')),[])


if __name__=='__main__':unittest.main()
