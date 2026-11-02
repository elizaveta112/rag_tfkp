"""Editorial report changes survive an explicitly requested layout migration."""
import sys
from pathlib import Path
for folder in ('src','scripts'):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/folder))
import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from common import sha256
from install_layout import install


class KeepReportTests(unittest.TestCase):
    def test_keep_report_preserves_bytes_and_updates_only_delivery_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'repo';root.mkdir()
            original='# Мой отчёт\n\nРедакция автора.\n'.encode('utf-8')
            (root/'report.md').write_bytes(original)
            with contextlib.redirect_stdout(io.StringIO()):
                backup=install(root,keep_report=True)
                second=install(root,keep_report=True)
            self.assertIsNone(second)
            self.assertEqual((root/'report.md').read_bytes(),original)
            manifest=json.loads((root/'artifacts/completion_manifest.json').read_text())
            self.assertEqual(manifest['files_sha256']['report.md'],sha256(root/'report.md'))
            self.assertEqual(manifest['retained_report']['sha256'],sha256(root/'report.md'))
            with zipfile.ZipFile(backup) as z:self.assertEqual(z.read('report.md'),original)

    def test_changed_report_without_flag_still_blocks_migration(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'repo';root.mkdir();(root/'report.md').write_text('My edits')
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'Local edits detected'):
                install(root)
            self.assertFalse((root/'run.py').exists())

    def test_oversized_report_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'repo';root.mkdir();(root/'report.md').write_text('word '*1501)
            with self.assertRaisesRegex(ValueError,'at most 1500'):
                install(root,keep_report=True)
            self.assertFalse((root/'run.py').exists())


if __name__=='__main__':unittest.main()
