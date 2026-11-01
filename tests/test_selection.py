"""Reject incoherent final deliverables without loading a model or index."""
import copy
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from common import ROOT, read_jsonl
from validate_submission import validate_selection


class SelectionTests(unittest.TestCase):
    def setUp(self):
        if not (ROOT/'selection.json').exists():
            self.skipTest('No finalized measured study yet')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('selection.json', 'config.json', 'measured_config.json', 'study_provenance.json',
                     'semantic_reviews.json', 'dev_answers.jsonl', 'dev_trace.jsonl'):
            shutil.copyfile(ROOT/name, self.root/name)
        self.answers=read_jsonl(self.root/'dev_answers.jsonl')
        self.traces=read_jsonl(self.root/'dev_trace.jsonl')
        with (ROOT/'experiments.csv').open() as f:
            self.rows=list(csv.DictReader(f))

    def verify(self, traces=None, rows=None):
        validate_selection(self.root, self.answers, traces or self.traces, rows or self.rows)

    def test_coherent_measured_selection_passes(self):
        self.verify()

    def test_wrong_trace_mode_fails(self):
        changed=copy.deepcopy(self.traces); changed[0]['mode']='single_rag' if changed[0]['mode']!='single_rag' else 'final'
        with self.assertRaisesRegex(ValueError, 'trace and selection differ'):
            self.verify(traces=changed)

    def test_post_measurement_retrieval_change_fails(self):
        path=self.root/'config.json';cfg=json.loads(path.read_text());cfg['retrieval']['seed_chunks']+=1
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, 'Inference settings changed'):
            self.verify()

    def test_manually_repaired_answer_fails(self):
        path=self.root/'dev_answers.jsonl';path.write_text(path.read_text().replace('rationale', 'rationale_modified', 1))
        with self.assertRaisesRegex(ValueError, 'answers differ'):
            self.verify()

    def test_wrong_csv_score_fails(self):
        rows=copy.deepcopy(self.rows);mode=json.loads((self.root/'selection.json').read_text())['selected_mode']
        next(r for r in rows if r['name']==mode)['Q']='1.0'
        with self.assertRaisesRegex(ValueError, 'CSV/selection mismatch'):
            self.verify(rows=rows)


if __name__=='__main__':
    unittest.main()
