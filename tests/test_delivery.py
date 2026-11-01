"""Offline tests of input reproduction, safe continuation and release filtering."""
import json
import ast
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, write_jsonl, read_jsonl
from run_state import start_or_resume, verify_saved_prefix
from tools.convert_qa import convert
import package_submission
from request_journal import record_attempt, validate_live_copy
from review_results import matching_reviews, response_hash
from formulas import canonical


class DeliveryTests(unittest.TestCase):
    def test_actual_message_function_preserves_raw_formula_sources(self):
        tree = ast.parse((ROOT/'predict.py').read_text(encoding='utf-8'))
        definitions = [n for n in tree.body if
            (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('SYSTEM','NO_CONTEXT') for t in n.targets)) or
            (isinstance(n, ast.FunctionDef) and n.name == 'messages')]
        namespace = {'canonical': canonical}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(ROOT/'predict.py'), 'exec'), namespace)
        q = {'id':'math', 'question':r'Condition: \tfrac12(f_x+i f_y)=0',
             'options':[{'id':str(i),'text':r'\partial f/\partial\bar z=0'} for i in range(4)]}
        source = {'block_id':'original', 'page':12, 'section':'formula',
                  'text':r'$\frac{\partial f}{\partial\bar z}=0$; n! stays.'}
        content = namespace['messages'](q, [source], 'final')[1]['content']
        self.assertIn(r'\frac{1}{2}', content)
        self.assertNotIn(r'\\partial', content)
        self.assertTrue(content.endswith(source['text']))

    def test_live_attempt_after_snapshot_blocks_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);local = root/'project/attempts.jsonl';drive = root/'backup'
            with patch.dict('os.environ', {'RAG_ATTEMPT_LOG': str(local), 'RAG_CHECKPOINT_DIR': str(drive)}):
                record_attempt({'id': 'first', 'stage': 'answer'})
                validate_live_copy(local)
                with (drive/'request_attempts_live.jsonl').open('a') as stream:
                    stream.write(json.dumps({'id': 'later', 'stage': 'answer'})+'\n')
                with self.assertRaises(ValueError):
                    validate_live_copy(local)

    def test_content_annotation_does_not_apply_to_changed_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            answer = {'id': 'q', 'rationale': 'First generation'}
            path = Path(folder)/'reviews.json'
            path.write_text(json.dumps({'reviews': [{'mode': 'final', 'id': 'q',
                'answer_sha256': response_hash(answer), 'verdict': 'rejected', 'reason': 'False claim'}]}))
            self.assertIn('q', matching_reviews(path, 'final', [answer]))
            self.assertEqual({}, matching_reviews(path, 'final', [dict(answer, rationale='Other generation')]))

    def test_source_qa_reproduces_all_supplied_question_records(self):
        q, g = convert((ROOT/'data/tfkp_15_qa.md').read_text(encoding='utf-8'))
        self.assertEqual(q, read_jsonl(ROOT/'dev_questions.jsonl'))
        self.assertEqual(g, read_jsonl(ROOT/'dev_references.jsonl'))

    def test_continue_only_the_same_committed_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            a, t = Path(folder)/'answers.jsonl', Path(folder)/'trace.jsonl'
            questions = [{'id': 'first'}, {'id': 'second'}]
            identity = {'mode': 'final', 'input_hash': 'original'}
            self.assertEqual(0, start_or_resume(a, t, questions, identity))
            write_jsonl(a, [{'id': 'first'}]);write_jsonl(t, [{'id': 'first'}])
            self.assertEqual(1, start_or_resume(a, t, questions, identity, True))
            with self.assertRaises(ValueError):
                start_or_resume(a, t, questions, dict(identity, input_hash='changed'), True)

    def test_half_written_or_reordered_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            a, t = Path(folder)/'answers.jsonl', Path(folder)/'trace.jsonl'
            q, identity = [{'id': 'first'}, {'id': 'second'}], {'mode': 'final'}
            start_or_resume(a, t, q, identity)
            write_jsonl(a, [{'id': 'first'}]);write_jsonl(t, [{'id': 'second'}])
            with self.assertRaises(ValueError):
                start_or_resume(a, t, q, identity, True)
            write_jsonl(a, [{'id': 'second'}]);write_jsonl(t, [{'id': 'second'}])
            with self.assertRaises(ValueError):
                start_or_resume(a, t, q, identity, True)

    def test_charge_without_committed_trace_blocks_continuation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);journal = root/'endpoint.jsonl'
            write_jsonl(root/'final.trace.jsonl', [{'id':'q', 'events':[
                {'input_tokens':100, 'output_tokens':20}]}])
            journal.write_text(json.dumps({'id':'q','prompt_tokens':100,'completion_tokens':20})+'\n')
            verify_saved_prefix(root, journal, ['final'])
            with journal.open('a') as stream:
                stream.write(json.dumps({'id':'next','prompt_tokens':50,'completion_tokens':10})+'\n')
            with self.assertRaises(ValueError):
                verify_saved_prefix(root, journal, ['final'])

    def test_submission_excludes_large_or_private_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = ['README.md', 'report.md', 'STATUS.md', 'START_HERE.md', 'docs/formulas.md',
                'predict.py', 'data/tfkp_15_qa.md', 'dev_answers.jsonl', 'dev_trace.jsonl',
                'runtime.json', 'assets/model.gguf', 'index/index.json.gz', 'corpus.jsonl',
                'experiments/final.trace.jsonl', 'checkpoint/no_context.answers.jsonl',
                'smoke/answer.jsonl', '.venv311/notes.md']
            for name in names:
                p = root/name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('fixture')
            with patch.object(package_submission, 'ROOT', root):
                actual = {str(p.relative_to(root)) for p in package_submission.files()}
            self.assertEqual(actual, {'README.md','report.md','predict.py','data/tfkp_15_qa.md','dev_answers.jsonl',
                'dev_trace.jsonl','runtime.json'})


if __name__ == '__main__':
    unittest.main()
