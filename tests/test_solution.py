import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, dump, write_jsonl, sha256, load_config, validate_answer
from formulas import canonical, math_tokens, chunk_text
from retrieval import Retriever
from llm import Budget, TokenCounter, load_binding
from predict import parse_output
from decisions import to_public_answer
from evaluate import citation_f1, evaluate
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from json_math import restore_math_escapes


class FormulaTests(unittest.TestCase):
    def test_actual_option_ids_are_derived_from_model_verdicts(self):
        q = question('x')
        result = {'id': 'x', 'checks': [{'id': o['id'], 'reason': 'TEST FIXTURE',
                  'correct': o['id'] == 'opt2'} for o in reversed(q['options'])],
                  'rationale': 'TEST FIXTURE', 'abstain': False, 'citations': ['original']}
        self.assertEqual(to_public_answer(result, q)['answer_ids'], ['opt2'])
        result['checks'].append(result['checks'][0])
        with self.assertRaises(ValueError):
            to_public_answer(result, q)
    def test_real_model_tex_control_escapes(self):
        self.assertEqual(restore_math_escapes('\tfrac{1}{2}'), r'\tfrac{1}{2}')
        self.assertEqual(restore_math_escapes('\frac{1}{2}'), r'\frac{1}{2}')
        self.assertEqual(restore_math_escapes('\\\bb C'), r'\mathbb C')
        self.assertEqual(restore_math_escapes(r'\bar{z}'), r'\bar{z}')
        self.assertEqual(restore_math_escapes('Текст.\nНовая строка.'), 'Текст.\nНовая строка.')
    def test_conjugation_and_sign_are_preserved(self):
        self.assertEqual(canonical('z̄'), r'\bar{z}')
        self.assertIn('bar_z', math_tokens(r'\partial f/\partial\bar{z}=0'))
        self.assertNotEqual(math_tokens(r'\partial f/\partial z'),
                            math_tokens(r'\partial f/\partial\bar{z}'))
        self.assertNotEqual(math_tokens('f_x+i f_y'), math_tokens('f_x-i f_y'))

    def test_cosmetic_fraction_equivalence(self):
        self.assertEqual(math_tokens(r'\tfrac{a}{b}'), math_tokens(r'\frac{a}{b}'))
        self.assertEqual(math_tokens(r'\tfrac12'), math_tokens(r'\frac{1}{2}'))
        self.assertEqual(math_tokens(r'\bar z'), math_tokens(r'\bar{z}'))
        self.assertNotEqual(math_tokens('n!'), math_tokens('n'))
        self.assertNotEqual(math_tokens('z^2'), math_tokens('z^3'))
        self.assertNotEqual(math_tokens('f_x'), math_tokens('fx'))
        self.assertEqual(math_tokens('α+β'), math_tokens(r'\alpha+\beta'))

    def test_chunking_preserves_text_and_whole_formulas(self):
        formula = '$$'+r'\frac{\partial f}{\partial\bar{z}}=0 '*20+'$$'
        text = 'Условие.\n'+formula+' Продолжение. '*30
        chunks = list(chunk_text(text, 80))
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(any(formula in c for c in chunks))
        for c in chunks:
            self.assertEqual(c.count('$$') % 2, 0)

    def test_json_and_option_id_validation(self):
        q = question('x')
        a = {'id': 'x', 'answer_ids': ['opt7'], 'abstain': False,
             'rationale': 'Краткое обоснование.', 'citations': ['original']}
        self.assertEqual(validate_answer(a, q, {'original'}), a)
        bad = dict(a, answer_ids=['A'])
        with self.assertRaises(ValueError):
            validate_answer(bad, q, {'original'})
        with self.assertRaises(ValueError):
            validate_answer(dict(a, citations=['invented']), q, {'original'})
        self.assertEqual(parse_output('<think>private</think>```json\n'+json.dumps(a)+'\n```'), a)

    def test_best_acceptable_citation_set(self):
        self.assertEqual(citation_f1(['a', 'b'], [['a'], ['a', 'b']]), 1)
        self.assertAlmostEqual(citation_f1(['a', 'b'], [['a']]), 2/3)


def question(qid):
    return {'id': qid, 'question': r'Когда производная по сопряжённой переменной $\bar z$ равна нулю?',
            'options': [{'id': i, 'text': text} for i, text in zip(
                ['opt7', 'opt2', 'opt9', 'opt1'], ['Условие из источника.', 'Вариант 2.', 'Вариант 3.', 'Вариант 4.'])]}


def fixture(root):
    config = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    dump(root/'config.json', config)
    dump(root/'runtime.json', {'TEST_ONLY': 'synthetic endpoint, not a Qwen run'})
    binding = {'model_repo': 'Qwen/Qwen3.5-4B', 'model_revision': 'a'*40,
        'official_runtime_path': 'runtime.json', 'official_runtime_sha256': sha256(root/'runtime.json'),
        'request_body': {'max_tokens': 512, 'temperature': 0}, 'timeout_seconds': 5,
        'completion_budget_includes_reasoning': True}
    dump(root/'runtime_binding.json', binding)
    tokenizer_root = root/'assets/qwen_tokenizer'
    tokenizer_root.mkdir(parents=True)
    tok = Tokenizer(WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(tokenizer_root/'tokenizer.json'))
    (tokenizer_root/'chat_template.jinja').write_text(
        "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}assistant:", encoding='utf-8')
    dump(tokenizer_root/'manifest.json', {'revision': 'a'*40, 'sha256': {
        f: sha256(tokenizer_root/f) for f in ('tokenizer.json', 'chat_template.jinja')}})
    blocks = [
        {'block_id': 'original_1', 'page': 12, 'section': 'Тестовый раздел',
         'text': r'Для непрерывно дифференцируемой f условие голоморфности: $\partial f/\partial\bar z=0$.'},
        {'block_id': 'original_2', 'page': 12, 'section': 'Тестовый раздел',
         'text': r'Производная по z: $\partial f/\partial z=\frac12(f_x-i f_y)$.'},
        {'block_id': 'original_3', 'page': 40, 'section': 'Другой раздел',
         'text': r'Изолированная особая точка и ряд Лорана.'},
    ]
    write_jsonl(root/'corpus.jsonl', blocks)
    subprocess.run([sys.executable, str(ROOT/'build_index.py'), '--corpus', str(root/'corpus.jsonl'),
                    '--output', str(root/'index'), '--config', str(root/'config.json')],
                   check=True, capture_output=True)
    return load_config(root/'config.json'), blocks


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config, self.blocks = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_retrieval_and_reproducible_index(self):
        r = Retriever(self.root/'index')
        for mode in ('single_rag', 'final', 'no_formula', 'no_neighbors'):
            chunks, _ = r.search(question('x'), mode, self.config['retrieval'])
            self.assertTrue(chunks)
            self.assertTrue({c['block_id'] for c in chunks} <= {b['block_id'] for b in self.blocks})
            self.assertTrue(any(c['block_id'] == 'original_1' for c in chunks))
        before = sha256(self.root/'index/index.json.gz')
        subprocess.run([sys.executable, str(ROOT/'build_index.py'), '--corpus', str(self.root/'corpus.jsonl'),
                        '--output', str(self.root/'index'), '--config', str(self.root/'config.json')],
                       check=True, capture_output=True)
        self.assertEqual(before, sha256(self.root/'index/index.json.gz'))

    def test_budget_reserves_full_output_and_rejects_missing_usage(self):
        binding = load_binding(self.config)
        counter = TokenCounter(self.config, binding)
        budget = Budget(self.config, dict(binding, output_cap=4096), counter, 'x')
        msg = [{'role': 'user', 'content': 'Проверка.'}]
        self.assertTrue(budget.fits(msg))
        budget.output = 1
        self.assertFalse(budget.fits(msg))
        budget.output = 0
        budget.calls = 4
        self.assertFalse(budget.fits(msg))

    def test_changed_runtime_is_rejected(self):
        dump(self.root/'runtime.json', {'TEST_ONLY': 'changed'})
        with self.assertRaises(ValueError):
            load_binding(self.config)

    def test_predict_cli_mock_endpoint_and_scoring_from_other_directory(self):
        # Every response below is a TEST FIXTURE, never a claimed model answer.
        config = self.config
        counter = TokenCounter(config, load_binding(config))
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                qid = self.headers['X-Question-ID']
                seen.append((qid, self.headers['X-Submission-ID'], body['max_tokens']))
                if qid == 'bad_usage':
                    response = {'choices': [{'message': {'content': '{}'}}]}
                else:
                    content = {'id': qid, 'checks': [] if qid == 'insuff' else [
                               {'id': o['id'], 'reason': 'TEST FIXTURE',
                                'correct': o['id'] == 'opt7'} for o in question(qid)['options']],
                               'abstain': qid == 'insuff',
                               'rationale': 'TEST FIXTURE: недостаточно значения.' if qid == 'insuff'
                                            else 'TEST FIXTURE: условие приведено в источнике.',
                               'citations': [] if qid == 'insuff' else ['original_1']}
                    raw = json.dumps(content)
                    if qid == 'repair' and sum(s[0] == qid for s in seen) == 1:
                        raw = 'bad JSON'
                    response = {'choices': [{'message': {'content': raw}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': counter.count(body['messages']), 'completion_tokens': 50}}
                raw = json.dumps(response).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(raw)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        questions = [question(i) for i in ('first', 'repair', 'insuff', 'bad_usage')]
        write_jsonl(self.root/'questions.jsonl', questions)
        env = dict(os.environ, LLM_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',
                   LLM_MODEL='synthetic-test-fixture', LLM_API_KEY='test-only', SUBMISSION_ID='test-submission')
        try:
            subprocess.run([sys.executable, str(ROOT/'predict.py'), '--index', str(self.root/'index'),
                '--questions', str(self.root/'questions.jsonl'), '--config', str(self.root/'config.json'),
                '--output', str(self.root/'answers.jsonl'), '--trace', str(self.root/'trace.jsonl')],
                cwd='/', env=env, capture_output=True, check=True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        answers = [json.loads(s) for s in (self.root/'answers.jsonl').read_text().splitlines()]
        traces = [json.loads(s) for s in (self.root/'trace.jsonl').read_text().splitlines()]
        self.assertEqual([a['id'] for a in answers], [q['id'] for q in questions])
        self.assertEqual([t['llm_calls'] for t in traces], [1, 2, 1, 1])
        self.assertFalse(traces[-1]['usage_verifiable'])
        self.assertEqual(traces[-1]['status'], 'technical_failure')
        self.assertTrue(all(x[1] == 'test-submission' and x[2] == 512 for x in seen))
        gold = [{'id': q['id'], 'answer_ids': [] if q['id'] == 'insuff' else ['opt7'],
                 'abstain': q['id'] == 'insuff', 'category': category,
                 'citation_sets': [] if q['id'] == 'insuff' else [['original_1']]}
                for q, category in zip(questions, ('conditions', 'comparison', 'insufficiency', 'application'))]
        result = evaluate(questions, gold, self.root/'answers.jsonl', self.root/'trace.jsonl',
                          {b['block_id'] for b in self.blocks}, c_ref=100000)
        self.assertEqual(result['Q'], .75)
        self.assertEqual(result['E'], 0)
        self.assertAlmostEqual(result['estimated_total'], 56.25)


if __name__ == '__main__':
    unittest.main()
