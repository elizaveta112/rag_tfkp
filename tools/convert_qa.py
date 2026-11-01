#!/usr/bin/env python3
"""Convert attached QA Markdown to questions + REFERENCE METADATA, never corpus."""
import argparse
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import write_jsonl, validate_question

CATEGORIES = {'условия': 'conditions', 'сопоставление': 'comparison',
              'применение': 'application', 'недостаточность данных': 'insufficiency'}


def convert(text):
    questions, references = [], []
    for match in re.finditer(r'^## (\d+)\. ([^\n]+)\n([\s\S]*?)(?=^## \d+\.|\Z)', text, re.M):
        number, title, body = match.groups()
        qtext = re.search(r'\*\*Вопрос\.\*\*\s*([\s\S]*?)(?=\n- \*\*)', body)
        options = re.findall(r'^- \*\*([A-Z])\.\*\*\s*(.+)$', body, re.M)
        answer = re.search(r'\*\*Ответ:\*\*\s*([^\n]+)', body)
        category = re.search(r'\*\*Тип:\*\*\s*([^.]+)', body)
        source = re.search(r'\*\*Источник:\*\*\s*([^\n]+)', body)
        if not all((qtext, answer, category, source)):
            raise ValueError(f'Cannot parse QA #{number}')
        abstain = answer[1].strip().startswith('Отказ')
        ids = [] if abstain else re.findall(r'\b[A-Z]\b', answer[1])
        rationale = body[answer.end():source.start()].strip()
        pages_text = re.search(r'PDF с\.\s*([\d, ]+)', source[1])
        pages = list(map(int, re.findall(r'\d+', pages_text[1]))) if pages_text else []
        q = {'id': f'qa_{int(number):03d}', 'question': qtext[1].strip(),
             'options': [{'id': i, 'text': t} for i, t in options]}
        validate_question(q)
        questions.append(q)
        references.append({'id': q['id'], 'category': CATEGORIES[category[1].strip()],
            'answer_ids': ids, 'abstain': abstain, 'rationale': rationale,
            'source_pages': pages, 'source_description': source[1],
            'citation_sets': [], 'gold_status': 'pending_original_block_ids', 'title': title})
    if len(questions) != 15:
        raise ValueError(f'Expected 15 QA pairs, parsed {len(questions)}')
    return questions, references


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--markdown', required=True)
    p.add_argument('--questions', required=True)
    p.add_argument('--references', required=True)
    args = p.parse_args()
    questions, references = convert(Path(args.markdown).read_text(encoding='utf-8'))
    write_jsonl(args.questions, questions)
    write_jsonl(args.references, references)


if __name__ == '__main__':
    main()
