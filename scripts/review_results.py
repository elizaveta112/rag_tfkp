import sys as _sys
from pathlib import Path as _Path
for _directory in (_Path(__file__).resolve().parents[1], _Path(__file__).resolve().parents[1]/'src', _Path(__file__).resolve().parents[1]/'scripts'):
    _sys.path.insert(0, str(_directory))
from common import project_path
#!/usr/bin/env python3
"""Optional manual content annotations. Never imported by prediction."""
import hashlib
import json
from pathlib import Path
from common import ROOT, read_jsonl, dump


def response_hash(answer):
    return hashlib.sha256(json.dumps(answer, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def matching_reviews(path, mode, answers):
    if not Path(path).exists():
        return {}
    actual = {a['id']: a for a in answers}
    matches = {}
    for row in json.loads(Path(path).read_text(encoding='utf-8'))['reviews']:
        if row['mode'] != mode or row['id'] not in actual or row['verdict'] == 'pending':
            continue
        if row['answer_sha256'] != response_hash(actual[row['id']]):
            continue
        if row['verdict'] not in ('accepted', 'rejected') or (row['verdict'] == 'rejected' and not row['reason'].strip()):
            raise ValueError('Content review requires accepted/rejected and a reason for rejection')
        if row['id'] in matches:
            raise ValueError('Duplicate content review')
        matches[row['id']] = row
    return matches


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='Rebuild pending rows for changed answers; retain only hash-matched existing judgments')
    args = parser.parse_args()
    destination = project_path(ROOT, 'semantic_reviews.json')
    previous_document = json.loads(destination.read_text()) if destination.exists() else {}
    previous = previous_document.get('reviews', [])
    saved = {(x['mode'], x['id'], x['answer_sha256']): x for x in previous}
    if destination.exists() and not args.refresh:
        raise FileExistsError('semantic_reviews.json already exists; existing author annotations are not overwritten')
    rows, text = [], ['# Приватная проверка математических обоснований\n']
    qmap = {q['id']: q for q in read_jsonl(project_path(ROOT, 'dev_questions.jsonl'))}
    gmap = {q['id']: q for q in read_jsonl(project_path(ROOT, 'dev_gold.jsonl'))}
    for mode in ('single_rag', 'final', 'no_formula', 'no_neighbors'):
        for answer in read_jsonl(ROOT/f'experiments/{mode}.answers.jsonl'):
            qid = answer['id']
            rows.append({'mode': mode, 'id': qid, 'answer_sha256': response_hash(answer),
                         'verdict': 'pending', 'reason': ''})
            text += [f'## {mode} / {qid}\n', qmap[qid]['question'],
                '\n'.join(o['id']+': '+o['text'] for o in qmap[qid]['options']),
                f"\nМодель: {answer['answer_ids']}, отказ={answer['abstain']}. "
                f"Эталон: {gmap[qid]['answer_ids']}, отказ={gmap[qid]['abstain']}.",
                '\nОбоснование модели: '+answer['rationale']+'\n',
                'Эталонный довод: '+gmap[qid]['rationale']+'\n']
    rows = [saved.get((x['mode'], x['id'], x['answer_sha256']), x) for x in rows]
    dump(destination, {**{k:v for k,v in previous_document.items() if k!='reviews'}, 'instructions': 'Set verdict to accepted or rejected; a rejected answer requires a mathematical reason. Pending annotations do not change scores.', 'reviews': rows})
    folder = ROOT/'private_review';folder.mkdir(exist_ok=True)
    (folder/'answers.md').write_text('\n\n'.join(text)+'\n', encoding='utf-8')
    print('Created semantic_reviews.json and private_review/answers.md. Edit verdict/reason, then rerun finalize_results.py.')


if __name__ == '__main__':
    main()
