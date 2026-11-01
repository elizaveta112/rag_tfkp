#!/usr/bin/env python3
"""Evaluation annotations, isolated from corpus construction and prediction.

Selectors name actual textbook statements. No answer/gold enters the index.
"""
import argparse
import re
from pathlib import Path
from common import ROOT, read_jsonl, write_jsonl


def make_gold(corpus, references, own):
    def source(page, prefix):
        found = [b for b in corpus if b['page'] == page and re.search(prefix, b['text'])]
        if len(found) != 1:
            raise ValueError(f'Expected exactly one source: page={page}, pattern={prefix}, found={len(found)}')
        return found[0]['block_id']
    cr = source(11, r'^Теорема 2\.1\.')
    cr_proof = source(11, r'^Доказательство\.')
    hol = source(11, r'^Определение 2\.2\.')
    wir = source(12, r'^Наглядный способ')
    harmonic = source(13, r'^Определение 2\.3\.')
    primitive = source(26, r'^Другой аспект')
    orientation = source(26, r'^Аддитивность')
    circle = source(28, r'^Наконец, рассмотрим')
    exceptional = source(30, r'^Теорема 4\.4\.')
    local = source(31, r'^Замечание 4\.1\.')
    integral = source(31, r'^Теорема 5\.1\b')
    derivatives = source(33, r'^Следствие 5\.1\.')
    cauchy = source(34, r'^Теорема 5\.3\.')
    morera = source(35, r'^Теорема 5\.4\b')
    taylor = source(36, r'^Теорема 6\.1\.')
    entire = source(37, r'^К целым функциям')
    growth = source(37, r'^Теорема 6\.2\.')
    liouville = source(37, r'^Следствие 6\.1\b')
    identity = source(38, r'^Теорема 6\.3\b')
    exponent = source(20, r'^Экспонента\.')
    trig = source(21, r'^Тригонометрические функции')
    late_harmonic = source(119, r'^Напомним, что под гармонической')
    residue_coefficient = source(54, r'^Теорема 9\.1\.')
    residue_theorem = source(54, r'^Теорема 9\.2\.')
    dev_sets = [
        [[cr], [cr, cr_proof], [cr, hol]], [[wir]], [[cr, harmonic], [wir, harmonic], [cr, late_harmonic]],
        [[local, circle]], [[exceptional]],
        [[derivatives], [integral, derivatives]], [[exponent, cauchy], [exponent, residue_coefficient, residue_theorem]], [[morera]],
        [[cr, harmonic, cauchy], [late_harmonic, cauchy]], [[taylor]], [[growth]], [[identity, entire, trig]],
        [], [], []]
    dev_support = [None]*12+[[liouville], [orientation, circle], [taylor, exponent]]
    own_sets = [[[exceptional]], [[taylor]], [[cr, harmonic]],
                [[entire, identity, exponent]], [[cauchy]], [[taylor]], [], []]
    own_support = [None]*6+[[cauchy], [taylor, entire]]
    output = []
    for rows, sets, support in ((references, dev_sets, dev_support), (own, own_sets, own_support)):
        if len(rows) != len(sets):
            raise ValueError('Unexpected annotation set length')
        result = []
        for row, alternatives, supporting in zip(rows, sets, support):
            value = {k: row[k] for k in ('id', 'category', 'answer_ids', 'abstain', 'rationale')}
            value.update(citation_sets=alternatives, gold_status='reviewed')
            if value['abstain']:
                value['missing_information'] = row.get('missing_information', row['rationale'])
                value['support_block_ids'] = supporting
            result.append(value)
        output.append(result)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', default=str(ROOT/'corpus.jsonl'))
    p.add_argument('--references', default=str(ROOT/'dev_references.jsonl'))
    p.add_argument('--own-draft', default=str(ROOT/'own_gold.jsonl'))
    p.add_argument('--output-dir', default=str(ROOT))
    args = p.parse_args()
    dev, own = make_gold(read_jsonl(args.corpus), read_jsonl(args.references), read_jsonl(args.own_draft))
    write_jsonl(Path(args.output_dir)/'dev_gold.jsonl', dev)
    write_jsonl(Path(args.output_dir)/'own_gold.jsonl', own)
    print('Bound 15 dev and 8 author gold annotations to real extracted PDF block IDs.')


if __name__ == '__main__':
    main()
