#!/usr/bin/env python3
"""Local scorer; official evaluation scripts remain authoritative."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from common import read_jsonl, validate_answer, dump

CATEGORIES = ('conditions', 'comparison', 'application', 'insufficiency')


def tolerant_rows(path):
    result = defaultdict(list)
    malformed = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('id'), str):
                raise ValueError()
            result[row['id']].append(row)
        except (ValueError, TypeError):
            malformed.append(number)
    return result, malformed


def cost(t):
    return t['input_tokens']+4*t['output_tokens']+256*t['llm_calls']


def citation_f1(actual, alternatives):
    actual = set(actual)
    return max((2*len(actual & set(g))/(len(actual)+len(set(g)))
                for g in alternatives if g), default=0.)


def check_trace(t):
    caps = {'llm_calls': 4, 'input_tokens': 16384, 'output_tokens': 4096}
    if any(type(t.get(k)) is not int or not 0 <= t[k] <= cap for k, cap in caps.items()):
        return False
    if not isinstance(t.get('elapsed_ms'), (int, float)) or t['elapsed_ms'] < 0:
        return False
    events = t.get('events', [])
    if len(events) != t['llm_calls'] or t.get('usage_verifiable') is not True:
        return False
    if sum(e.get('input_tokens', -100000) for e in events) != t['input_tokens']:
        return False
    if sum(e.get('output_tokens', -100000) for e in events) != t['output_tokens']:
        return False
    return all(type(e.get('requested_output_cap')) is int and
               e['input_tokens']+e['requested_output_cap'] <= 8192 and
               0 <= e['output_tokens'] <= e['requested_output_cap'] for e in events)


def evaluate(questions, gold, answers_path, trace_path, corpus_ids, no_context=False, c_ref=None,
             semantic_reviews=None):
    by_q, by_g = {q['id']: q for q in questions}, {g['id']: g for g in gold}
    if set(by_q) != set(by_g):
        raise ValueError('Gold and questions must have exactly the same IDs')
    if not no_context:
        for g in gold:
            if g.get('gold_status', 'reviewed') != 'reviewed':
                raise ValueError('Gold source blocks are not reviewed; cannot claim citation Q')
            if not g['abstain'] and not g.get('citation_sets'):
                raise ValueError('Missing acceptable citation sets')
            if any(not s or len(s) > 4 or len(s) != len(set(s)) or not set(s) <= corpus_ids
                   for s in g.get('citation_sets', [])):
                raise ValueError('Invalid gold block-ID set')
    answers, malformed = tolerant_rows(answers_path)
    traces, bad_trace = tolerant_rows(trace_path)
    records, valid_traces = [], []
    accounting_ok = not bad_trace and set(traces) == set(by_q)
    extras = sorted(set(answers)-set(by_q))
    for qid, q in by_q.items():
        g = by_g[qid]
        a = answers[qid][0] if len(answers[qid]) == 1 else None
        t = traces[qid][0] if len(traces[qid]) == 1 else None
        trace_ok = t is not None and check_trace(t)
        accounting_ok = accounting_ok and trace_ok
        if t and all(type(t.get(k)) is int for k in ('input_tokens', 'output_tokens', 'llm_calls')):
            valid_traces.append(t)
        valid, reason = True, ''
        try:
            if a is None:
                raise ValueError('missing or duplicate answer')
            validate_answer(a, q, corpus_ids, no_context)
            if t and not set(a['citations']) <= set(t.get('context_block_ids', [])):
                raise ValueError('citation not present in delivered context')
            if not trace_ok:
                raise ValueError('unverifiable accounting or exceeded limits')
            if t.get('status') != 'ok':
                raise ValueError('technical failure')
        except (ValueError, TypeError, KeyError) as exc:
            valid, reason = False, str(exc)
        correct = bool(valid and a['abstain'] == g['abstain'] and
                       set(a['answer_ids']) == set(g['answer_ids']))
        f1 = citation_f1(a['citations'], g.get('citation_sets', [])) if valid else 0.
        score = float(correct) if g['abstain'] else float(correct)*(0.8+0.2*f1)
        review = (semantic_reviews or {}).get(qid)
        rejected = review and review['verdict'] == 'rejected'
        if rejected and not no_context:
            score = 0.
        if g['category'] not in CATEGORIES:
            raise ValueError(f"Unknown category {g['category']}; adapt gold schema explicitly")
        records.append({'id': qid, 'category': g['category'], 'valid': valid,
                        'correct': correct, 'citation_f1': f1, 'score': score, 'error': reason,
                        'semantic_reviewed': review is not None,
                        'semantic_error': review['reason'] if rejected else ''})
    category_means = {}
    if not no_context:
        for category in CATEGORIES:
            values = [r['score'] for r in records if r['category'] == category]
            if not values:
                raise ValueError(f'Missing category: {category}; Q requires all four')
            category_means[category] = statistics.mean(values)
    n = len(questions)
    result = {'answer_accuracy': sum(r['correct'] for r in records)/n,
              'Q': None if no_context else statistics.mean(category_means.values()),
              'mean_input_tokens': sum(t['input_tokens'] for t in valid_traces)/n,
              'mean_output_tokens': sum(t['output_tokens'] for t in valid_traces)/n,
              'mean_llm_calls': sum(t['llm_calls'] for t in valid_traces)/n,
              'mean_cost': sum(cost(t) for t in valid_traces)/n,
              'median_latency_ms': statistics.median([t.get('elapsed_ms', 0) for t in valid_traces])
                                   if valid_traces else None,
              'accounting_verified': accounting_ok, 'category_means': category_means,
              'semantic_reviewed': sum(r['semantic_reviewed'] for r in records),
              'records': records, 'extra_answer_ids': extras, 'malformed_answer_lines': malformed}
    if result['Q'] is not None and c_ref is not None:
        e = min(1., c_ref/max(result['mean_cost'], 1)) if accounting_ok else 0.
        result.update(E=e, C_ref=c_ref, estimated_total=75*result['Q']+25*result['Q']*e)
    return result


def main():
    p = argparse.ArgumentParser()
    for arg in ('questions', 'gold', 'answers', 'trace', 'corpus', 'output'):
        p.add_argument('--'+arg, required=True)
    p.add_argument('--no-context', action='store_true')
    p.add_argument('--reference-trace', help='Official baseline trace, NOT our single_rag trace')
    args = p.parse_args()
    reference = read_jsonl(args.reference_trace) if args.reference_trace else None
    c_ref = statistics.mean(cost(t) for t in reference) if reference else None
    result = evaluate(read_jsonl(args.questions), read_jsonl(args.gold), args.answers,
                      args.trace, {b['block_id'] for b in read_jsonl(args.corpus)},
                      args.no_context, c_ref)
    dump(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
