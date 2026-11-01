#!/usr/bin/env python3
"""Reconcile all five traces with the local endpoint's independent counters."""
from collections import defaultdict
from common import ROOT, read_jsonl, dump, sha256


def main():
    traces, endpoint = defaultdict(lambda: [0, 0, 0]), defaultdict(lambda: [0, 0, 0])
    for mode in ('no_context', 'single_rag', 'final', 'no_formula', 'no_neighbors'):
        for t in read_jsonl(ROOT/f'experiments/{mode}.trace.jsonl'):
            target = traces[t['id']]
            target[0] += t['llm_calls']
            target[1] += t['input_tokens']
            target[2] += t['output_tokens']
    # Endpoint contains repeated IDs across modes, unlike question JSONL.
    import json
    for line in (ROOT/'endpoint_usage.jsonl').read_text().splitlines():
        row = json.loads(line)
        target = endpoint[row['id']]
        target[0] += 1
        target[1] += row['prompt_tokens']
        target[2] += row['completion_tokens']
    if dict(traces) != dict(endpoint):
        raise ValueError('Endpoint and trace accounting differ; inspect logs before submitting')
    from run_state import verify_saved_prefix
    verify_saved_prefix(ROOT/'experiments', ROOT/'endpoint_usage.jsonl',
                        ('no_context', 'single_rag', 'final', 'no_formula', 'no_neighbors'),
                        ROOT/'request_attempts.jsonl' if (ROOT/'request_attempts.jsonl').exists() else None)
    total = [sum(v[i] for v in traces.values()) for i in range(3)]
    dump(ROOT/'usage_reconciliation.json', {'verified': True, 'llm_calls': total[0],
         'input_tokens': total[1], 'output_tokens': total[2], 'questions': len(traces), 'modes': 5,
         'trace_sha256': {m: sha256(ROOT/f'experiments/{m}.trace.jsonl')
              for m in ('no_context', 'single_rag', 'final', 'no_formula', 'no_neighbors')},
         'endpoint_sha256': sha256(ROOT/'endpoint_usage.jsonl')})
    print('Verified endpoint/trace accounting:', total)


if __name__ == '__main__':
    main()
