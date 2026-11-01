#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_jsonl, load_config, write_jsonl
from retrieval import Retriever


def main():
    p = argparse.ArgumentParser()
    for arg in ('index', 'questions', 'output', 'config'):
        p.add_argument('--'+arg, required=True)
    p.add_argument('--mode', default='final', choices=['single_rag', 'final', 'no_formula', 'no_neighbors'])
    args = p.parse_args()
    retriever, config = Retriever(args.index), load_config(args.config)
    rows = []
    for q in read_jsonl(args.questions):
        chunks, _ = retriever.search(q, args.mode, config['retrieval'])
        rows.append({'id': q['id'], 'question': q['question'], 'candidates': chunks})
    write_jsonl(args.output, rows)


if __name__ == '__main__':
    main()
