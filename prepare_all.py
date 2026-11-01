#!/usr/bin/env python3
"""Reproduce data, runtime assets and index before inference; no LLM calls."""
import argparse
import subprocess
import sys
from common import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', help='Optional verified local textbook PDF instead of downloading it')
    parser.add_argument('--skip-model', action='store_true', help='Only rebuild data and gold; do not download weights or build index')
    args = parser.parse_args()
    commands = [
        ['prepare_data.py']+(['--pdf', args.pdf] if args.pdf else []),
        ['prepare_questions.py'], ['prepare_gold.py']]
    if not args.skip_model:
        commands += [['tools/prepare_local.py'], ['build_index.py', '--corpus', str(ROOT/'corpus.jsonl'),
            '--output', str(ROOT/'index'), '--config', str(ROOT/'config.json')]]
    for number, command in enumerate(commands, 1):
        print(f'Preparation {number}/{len(commands)}: {command[0]}', flush=True)
        subprocess.run([sys.executable, str(ROOT/command[0]), *command[1:]], check=True)


if __name__ == '__main__':
    main()
