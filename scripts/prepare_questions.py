import sys as _sys
from pathlib import Path as _Path
for _directory in (_Path(__file__).resolve().parents[1], _Path(__file__).resolve().parents[1]/'src', _Path(__file__).resolve().parents[1]/'scripts'):
    _sys.path.insert(0, str(_directory))
from common import project_path
#!/usr/bin/env python3
"""Rebuild question inputs and reference metadata from the provided QA source."""
import argparse
import subprocess
import sys
from pathlib import Path
from common import ROOT, write_jsonl
from tools.convert_qa import convert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa', default=str(ROOT/'data/tfkp_15_qa.md'))
    parser.add_argument('--dev-only', action='store_true', help='Do not regenerate the eight author questions')
    args = parser.parse_args()
    questions, references = convert(Path(args.qa).read_text(encoding='utf-8'))
    write_jsonl(project_path(ROOT, 'dev_questions.jsonl'), questions)
    write_jsonl(project_path(ROOT, 'dev_references.jsonl'), references)
    if not args.dev_only:
        subprocess.run([sys.executable, str(ROOT/'tools/make_own_questions.py')], check=True)
    print('Question JSONL rebuilt from source QA; run prepare_gold.py after rebuilding the PDF corpus.')


if __name__ == '__main__':
    main()
