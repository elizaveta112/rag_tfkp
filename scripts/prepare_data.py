import sys as _sys
from pathlib import Path as _Path
for _directory in (_Path(__file__).resolve().parents[1], _Path(__file__).resolve().parents[1]/'src', _Path(__file__).resolve().parents[1]/'scripts'):
    _sys.path.insert(0, str(_directory))
from common import project_path
#!/usr/bin/env python3
"""Build the corpus independently of QA labels; derive dev input from supplied QA."""
import argparse
import urllib.request
from pathlib import Path
from common import ROOT, write_jsonl, dump, sha256
from pdf_corpus import extract_document
from tools.convert_qa import convert

BOOK_URL = 'https://new.mipt.ru/upload/mathematics/study/uchebniki/%D0%9B_%D0%A2%D0%A4%D0%9A%D0%9F_%D0%93%D0%BE%D1%80%D1%8F%D0%B9%D0%BD%D0%BE%D0%B2_%D0%9F%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD_2025.pdf'
BOOK_SHA256 = '629f4b668ff43abf2a6e621aefd0f50e165ce8e88c52c05769149e1267d033ec'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pdf', default=str(ROOT/'source/textbook.pdf'))
    p.add_argument('--qa', default=str(ROOT/'data/tfkp_15_qa.md'),
                   help='Source QA Markdown; generated dev JSONL is always rebuilt from this input')
    p.add_argument('--output-dir', default=str(ROOT))
    args = p.parse_args()
    pdf, output = Path(args.pdf), Path(args.output_dir)
    if not pdf.exists():
        pdf.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BOOK_URL, timeout=120) as r:
            pdf.write_bytes(r.read())
    if sha256(pdf) != BOOK_SHA256:
        raise ValueError('Textbook PDF differs from the pinned 2025 source; verify the document before extraction')
    blocks, audit = extract_document(pdf)
    audit['source_url'], audit['pdf_sha256'] = BOOK_URL, sha256(pdf)
    write_jsonl(project_path(output, 'corpus.jsonl'), blocks)
    dump(project_path(output, 'corpus_audit.json'), audit)
    questions, references = convert(Path(args.qa).read_text(encoding='utf-8'))
    write_jsonl(project_path(output, 'dev_questions.jsonl'), questions)
    write_jsonl(project_path(output, 'dev_references.jsonl'), references)
    print(f"Prepared {audit['pages']} pages, {len(blocks)} blocks; corpus contains no QA/gold.")


if __name__ == '__main__':
    main()
