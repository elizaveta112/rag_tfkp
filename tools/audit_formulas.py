import sys as _sys
from pathlib import Path as _Path
for _directory in (_Path(__file__).resolve().parents[1], _Path(__file__).resolve().parents[1]/'src', _Path(__file__).resolve().parents[1]/'scripts'):
    _sys.path.insert(0, str(_directory))
#!/usr/bin/env python3
"""Report suspicious supplied text without rewriting mathematical evidence."""
import argparse
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_jsonl, dump
from formulas import MATH


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    issues, n_formulas = [], 0
    for b in read_jsonl(args.corpus):
        text = b['text']
        n_formulas += len(list(MATH.finditer(text)))
        reasons = []
        if '\ufffd' in text:
            reasons.append('replacement_character_U+FFFD')
        outside = MATH.sub('', text)
        if re.search(r'(?<!\\)\$', outside):
            reasons.append('unpaired_dollar_delimiter')
        if outside.count(r'\[') != outside.count(r'\]'):
            reasons.append('unpaired_display_delimiter')
        for formula in MATH.finditer(text):
            f = re.sub(r'\\[{}]', '', formula.group())
            if f.count('{') != f.count('}'):
                reasons.append('unbalanced_braces_in_formula')
        if reasons:
            issues.append({'block_id': b['block_id'], 'page': b['page'],
                           'flags': sorted(set(reasons)), 'text': text})
    dump(args.output, {'formula_spans': n_formulas, 'suspicious_blocks': issues,
                       'note': 'Flags are diagnostics, not proof of corruption. No auto-correction.'})


if __name__ == '__main__':
    main()
