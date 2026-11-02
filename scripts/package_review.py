import sys as _sys
from pathlib import Path as _Path
for _directory in (_Path(__file__).resolve().parents[1], _Path(__file__).resolve().parents[1]/'src', _Path(__file__).resolve().parents[1]/'scripts'):
    _sys.path.insert(0, str(_directory))
from common import project_path
#!/usr/bin/env python3
"""Optional small private bundle for reviewing all five sets of rationales."""
import zipfile
import json
from common import ROOT, read_jsonl, sha256
from run_experiments import MODES


def main():
    names = ['dev_questions.jsonl', 'dev_gold.jsonl', 'dev_answers.jsonl', 'dev_trace.jsonl',
             'experiments.csv', 'report.md', 'config.json', 'runtime.json', 'server_resources.json',
             'usage_reconciliation.json', 'index/manifest.json', 'corpus_audit.json',
             'selection.json', 'measured_config.json', 'study_provenance.json']
    names += [f'experiments/{mode}.answers.jsonl' for mode in MODES]
    if (project_path(ROOT, 'semantic_reviews.json')).exists():
        names.append('semantic_reviews.json')
    for name in ('endpoint_usage.jsonl', 'request_attempts.jsonl'):
        if (project_path(ROOT, name)).exists():
            names.append(name)
    missing = [n for n in names if not (project_path(ROOT, n)).exists()]
    if missing:
        raise ValueError('Complete the experiments and report first: '+', '.join(missing))
    with zipfile.ZipFile(ROOT/'results_for_review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(project_path(ROOT, name), name)
        fragments, hashes = {}, {}
        for mode in MODES:
            name = f'experiments/{mode}.trace.jsonl'
            hashes[name] = sha256(project_path(ROOT, name))
            traces = read_jsonl(project_path(ROOT, name))
            compact = []
            for trace in traces:
                for fragment in trace.get('context_fragments', []):
                    key = json.dumps(fragment, ensure_ascii=False, sort_keys=True)
                    fragments.setdefault(key, {**fragment, 'shown_in': []})['shown_in'].append(
                        {'mode': mode, 'id': trace['id']})
                compact.append({k: v for k, v in trace.items() if k != 'context_fragments'})
            archive.writestr(name, ''.join(json.dumps(t, ensure_ascii=False)+'\n' for t in compact))
            archive.write(ROOT/f'experiments/{mode}.metrics.json', f'experiments/{mode}.metrics.json')
        archive.writestr('shown_fragments.jsonl', ''.join(
            json.dumps(f, ensure_ascii=False)+'\n' for f in fragments.values()))
        archive.writestr('original_trace_sha256.json', json.dumps(hashes, indent=2))
    print('Created results_for_review.zip for private review; this is not the GitHub upload bundle.')


if __name__ == '__main__':
    main()
