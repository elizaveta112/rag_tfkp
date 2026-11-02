import sys as _sys
from pathlib import Path as _Path
for _directory in (_Path(__file__).resolve().parents[1], _Path(__file__).resolve().parents[1]/'src', _Path(__file__).resolve().parents[1]/'scripts'):
    _sys.path.insert(0, str(_directory))
#!/usr/bin/env python3
"""Distribute runnable source before measurements; never invent result files."""
import argparse
import json
import zipfile
from pathlib import Path
from common import ROOT, sha256
from package_submission import files

GENERATED = {'dev_answers.jsonl', 'dev_trace.jsonl', 'experiments.csv', 'corpus_audit.json',
    'retrieval_audit.json', 'selection.json', 'study_provenance.json', 'measured_config.json', 'server_resources.json', 'completion_manifest.json',
    'usage_reconciliation.json', 'final_metrics.json', 'semantic_reviews.json', 'error_reviews.json'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT.parent/'RAG_TFKP_solution.zip'))
    args = parser.parse_args()
    selected = [p for p in files() if p.name not in GENERATED]
    manifest = {'release': '2026-10-29-v4',
        'kind': 'source run kit; five fresh experiments must be executed by the user',
        'results_included': False, 'gpu_run_verified_here': False,
        'files_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in selected}}
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, Path('rag_solution')/path.relative_to(ROOT))
        archive.writestr('rag_solution/run_kit_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print('Created source run kit:', args.output, '; files:', len(selected))


if __name__ == '__main__':
    main()
