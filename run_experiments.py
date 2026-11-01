#!/usr/bin/env python3
"""Five complete runs with identical questions and immutable generation settings."""
import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from common import ROOT, read_jsonl, dump, sha256
from evaluate import evaluate, cost

MODES = ('no_context', 'single_rag', 'final', 'no_formula', 'no_neighbors')
FIELDS = ('name', 'answer_accuracy', 'Q', 'mean_input_tokens', 'mean_output_tokens',
          'mean_llm_calls', 'mean_cost', 'median_latency_ms')


def main():
    p = argparse.ArgumentParser()
    for arg in ('index', 'questions', 'gold', 'corpus', 'config'):
        p.add_argument('--'+arg, required=True)
    p.add_argument('--output-dir', default=str(ROOT/'experiments'))
    p.add_argument('--reference-trace')
    p.add_argument('--resume', action='store_true', help='Continue matching interrupted development run files')
    args = p.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        previous = [p for mode in MODES for p in out.glob(mode+'.*') if p.is_file()]
        if previous:
            archive = out/'archive'/str(time.time_ns())
            archive.mkdir(parents=True)
            for path in previous:
                path.replace(archive/path.name)
    # Preserve exact bytes: prediction fingerprints bind to this configuration.
    snapshot = ROOT/'measured_config.json'
    if args.resume and snapshot.exists() and snapshot.read_bytes() != Path(args.config).read_bytes():
        raise ValueError('Resume config differs from the measured snapshot')
    if Path(args.config).resolve() != snapshot.resolve():
        shutil.copyfile(args.config, snapshot)
    questions, gold = read_jsonl(args.questions), read_jsonl(args.gold)
    blocks = {b['block_id'] for b in read_jsonl(args.corpus)}
    if any(g.get('gold_status', 'reviewed') != 'reviewed' for g in gold):
        p.error('Review original block IDs first; draft page references are not citation gold')
    reference = read_jsonl(args.reference_trace) if args.reference_trace else None
    c_ref = statistics.mean(cost(t) for t in reference) if reference else None
    results = []
    for number, mode in enumerate(MODES, 1):
        print(f'Experiment {number}/{len(MODES)}: {mode}', flush=True)
        answer, trace = out/f'{mode}.answers.jsonl', out/f'{mode}.trace.jsonl'
        command = [sys.executable, str(ROOT/'predict.py'), '--index', str(Path(args.index).resolve()),
            '--questions', str(Path(args.questions).resolve()), '--config', str(Path(args.config).resolve()),
            '--output', str(answer), '--trace', str(trace), '--mode', mode]
        if args.resume:
            command.append('--resume')
        subprocess.run(command, check=True)
        result = evaluate(questions, gold, answer, trace, blocks, mode == 'no_context', c_ref)
        result['name'] = mode
        results.append(result)
        dump(out/f'{mode}.metrics.json', result)
        if not result['accounting_verified']:
            raise RuntimeError(f'{mode}: endpoint usage failed verification; inspect trace/server.log before continuing')
    with (ROOT/'experiments.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    selected = json.loads(Path(args.config).read_text())['mode']
    if selected not in MODES or selected == 'no_context':
        raise ValueError('Default submission mode must be a measured RAG mode')
    shutil.copyfile(out/f'{selected}.answers.jsonl', ROOT/'dev_answers.jsonl')
    # Publish required accounting and IDs; large source excerpts stay in private diagnostics.
    from common import write_jsonl
    write_jsonl(ROOT/'dev_trace.jsonl', [{k: v for k, v in t.items() if k != 'context_fragments'}
                                        for t in read_jsonl(out/f'{selected}.trace.jsonl')])
    dump(out/'run_manifest.json', {'question_sha256': sha256(args.questions),
        'gold_sha256': sha256(args.gold), 'config_sha256': sha256(args.config),
        'modes': MODES, 'note': 'Generation parameters are shared via one runtime binding; explicit continuation does not reuse answers across new runs'})
    lines = ['# Измеренные результаты\n', '| Система | Accuracy | Q | C |', '|---|---:|---:|---:|']
    for r in results:
        q_value = '' if r['Q'] is None else f"{r['Q']:.4f}"
        lines.append(f"| {r['name']} | {r['answer_accuracy']:.4f} | {q_value} | {r['mean_cost']:.1f} |")
    final = next(r for r in results if r['name'] == 'final')
    for name in ('no_formula', 'no_neighbors'):
        ablation = next(r for r in results if r['name'] == name)
        delta = final['Q']-ablation['Q']
        lines.append(f'\nОтключение {name}: Q(final) − Q(ablation) = {delta:+.4f}. '
                     'Это наблюдение на dev; проверьте изменившиеся ответы и цитаты перед выводом о причине.')
    lines.append('\n## Ошибки для содержательного разбора\n')
    final_answers = {r['id']: r for r in read_jsonl(out/'final.answers.jsonl')}
    final_traces = {r['id']: r for r in read_jsonl(out/'final.trace.jsonl')}
    q_map = {q['id']: q for q in questions}
    errors = [r for r in final['records'] if r['score'] < 1]
    for r in errors[:3]:
        lines += [f"\n### {r['id']}\n", q_map[r['id']]['question'],
                  '\n```json\n'+json.dumps({'prediction': final_answers[r['id']],
                      'score': r, 'found_fragments': final_traces[r['id']]['context_fragments']},
                      ensure_ascii=False, indent=2)+'\n```',
                  '\nПричина: требуется содержательный разбор приведённых фрагментов.']
    if len(errors) < 3:
        lines.append(f'\nНайдено {len(errors)} ошибок; недостающие ошибки не выдумываются.')
    (out/'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    from checkpoint_run import backup_if_enabled
    backup_if_enabled()
    print('Experiments finished. Merge measured results and reviewed error causes into report.md.')


if __name__ == '__main__':
    main()
