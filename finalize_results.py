#!/usr/bin/env python3
"""Verify measured runs, uniformly rescore, then publish one complete mode."""
import csv
import json
import shutil
from common import ROOT, read_jsonl, write_jsonl, dump, sha256
from evaluate import evaluate
from review_results import matching_reviews
from run_experiments import MODES
from run_state import fingerprint

FIELDS = ('name', 'answer_accuracy', 'Q_automatic', 'Q', 'semantic_reviewed',
          'mean_input_tokens', 'mean_output_tokens', 'mean_llm_calls', 'mean_cost', 'median_latency_ms')


def main():
    measured = ROOT/'measured_config.json'
    if not measured.exists():
        raise ValueError('Missing measured_config.json: preserve the exact configuration used for inference')
    original = json.loads(measured.read_text())
    active = json.loads((ROOT/'config.json').read_text())
    if {k:v for k,v in original.items() if k != 'mode'} != {k:v for k,v in active.items() if k != 'mode'}:
        raise ValueError('Only default mode may change after measurement; inference settings were modified')
    reconciliation = json.loads((ROOT/'usage_reconciliation.json').read_text())
    if not reconciliation.get('verified') or reconciliation['endpoint_sha256'] != sha256(ROOT/'endpoint_usage.jsonl'):
        raise ValueError('Run verify_usage.py for the current independent endpoint journal')
    questions, gold = read_jsonl(ROOT/'dev_questions.jsonl'), read_jsonl(ROOT/'dev_gold.jsonl')
    corpus_ids = {b['block_id'] for b in read_jsonl(ROOT/'corpus.jsonl')}
    results, provenance = {}, {}
    for mode in MODES:
        answer, trace = ROOT/f'experiments/{mode}.answers.jsonl', ROOT/f'experiments/{mode}.trace.jsonl'
        if reconciliation['trace_sha256'][mode] != sha256(trace):
            raise ValueError('Measured trace changed: '+mode)
        state = json.loads((ROOT/f'experiments/{mode}.answers.jsonl.run.json').read_text())
        current = fingerprint(ROOT/'index', ROOT/'dev_questions.jsonl', measured, mode)
        if state != current:
            raise ValueError('Measured source, configuration, inputs or runtime changed: '+mode)
        annotations = matching_reviews(ROOT/'semantic_reviews.json', mode, read_jsonl(answer))
        automatic = evaluate(questions, gold, answer, trace, corpus_ids, mode == 'no_context')
        result = evaluate(questions, gold, answer, trace, corpus_ids, mode == 'no_context', semantic_reviews=annotations)
        if not result['accounting_verified']:
            raise ValueError('Unverified usage: '+mode)
        result.update(name=mode, Q_automatic=automatic['Q'])
        results[mode] = result
        dump(ROOT/f'experiments/{mode}.metrics.json', result)
        provenance[mode] = {'answers_sha256': sha256(answer), 'trace_sha256': sha256(trace),
                            'run_identity': state, 'metrics': {k:v for k,v in result.items() if k != 'records'}}
    complete = all(results[m]['semantic_reviewed'] == len(questions) for m in MODES if m != 'no_context')
    # No question-wise mixing. On a fresh run, unmatched old reviews cannot affect selection.
    key = 'Q' if complete else 'Q_automatic'
    selected = max((m for m in MODES if m != 'no_context'),
                   key=lambda m: (results[m][key], -results[m]['mean_cost']))
    active['mode'] = selected
    dump(ROOT/'config.json', active)
    shutil.copyfile(ROOT/f'experiments/{selected}.answers.jsonl', ROOT/'dev_answers.jsonl')
    write_jsonl(ROOT/'dev_trace.jsonl', [{k:v for k,v in t.items() if k != 'context_fragments'}
                for t in read_jsonl(ROOT/f'experiments/{selected}.trace.jsonl')])
    with (ROOT/'experiments.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, FIELDS, extrasaction='ignore')
        writer.writeheader(); writer.writerows(results.values())
    chosen = results[selected]
    selection = {'selected_mode': selected, 'basis': key, 'review_complete': complete,
        'reviewed_count': sum(results[m]['semantic_reviewed'] for m in MODES if m != 'no_context'),
        'policy': 'Maximum Q after complete uniform review; otherwise preliminary automatic Q. Lower mean cost breaks an exact tie. One whole mode, no answer edits.',
        'answer_accuracy': chosen['answer_accuracy'], 'Q_automatic': chosen['Q_automatic'], 'Q': chosen['Q'],
        'dev_answers_sha256': sha256(ROOT/'dev_answers.jsonl'), 'dev_trace_sha256': sha256(ROOT/'dev_trace.jsonl'),
        'measured_config_sha256': sha256(measured), 'config_sha256': sha256(ROOT/'config.json')}
    dump(ROOT/'selection.json', selection)
    manifest = json.loads((ROOT/'index/manifest.json').read_text())
    dump(ROOT/'study_provenance.json', {'kind': 'Original five measured runs, not new inference',
        'experiments': provenance, 'endpoint_reconciliation': reconciliation, 'index_manifest': manifest,
        'gold_sha256': sha256(ROOT/'dev_gold.jsonl'), 'review_sha256': sha256(ROOT/'semantic_reviews.json') if (ROOT/'semantic_reviews.json').exists() else None})
    c_ref = results['single_rag']['mean_cost']
    efficiency = min(1., c_ref/max(chosen['mean_cost'], 1))
    dump(ROOT/'final_metrics.json', {'selected_mode': selected, 'Q_automatic': chosen['Q_automatic'],
        'Q': chosen['Q'], 'E_local': efficiency, 'local_C_ref': c_ref,
        'local_score': chosen['Q']*(75+25*efficiency), 'reference': 'self-defined single_rag, not official baseline'})
    write_content_review(results, selected, complete)
    report = make_report(results, selected, complete, manifest, reconciliation)
    if len(report.split()) > 1500:
        raise ValueError('Report exceeds 1500 words')
    (ROOT/'report.md').write_text(report, encoding='utf-8')
    required = ('dev_questions.jsonl', 'dev_gold.jsonl', 'own_questions.jsonl', 'own_gold.jsonl',
        'dev_answers.jsonl', 'dev_trace.jsonl', 'experiments.csv', 'config.json', 'measured_config.json',
        'runtime.json', 'runtime_binding.json', 'selection.json', 'study_provenance.json',
        'semantic_reviews.json', 'report.md', 'final_metrics.json')
    dump(ROOT/'completion_manifest.json', {'status': 'computed', 'review_complete': complete,
        'selected_mode': selected, 'files_sha256': {n:sha256(ROOT/n) for n in required if (ROOT/n).exists()},
        'source_sha256': current['source_sha256'],
        'notes': 'Measured outputs are unchanged. Content review is assisted by ChatGPT, not an official grade.'})
    print(f'Selected {selected}; automatic Q={chosen["Q_automatic"]:.6f}; reviewed Q={chosen["Q"]:.6f}; '
          f'reviewed={selection["reviewed_count"]}/60; report={len(report.split())} words')


def write_content_review(results, selected, complete):
    path=ROOT/'semantic_reviews.json'
    document=json.loads(path.read_text()) if path.exists() else {}
    n=sum(results[m]['semantic_reviewed'] for m in MODES if m!='no_context')
    lines=['# Содержательная проверка RAG-ответов',
        f'\nПроверено {n} ответов. Выбран {selected}; проверка '+('завершена.' if complete else 'неполная, выбор предварительный.'),
        '\nЭто разметка с помощью ChatGPT, не официальная независимая оценка. Исходные генерации не исправлялись. Разметка не используется в inference.',
        '\n## Критерий', document.get('rubric', 'Неверные варианты, технический отказ или существенное ложное математическое утверждение обнуляют вклад. Ошибки нормировки и подмена достаточности необходимостью существенны; неполная ссылка учитывается citation F1.'),
        '\n## Сопоставление', '| Режим | Accuracy ID | Q auto | Q review | Проверено |', '|---|---:|---:|---:|---:|']
    for m in MODES:
        if m=='no_context': continue
        x=results[m]
        lines.append(f'| {m} | {x["answer_accuracy"]:.6f} | {x["Q_automatic"]:.6f} | {x["Q"]:.6f} | {x["semantic_reviewed"]} |')
    lines+=['\nПограничные формулировки не скрыты: «нужна односвязность» как необходимое условие, утрата непрерывности в исключительной точке и смешение нормированного/исходного интегралов считаются существенными ошибками. Более мягкая оценка может изменить ранжирование; вывод относится к указанному строгому критерию. При несогласии изменяйте вердикт с объяснением, сохраняя исходные ответы и логи.',
        '\n## Все решения', '| Режим | Вопрос | Вердикт | Основание |', '|---|---|---|---|']
    for mode in MODES:
        if mode=='no_context': continue
        answers=read_jsonl(ROOT/f'experiments/{mode}.answers.jsonl')
        matched=matching_reviews(path, mode, answers)
        for answer in answers:
            item=matched.get(answer['id'], {'verdict':'pending','reason':'Нужна содержательная проверка текущего ответа.'})
            reason=item['reason'].replace('|','∣').replace('\n',' ')
            lines.append(f'| {mode} | {answer["id"]} | {item["verdict"]} | {reason} |')
    lines+=['\nКаждый вердикт привязан к SHA256 полного ответа в semantic_reviews.json. Изменившиеся генерации не наследуют старую оценку. review_results.py --refresh сохраняет только совпавшие записи; новые ответы становятся pending. После проверки finalize_results.py пересчитывает все режимы и выбирает один целиком. Для пересчёта нужны приватные логи и восстановленные corpus/index; большие логи в сдачу не входят.']
    directory=ROOT/'private_review'
    directory.mkdir(exist_ok=True)
    (directory/'content_review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def make_report(results, selected, complete, manifest, usage):
    runtime = json.loads((ROOT/'runtime.json').read_text())
    resource = json.loads((ROOT/'server_resources.json').read_text())
    rows = ['| Режим | Accuracy | Q auto | Q review | In | Out | Calls | C | Median ms |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode, r in results.items():
        auto = '—' if r['Q_automatic'] is None else f'{r["Q_automatic"]:.4f}'
        reviewed = '—' if r['Q'] is None else f'{r["Q"]:.4f}'
        rows.append(f'| {mode} | {r["answer_accuracy"]:.4f} | {auto} | {reviewed} | '
                    f'{r["mean_input_tokens"]:.1f} | {r["mean_output_tokens"]:.1f} | '
                    f'{r["mean_llm_calls"]:.2f} | {r["mean_cost"]:.1f} | {r["median_latency_ms"]:.0f} |')
    qmap = {q['id']:q for q in read_jsonl(ROOT/'dev_questions.jsonl')}
    amap = {a['id']:a for a in read_jsonl(ROOT/'dev_answers.jsonl')}
    tmap = {t['id']:t for t in read_jsonl(ROOT/f'experiments/{selected}.trace.jsonl')}
    # These three current-run failures span sign extraction, reasoning and refusal.
    preferred = ('qa_002', 'qa_012', 'qa_013')
    failures = [r for r in results[selected]['records'] if not r['correct'] or r['semantic_error']]
    failures.sort(key=lambda r: (preferred.index(r['id']) if r['id'] in preferred else 99, r['id']))
    errors=[]
    gold = {g['id']:g for g in read_jsonl(ROOT/'dev_gold.jsonl')}
    for r in failures[:3]:
        qid=r['id']; a=amap[qid]; t=tmap[qid]
        reason=r['semantic_error'] or r['error'] or gold[qid]['rationale']
        focus = {'qa_002':'p0013_b003', 'qa_012':'p0038_b008', 'qa_013':'p0037_b003'}.get(qid)
        fragment = next((f for f in t['context_fragments'] if f['block_id']==focus), t['context_fragments'][0] if t['context_fragments'] else None)
        excerpt = (' '.join(fragment['text'].split())[:170] if fragment else 'Контекст пуст')
        errors.append(f'**{qid}.** {qmap[qid]["question"]} '
            f'Ответ: {a["answer_ids"]}, отказ={a["abstain"]}. '
            f'Показаны: {", ".join(t["context_block_ids"])}. '
            f'Ссылки ответа: {", ".join(a["citations"]) or "нет"}. '
            f'Фрагмент {fragment["block_id"] if fragment else "нет"}: «{excerpt}…». '
            f'Причина: {reason}')
    full=results['final']; nof=results['no_formula']; non=results['no_neighbors']
    review_count=sum(results[m]['semantic_reviewed'] for m in MODES if m!='no_context')
    selected_r=results[selected]
    audit_status='завершена' if complete else 'не завершена; выбор предварительный, нужна новая проверка изменившихся ответов'
    return f'''# RAG по учебнику ТФКП

Смагина Елизавета Марковна, М05-512д.

## Данные и извлечение формул

Корпус построен из учебника Горяйнова—Половинкина (2025): 147 страниц,
{manifest['blocks']} блоков и {manifest['chunks']} чанков. Исходные 15 QA используются
для dev-набора: 5 вопросов на условия, 4 на сопоставление, 3 на применение и
3 на недостаточность данных. Подготовлены 8 авторских вопросов, по два каждой
категории. QA и эталоны в корпус не включены.

Извлечение реализовано через PyMuPDF. Для Type3-шрифтов восстанавливается
кириллица; по координатам глифов и векторным линиям определяются дроби, степени,
индексы и сопряжение. В тексте сохраняются знаки и факториалы:
f_zbar=(f_x+i·f_y)/2, f_z=(f_x−i·f_y)/2, c_n=f^(n)(0)/n!.
Поисковая нормализация сопоставляет Unicode и LaTeX; модель получает текст
источника. Чанки около 1400 символов сохраняют страницу и block_id, размеченные
формулы не разрезаются. Извлечение эвристическое: сложные радикалы и матрицы
могут восстанавливаться неточно. Ключевые формулы сверены с изображениями PDF.

## Поиск и модель

Полная конфигурация final объединяет BM25 по словам, символьным триграммам и
структурным признакам формул через weighted RRF. Запросы строятся из вопроса
и его вариантов; словарь терминов связывает обозначения с названиями тем.
В контекст добавляются соседи той же секции. single_rag использует BM25 по словам;
no_context отвечает без учебника. no_formula отключает структурный канал формул,
но сохраняет сами формулы в тексте. no_neighbors отключает расширение соседями.

Генератор — локальная Qwen3.5-4B Q4_K_M, llama-cpp-python 0.3.36.
Общие параметры: temperature={runtime['generation']['temperature']},
top_p={runtime['generation']['top_p']}, top_k={runtime['generation']['top_k']},
seed={runtime['generation']['seed']}, max_tokens=768, thinking=false, n_ctx=8192.
Qwen проверяет каждый вариант и возвращает JSON; answer_ids собираются из её
вердиктов. Допускается один повтор при ошибке формата. Все вызовы входят в расход.
Embedding-модели, reranker и дообучение не применялись.

## Эксперименты

Accuracy — точность набора ID и отказа. Q auto учитывает также F1 ссылок по лучшему
допустимому набору и поровну усредняет четыре категории. Q review обнуляет вклад
при существенной математической ошибке в объяснении. Проверено {review_count}/60
RAG-ответов; проверка {audit_status}. Разметка с причинами и SHA256 ответов
сохранена в semantic_reviews.json. Неверная нормировка и подмена достаточного
условия необходимым считаются существенными; более мягкий критерий может изменить
ранжирование. Это локальная оценка открытого dev-набора.

{chr(10).join(rows)}

Итоговый режим — **{selected}**, выбран по максимальному
{'Q review после полной проверки' if complete else 'предварительному Q auto'};
при равенстве учитывается меньшая средняя стоимость. Ответы взяты из одного
режима целиком. Q auto={selected_r['Q_automatic']:.4f}, Q review={selected_r['Q']:.4f}.
Точность вариантов — {round(selected_r['answer_accuracy']*len(qmap))}/{len(qmap)},
у no_context — {round(results['no_context']['answer_accuracy']*len(qmap))}/{len(qmap)}.
Устойчивое улучшение относительно ответа без контекста не установлено; выбор на
15 dev-вопросах не позволяет сделать вывод о качестве на новых данных.

Для полного кандидата отключение структурного канала меняет Q review на
{nof['Q']-full['Q']:+.4f}, отключение соседей — на {non['Q']-full['Q']:+.4f}.
Комбинация без обоих компонентов не измерялась, поэтому их взаимодействие
и влияние соседей на выбранную конфигурацию не установлены.

## Три ошибки выбранной системы

{chr(10).join(e+chr(10) for e in errors)}

## Ресурсы и воспроизводимость

Опыты выполнены в Colab: Tesla T4, Intel Xeon, Python 3.11.17, Linux,
{resource['threads']} CPU-потока, gpu_layers=-1. Загрузка модели —
{resource['load_seconds']:.2f} с; веса — 2,74 GB. Построение индекса —
{manifest['build_seconds']:.2f} с, пиковая RSS — {manifest['peak_rss_bytes']/1024**2:.1f} MiB,
размер индекса с токенизатором — {manifest['index_bytes']/1024**2:.2f} MiB.
С независимым журналом endpoint сверены {usage['llm_calls']} вызов,
{usage['input_tokens']} входных и {usage['output_tokens']} выходной токен.
Стоимость C=In+4·Out+256·Calls. Лимиты соблюдены; повторы включены.
В final_metrics.json эффективность рассчитана относительно собственного single_rag,
а не официального baseline.

Полный запуск описан в README.md. Код пересоздаёт данные из PDF и исходных QA;
версии модели и SHA256 закреплены в runtime.json. Файлы происхождения результатов
сохраняют исходные хеши измерений. Новые генерации могут отличаться даже с тем же
seed и требуют новой содержательной проверки. Ограничение длины внутренней причины
иногда приводит к техническому отказу; этот недостаток измеренной версии сохранён.

При подготовке кода, отчёта и содержательной разметки использовался ChatGPT.
Экспериментальные ответы генерировала локальная Qwen3.5-4B.
'''


if __name__ == '__main__':
    main()
