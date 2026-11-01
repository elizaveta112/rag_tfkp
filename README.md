# RAG по учебнику ТФКП

Локальная Qwen3.5-4B с поиском по учебнику Горяйнова—Половинкина (2025).
Из PDF восстанавливаются текст и формулы; корпус и индекс строятся автоматически.
Результаты пяти конфигураций — в `experiments.csv`, описание и разбор ошибок — в
`report.md`. По содержательной оценке выбран режим `no_formula`.

## Запуск в Colab

1. Открыть `notebooks/RAG_TFKP_Qwen35_4B.ipynb` в новой сессии Colab и выбрать GPU T4.
2. Выполнить первую ячейку и загрузить `RAG_TFKP_final.zip`. При запуске из клонированного репозитория разместить проект в `/content/rag_solution`.
3. Выполнять ячейки по порядку: установка, подготовка данных, тесты, пробный запрос, пять экспериментов и сборка результатов.

Модель работает в сессии Colab. Веса Q4_K_M занимают
около 2,74 GB. Сборка CUDA включена в ноутбук.

## Запуск через CLI

Python 3.11, NVIDIA GPU, CUDA toolkit и C++-компилятор. Команды выполняются из корня
проекта. Для T4 используется архитектура CUDA 75.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
CMAKE_ARGS='-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75' CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install -r requirements.txt
python prepare_all.py
python -m unittest discover -s tests -v
python run_local_experiments.py --gpu-layers -1 --threads 2
python verify_usage.py
python review_results.py --refresh
```

После расчёта проверить объяснения по `private_review/answers.md` и заполнить
`pending` в `semantic_reviews.json`, затем:

```bash
python finalize_results.py
python validate_submission.py
python package_submission.py
```

`Q_automatic` оценивает варианты и ссылки; `Q` дополнительно учитывает
содержательную проверку. Старая разметка применяется только к ответам с тем же
SHA256. Без полной проверки выбор режима помечается как предварительный.

Готовые результаты можно проверить командой `python validate_submission.py`
без GPU и установленных ML-пакетов. 

## Данные и конфигурация

- `data/tfkp_15_qa.md` — исходные QA; `dev_*` и `own_*` — вопросы, эталоны и ответы.
- `config.json` — выбранный режим; `measured_config.json` — исходная конфигурация опытов.
- `runtime.json` — версии модели, веса и параметры генерации; `selection.json` — результат выбора режима.

`prepare_all.py` скачивает закреплённый PDF, извлекает формулы, создаёт корпус,
вопросы, эталоны и индекс. Дроби, степени, индексы и сопряжение восстанавливаются
по расположению глифов и линий PDF. В индекс не попадают QA и эталоны.
PDF, корпус, индекс, веса, окружение и журналы других режимов в Git не включаются.
