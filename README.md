# Ma_Siggil

Кратко: бинарная сегментация дорог на спутниковых изображениях.

## Что нужно

- Python 3.8+
- `requirements.txt`
- (опционально) `pyproject.toml` + `uv.lock` для фиксированной среды

## Установка зависимостей

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Опционально через uv:

```bash
uv sync --frozen
```

## Датасет

Ссылка: http://www.cs.toronto.edu/~vmnih/data/mass_roads/

Ожидаемая структура:

```text
src/data/tiff/
├── train/
├── train_labels/
├── val/
├── val_labels/
├── test/
└── test_labels/
```

## Запуск обучения

```bash
python scripts/train.py
```

## Запуск валидации

```bash
python scripts/val.py --checkpoint src/results/checkpoints/model_epoch_XXX_dice_0.XXXX.pt
```

## Как посмотреть метрики и финальный скор

`scripts/val.py` печатает в консоль:
`Loss`, `Dice`, `IoU`, `Recall`, `Precision`, `F1`, `Specificity`, `Accuracy`.

Результаты валидации сохраняются в JSON рядом с checkpoint.

## Артефакты, которые сохраняются

После `python scripts/train.py`:

- Веса лучшей модели:
  - `src/results/checkpoints/best_model.pt`
  - `src/results/checkpoints/best_model_weights.pth`
- История метрик по эпохам:
  - `src/results/logs/history_YYYYMMDD_HHMMSS.csv`
  - `src/results/logs/history_YYYYMMDD_HHMMSS.json`
- Конфигурация запуска:
  - `src/results/logs/run_config_YYYYMMDD_HHMMSS.json`
- Графики обучения (лосс + метрики):
  - `src/results/reports/training_curves_YYYYMMDD_HHMMSS.png`

После `python scripts/val.py --checkpoint ...`:

- Метрики выбранного checkpoint:
  - `src/results/checkpoints/validation_results_<checkpoint_name>.json`
