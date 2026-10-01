# Spaceship Titanic

Учебный проект для сравнения моделей классификации на данных соревнования Kaggle Spaceship Titanic. Программа подготавливает данные и оценивает accuracy моделей с помощью стратифицированной кросс-валидации на пяти фолдах.

Реализованы логистическая регрессия, KNN, дерево решений, случайный лес, XGBoost, LightGBM, CatBoost и нейронная сеть на PyTorch.

## Установка

Локальный запуск проверен на macOS с Python 3.14.4. Версии зависимостей зафиксированы в `requirements.txt`.

Из корня проекта выполните в Bash или Zsh:

```bash
bash setup.sh
source .venv/bin/activate
```

Скрипт создаёт виртуальное окружение `.venv` и устанавливает зависимости. Команда `python3` должна указывать на нужную версию Python.

## Запуск

Для обучения нужен файл `raw_dataset/train.csv`. Файлы `test.csv` и `sample_submission.csv` текущий сценарий оценки не использует; submission для Kaggle не создаётся.

Одна модель:

```bash
python src/main.py -m simpletree
```

Несколько моделей:

```bash
python src/main.py -m logreg knn catboost
```

Все модели, включая нейронную сеть:

```bash
python src/main.py -m all
```

Без флага `-m` также запускаются все модели. Доступные значения: `logreg`, `knn`, `simpletree`, `randforest`, `xgboost`, `lightgbm`, `catboost`, `nn`, `all`.

Справка:

```bash
python src/main.py --help
```

После разбора аргументов программа автоматически создаёт папки `preprocessed_data/` и `results/`. Предобработанные данные пересоздаются перед каждым запуском обучения. Гиперпараметры моделей и пути задаются в `src/config.py`.

## Результаты

После завершения работы выбранных моделей программа выводит таблицу в терминал и сохраняет её в `results/results.csv`:

- `model` — название модели;
- `mean` и `std` — средняя accuracy и стандартное отклонение по пяти фолдам;
- `fold1`–`fold5` — accuracy на каждом фолде.

Значения accuracy приведены в процентах. При повторном запуске результат той же модели заменяется, результаты остальных моделей сохраняются.

## Структура проекта

```text
SpaceShip Titanic/
├── src/
│   ├── main.py              # Точка входа и аргументы командной строки
│   ├── initialize.py        # Создание выходных каталогов
│   ├── config.py            # Пути и параметры моделей
│   ├── data_preprocess.py   # Подготовка данных и результатов
│   ├── transform.py         # Заполнение числовых пропусков
│   ├── models_list.py       # Базовые модели
│   ├── model_factory.py     # Обучение и кросс-валидация
│   └── dlmodel.py           # Архитектура нейронной сети
├── raw_dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── EDA.ipynb
├── preprocessed_data/      # Создаётся автоматически
├── results/                # Создаётся автоматически
├── requirements.txt
└── setup.sh
```

Виртуальное окружение, сгенерированные данные, результаты и логи CatBoost исключены из Git через `.gitignore`.
