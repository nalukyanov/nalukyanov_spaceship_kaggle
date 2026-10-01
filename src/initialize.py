from pathlib import Path

from config import config


def initialize_project():
    """Создать каталоги для сгенерированных данных и результатов."""
    for output_path in (
        config.path.preprocessed,
        config.path.prepcocessed_onehot,
        config.path.result,
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
