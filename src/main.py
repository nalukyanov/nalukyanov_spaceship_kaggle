import argparse
import pandas as pd
from model_factory import basemlmodels, catboostmodel, dlmodel
from sklearn.preprocessing import StandardScaler
from models_list import base_models
from data_preprocess import create_data, result_data
from config import config
from initialize import initialize_project

def main():

    #список всех моделей, он потом еще понадобится
    models = config.models
    base_model_names = list(base_models.keys())

    #добавляем возможность запуска через командную строку
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        choices=models,
        nargs="+"
    )

    #считываем все модели, переданные в командной строке
    #если ни один флаг не передан, то по умолчанию запускается все
    models_to_run = parser.parse_args().m
    if models_to_run is None or models_to_run[0] == "all":
        models_to_run = models

    initialize_project()

    # перед каждым запуском создаем новый датасет чтобы не прогнать тесты на старом после правок
    create_data()

    #тут будет список словарей
    results = []

    #модели для которых требуется масштабирование числовых признаков
    scalable_models = ['logreg', 'knn']

    #запускаем цикл, в котором обрабатываем все переданные модели
    #результат передается в вспомогательную функцию, которая возвращает словарь
    #словарь записывается в список
    for model in models_to_run:
        if model in base_model_names:
            results.append((
                basemlmodels(
                base_models[model], 
                scaler_obj=StandardScaler() if model in scalable_models else None
                ), model))
        elif model == "catboost":
            results.append((catboostmodel(), "catboost"))
        elif model == "nn":
            results.append((dlmodel(), "neural network"))

    results = [result_data(result[0], result[1]) for result in results]

    #список словарей преборазуется в датафрейм для удобства отображения
    new_results = pd.DataFrame(results)

    #выводим результат работы по переданным флагам
    print(new_results)

    #сохраняем или перезаписываем результаты
    try: #ai
        old_results = pd.read_csv(config.path.result)

        result = pd.concat([old_results, new_results])
        result = result.drop_duplicates(subset="model", keep="last")

    except FileNotFoundError:
        result = new_results

    result.to_csv(config.path.result, index=False)

if __name__ == "__main__":
    main()
