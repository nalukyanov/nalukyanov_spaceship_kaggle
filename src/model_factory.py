import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from transform import MyTransformer
from config import config
from catboost import CatBoostClassifier
from dlmodel import DLModel
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


#базовая функция для моделей из склерна и бустингов помимо катбуста
#скейлер передается не для всех моделей
def basemlmodels(model_obj: object, scaler_obj: object):

    df = pd.read_csv(config.path.prepcocessed_onehot)
    X = df.drop(columns=["Transported"])
    y = df["Transported"]

    #кастомный импьютер, который считает среднее по трейну и заполняет пропуски на трейне и тесте
    my_imputer = MyTransformer(cols = config.columns.numerical)

    #выбираем скейлер: там где он нужен, надо чтобы он обрабатывал только числовые колонки
    if scaler_obj is not None:
        scaler = ColumnTransformer(
            transformers=[("num", scaler_obj, list(config.columns.numerical))],
            remainder="passthrough"
        )
    else:
        scaler = None

    pipeline = Pipeline([
        ("imputer", my_imputer),
        ("scaler", scaler),
        ("model",  model_obj)
    ])

    cv = StratifiedKFold(
        n_splits= 5,
        shuffle=True,
        random_state=13,
    )

    scores = cross_val_score(
        estimator=pipeline,
        X=X, y=y,
        cv=cv,
        scoring="accuracy"
    )

    return scores


#отедельная функция для катбуста, т.к. она требует немного другого препроцессинга
def catboostmodel():

    df = pd.read_csv(config.path.preprocessed)
    X = df.drop(columns=["Transported"])
    y = df["Transported"]

    #кастомный импьютер, который считает среднее по трейну и заполняет пропуски на трейне и тесте
    my_imputer = MyTransformer(cols = config.columns.numerical)
    pipeline = Pipeline([
        ("imputer", my_imputer),
        ("model",  CatBoostClassifier(**config.params.catboost))
    ])

    cv = StratifiedKFold(
        n_splits= 5,
        shuffle=True,
        random_state=13,
    )

    scores = cross_val_score(
        estimator=pipeline,
        X=X, y=y,
        cv=cv,
        scoring="accuracy",
        params={"model__cat_features": config.columns.categorial} #ai
    )

    return scores

#функция для ДЛ модели
def dlmodel():
    torch.manual_seed(13)
    results = []

    df = pd.read_csv(config.path.prepcocessed_onehot)
    X = df.drop(columns=["Transported"])
    y = df["Transported"]

    #тут нам нужны будут только индексы для сплита
    cv = StratifiedKFold(
        n_splits= 5,
        shuffle=True,
        random_state=13,
    )

    #цикл кросс-валидации
    for train_idx, val_idx in cv.split(X, y):

        #тест-трейн сплит на каждом фолде
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_val = y.iloc[val_idx].copy()

        #таргет был формата (n_samples, ) приводим к (n_samples, 1)
        y_train = y_train.to_numpy().reshape(y_train.shape[0], 1)
        y_val = y_val.to_numpy().reshape(y_val.shape[0], 1)


        #для этого фолда считаем пропуски по трейну и заполняем на трейне и валидации
        my_imputer = MyTransformer(cols = config.columns.numerical)
        my_imputer.fit(X_train)
        X_train = my_imputer.transform(X_train)
        X_val = my_imputer.transform(X_val)

        #скейлер – необходим для MLP
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        #приводим к формату, с которым работает пайторч
        X_train = torch.from_numpy(X_train.astype("float32"))
        X_val = torch.from_numpy(X_val.astype("float32"))
        y_train = torch.from_numpy(y_train.astype("float32"))
        y_val = torch.from_numpy(y_val.astype("float32"))

        #создаем лоудеры для работы с мини-батчами
        train_dataset = TensorDataset(X_train, y_train) #ai
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.params.nn.batch_size,
            shuffle=True
        )

        #по сути все выше - предобработка данных, тут начинаем обучать сеть
        model = DLModel(X_train.shape[1])
        loss_func = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
                lr=config.params.nn.learning_rate,
                weight_decay=config.params.nn.weight_decay,
        )

        #инициализируем параметры для early stoping-а
        best_loss = float("inf")
        patience = config.params.nn.patience
        patience_count = 0

        #цикл обучения самой сети
        for epoch in range(config.params.nn.epochs):

            model.train()

            #обучение по мини-батчу
            for X_batch, y_batch in train_loader:

                optimizer.zero_grad()
                pred = model(X_batch)
                loss = loss_func(pred, y_batch)
                loss.backward()
                optimizer.step()

            # if epoch % 100 == 0:
            #     print(f"loss at epoch {epoch}: {loss}")

            #считаем валидационый лосс
            model.eval()
            with torch.no_grad():
                pred = model(X_val)
            val_loss = loss_func(pred, y_val).item()

            #реализуем early stopping
            #если валидационный лосс не улучшается N эпох, прекращаем обучение
            if val_loss > best_loss:
                patience_count += 1
            else:
                best_loss = val_loss
                patience_count = 0
                best_state = copy.deepcopy(model.state_dict())
            if patience_count >= patience:
                break

        #загружаем состояние модели, которое было на лучшей эпохе
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = model(X_val)

        #получаем предсказание и считаем целевую метрику
        probs = torch.sigmoid(pred)
        preds = (probs >= 0.5).float()
        accuracy = (preds == y_val).float().mean()
        results.append(accuracy.item())

    return np.array(results)