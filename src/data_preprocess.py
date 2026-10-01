#файл с stateless препроцессингом — изменения которые необходимо сделать с датасетом в любом случае до fit/transfrom-а 
import pandas as pd
from config import config

def create_data():

    raw_data = pd.read_csv(config.path.raw_data)

    #извлекаем кол-во однофамильцев внутри группы из имени и дропаем имя
    raw_data["SameLastname"] = raw_data.groupby([raw_data["PassengerId"].str.split("_").str[0], raw_data["Name"].str.split().str[-1]])["Name"].transform("size") #ai
    raw_data.drop(["Name"], inplace=True, axis=1)

    #извлекаем кол-во людей в группе из айди и дропаем айди 
    raw_data["N_in_group"] = raw_data.groupby(raw_data["PassengerId"].str.split("_").str[0])["PassengerId"].transform("size")
    raw_data.drop(["PassengerId"], inplace=True, axis=1)

    #считаем соотношение числа людей в группе и однофамильцев — если оно близко к единице, вероятно люди путешествуют семьей
    raw_data["FamilyProbability"] = raw_data["SameLastname"] / raw_data["N_in_group"]

    #считаем сколько потратил человек всего, добавляем соответствующие колонки
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    raw_data["TotalSpend"] = raw_data[spending_cols].sum(axis=1) #ai
    raw_data["NoSpend"] = (raw_data["TotalSpend"] == 0)

    #две фичи — у кого нет трат и нет крео-сна и у кого нет трат и есть крео-сон
    raw_data["VeryPoor"] = raw_data["NoSpend"] & (raw_data["CryoSleep"] == True)
    raw_data["JustSleeping"] = raw_data["NoSpend"] & (raw_data["CryoSleep"] == False)

    #обрабатываем категорию "Cabin"
    raw_data[["deck", "num", "side"]] = raw_data["Cabin"].str.split("/", expand=True)
    raw_data.drop(["Cabin", "num"], inplace=True, axis=1)

    #приводим к одному формату категориальные признаки
    raw_data["VIP"] = raw_data["VIP"].fillna("Unknown").astype(str)
    raw_data["CryoSleep"] = raw_data["CryoSleep"].fillna("Unknown").astype(str)

    #заменяем пропуски в категориальных признаках по всему датасету категорий "неизвестно"
    cat_features = [col for col in raw_data.columns if raw_data[col].dtype == "str"]
    raw_data[cat_features] = raw_data[cat_features].fillna("Unknown")

    #сохраняем в новый файл
    raw_data.to_csv(config.path.preprocessed, index=False)

    #добавляем файл с one-hot векторами
    one_hoted_data = pd.get_dummies(raw_data, columns=cat_features)
    one_hoted_data.to_csv(config.path.prepcocessed_onehot, index=False)


#служебная функция для обработки результата
def result_data(results, model):
    result = {
        "model": model,
        "mean": round(results.mean()*100, 3),
        "std": round(results.std()*100, 3),
        "fold1": round(results[0]*100, 3),
        "fold2": round(results[1]*100, 3),
        "fold3": round(results[2]*100, 3),
        "fold4": round(results[3]*100, 3),
        "fold5": round(results[4]*100, 3)
    }

    return result