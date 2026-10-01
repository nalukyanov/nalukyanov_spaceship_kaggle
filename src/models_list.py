from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from config import config

#в этом файле инициализируются сразу все модели кроме катбуста чтобы передать их как аргумент в функцию в цикле

params = config.params
base_models = {
    "logreg": LogisticRegression(**params.logreg),
    "knn": KNeighborsClassifier(**params.knn),
    "simpletree": DecisionTreeClassifier(**params.simpletree),
    "randforest": RandomForestClassifier(**params.randforest),
    "xgboost": XGBClassifier(**params.xgboost),
    "lightgbm": LGBMClassifier(**params.lightgbm),
}