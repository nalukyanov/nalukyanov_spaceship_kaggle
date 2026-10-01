from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent #ai

conf = {
    "models": ['logreg', 'knn', 'simpletree', 'randforest', 'xgboost', 'lightgbm', 'catboost', 'nn', 'all'],
    
    "path": {
        "raw_data": str(PROJECT_ROOT / "raw_dataset" / "train.csv"), #ai
        "preprocessed": str(PROJECT_ROOT / "preprocessed_data" / "preprocessed_data.csv"), #ai
        "prepcocessed_onehot": str(PROJECT_ROOT / "preprocessed_data" / "one_hoted_data.csv"), #ai
        "result": str(PROJECT_ROOT / "results" / "results.csv")
    },
    "columns": {
        "categorial": ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'deck', 'side', "NoSpend", "VeryPoor", "JustSleeping"],
        "numerical": ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', "SameLastname", "N_in_group", "TotalSpend", "FamilyProbability"]
    },
    "params": {
        "logreg": {
           "C": 1.0,
            "l1_ratio": 0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "class_weight": None,
            "random_state": 13
        },
        "knn": {
                "n_neighbors": 15,
                "weights": "distance",
                "algorithm": "auto",
                "leaf_size": 30,
                "p": 2,
                "metric": "minkowski",
                "n_jobs": -1
        },
        "simpletree" : {
                "criterion": "gini",
                "max_depth": 6,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
                "max_features": None,
                "class_weight": None,
                "random_state": 13
        },
        "randforest": {
                "n_estimators": 700,
                "criterion": "gini",
                "max_depth": 10,
                "min_samples_split": 8,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
                "class_weight": None,
                "random_state": 13,
                "n_jobs": -1
        },
        "xgboost": {
                "n_estimators": 700,
                "learning_rate": 0.03,
                "max_depth": 5,
                "min_child_weight": 3,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "gamma": 0.05,
                "reg_alpha": 0.05,
                "reg_lambda": 1.5,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 13,
                "n_jobs": 1 #вот это не идеальная настройка, но без нее ломается что-то под капотом и выдает segfault
        },
        "lightgbm": {
                "n_estimators": 700,
                "learning_rate": 0.03,
                "num_leaves": 24,
                "max_depth": 7,
                "min_child_samples": 25,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.05,
                "reg_lambda": 1.0,
                "random_state": 13,
                "n_jobs": 1, #вот это не идеальная настройка, но без нее ломается что-то под капотом и выдает segfault
                "verbosity": -1
        },
        "catboost": {
                "iterations": 700,
                "learning_rate": 0.04,
                "depth": 6,
                "l2_leaf_reg": 5,
                "random_strength": 1,
                "bagging_temperature": 1,
                "loss_function": "Logloss",
                "eval_metric": "Accuracy",
                "random_seed": 13,
                "verbose": False
        },
        "nn": {
                "epochs": 1000,
                "layers_size": (128, 64, 32),
                "dropout": 0.5,
                "batch_size": 64,
                "patience": 20
        }
    }
}

config = OmegaConf.create(conf)
