# кастомный трансфор, который придется делать в рантайме
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin 


#кастомные трансформ классы
#считает среднее по колонкам и заменяет средним пропуски
class MyTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, cols:list=None):
        self.cols = cols

    def fit(self, X:pd.DataFrame, y=None):
        self.means_ = {}
        if self.cols is not None:
            for col in self.cols:
                self.means_[col] = X[col].mean()
        return self

    def transform(self, X:pd.DataFrame):
        X = X.fillna(value=self.means_)
        return X
