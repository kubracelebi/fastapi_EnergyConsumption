from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

df = pd.read_csv('GercekZamanliTuketim.csv')

data = df.groupby('Tarih').agg({'TuketimMiktari' : lambda TuketimMiktari: TuketimMiktari.sum()})

data.reset_index(inplace=True)

X = data[['Tarih']]
y = data[['TuketimMiktari']]


class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(Self, X, y = None):
        X['Tarih'] = pd.to_datetime(X['Tarih'], infer_datetime_format=True)
        X["year"] = X.Tarih.dt.year
        X["month"] = X.Tarih.dt.month
        X["day"] = X.Tarih.dt.day
        X["dow"] = X.Tarih.dt.dayofweek
        X["quarter"] = X.Tarih.dt.quarter
        X = X.drop("Tarih", axis=1)
        X = X.astype(str)
        return X