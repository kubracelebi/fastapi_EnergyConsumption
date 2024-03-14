import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
import joblib


df = pd.read_csv('GercekZamanliTuketim.csv')

X = df[['Saat']]
y = df[['TuketimMiktari']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

pipeline = Pipeline([
    ('ct-ohe', ColumnTransformer(
        [('ct',
          OneHotEncoder(handle_unknown='ignore', categories='auto'),
          [0,-1])], remainder='passthrough')
    ),
    ('scaler', StandardScaler(with_mean=False)),
    ('estimator', RandomForestRegressor(n_estimators=10))
])

pipeline.fit(X_train,y_train)
pipeline.predict(X_test)



joblib.dump(pipeline, "saved_models/03.randomforest_saat.pkl")

    # read models

estimator_loaded = joblib.load("saved_models/03.randomforest_saat.pkl")



single_pred_raw = np.array([['20:00']])

prediction = estimator_loaded.predict(single_pred_raw)

print("prediction", prediction)