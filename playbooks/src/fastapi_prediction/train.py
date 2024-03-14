import pandas as pd
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from transformer import DateTransformer



df = pd.read_csv('GercekZamanliTuketim.csv')

data = df.groupby('Tarih').agg({'TuketimMiktari' : lambda TuketimMiktari: TuketimMiktari.sum()})

data.reset_index(inplace=True)

X = data[['Tarih']]
y = data[['TuketimMiktari']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# pipeline




pipeline = Pipeline([
    ('dates', DateTransformer()),
    ('ct-ohe', ColumnTransformer(
        [('ct',
            OneHotEncoder(handle_unknown='ignore', categories='auto'),
            [0,1])], remainder='passthrough')
    ),
    ('scaler', StandardScaler(with_mean=False)),
    ('estimator', RandomForestRegressor(n_estimators=50))
])


pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_true=y_test, y_pred=y_pred)
print(f"R2: {r2}")

# Save model


joblib.dump(pipeline, "saved_models/03.randomforest_date.pkl")

    # read models

estimator_loaded = joblib.load("saved_models/03.randomforest_date.pkl")



single_pred_raw = pd.DataFrame([['01.01.2023']])
single_pred_raw.columns = ['Tarih']

prediction = estimator_loaded.predict(single_pred_raw)

print("prediction", prediction)



