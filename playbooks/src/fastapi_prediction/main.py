from fastapi import FastAPI, Depends, Request
import joblib
import os
import pathlib
import pandas as pd
from transformer import DateTransformer
from models import Daily, Hourly, SaatTahmin, GunlukTahmin
import pandas as pd
from database import get_db, create_db_and_tables
from sqlalchemy.orm import Session

estimator_advertising_loaded = joblib.load("saved_models/03.randomforest_date.pkl")

estimator_saat_loaded = joblib.load("saved_models/03.randomforest_saat.pkl")


app = FastAPI()

create_db_and_tables()


def insert_daily(request, prediction, db):
    new_Daily = GunlukTahmin(
        Tarih=str(request["Tarih"]),
        prediction=prediction

    )

    with db as session:
        session.add(new_Daily)
        session.commit()
        session.refresh(new_Daily)

    return new_Daily


def insert_hourly(request, prediction, db):
    new_hourly = SaatTahmin(
        Saat=str(request["Saat"]),
        prediction=prediction

    )

    with db as session:
        session.add(new_hourly)
        session.commit()
        session.refresh(new_hourly)

    return new_hourly



def make_daily_prediction(model, request):
    # parse input from request
    Tarih = request["Tarih"]
    # Make an input vector
    single_pred = pd.DataFrame([[Tarih]])
    single_pred.columns = ['Tarih']

    # Predict
    prediction = model.predict(single_pred)

    return prediction[0]


def make_hourly_prediction(model, request):
    # parse input from request
    Saat = request["Saat"]


    single_pred = np.array([[Saat]])

    # Predict
    prediction = model.predict(single_pred)

    return prediction[0]


# Advertising prediction endpoint
# @app.post("/prediction/daily")
# def predict_iris(request: Daily):
#    prediction = make_daily_prediction(estimator_advertising_loaded, request.dict())
#    return {"result": prediction}


@app.post("/prediction/daily")
async def predict_Daily(request: Daily,  db: Session = Depends(get_db)):
    prediction = make_daily_prediction(estimator_advertising_loaded, request.dict())
    db_insert_record = insert_daily(request=request.dict(), prediction=prediction, db=db)
    return {"prediction": str(prediction), "db_record": db_insert_record}


@app.post("/prediction/Hourly")
async def predict_Hourly(request: Hourly,  db: Session = Depends(get_db)):
    prediction = make_hourly_prediction(estimator_saat_loaded, request.dict())
    db_insert_record = insert_hourly(request=request.dict(), prediction=prediction, db=db)
    return {"prediction": str(prediction), "db_record": db_insert_record}
