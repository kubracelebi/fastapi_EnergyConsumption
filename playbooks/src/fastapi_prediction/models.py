from typing import Optional
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from sqlmodel import SQLModel, Field

class GunlukTahmin(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    Tarih: str
    prediction: float


class SaatTahmin(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    Saat: str
    prediction: float


class Daily(SQLModel):
    Tarih: str
    class Config:
        schema_extra = {
            "example": {
              'Tarih': '01.01.2023'
            }
        }



class Hourly(SQLModel):
    Saat: str

    class Config:
        schema_extra = {
            "example": {
              'Saat': '20:00'
            }
        }


