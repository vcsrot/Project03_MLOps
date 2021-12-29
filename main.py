# Put the code for your API here.
import os
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from starter.train_model import online_inference

app = FastAPI()

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class RowData(BaseModel):
    age: int = Field(..., example=25)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=14)
    marital_status: str = Field(..., example="Divorced")
    occupation: str = Field(..., example="Exec-managerial")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example="United-States")


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system('rm -rf .dvc/cache')
    os.system('rm -rf .dvc/tmp/lock')
    os.system('dvc config core.hardlink_lock true')
    if os.system("dvc pull -q") != 0:
        exit("dvc pull failed")
    os.system("rm -rf .dvc .apt/usr/lib/dvc")


@app.get("/")
def home():
    return {"Ok!": "Status code success."}


@app.post('/inference')
async def predict_income(inputrow: RowData):
    row_dict = jsonable_encoder(inputrow)
    model_path = 'model/random_forest_model.pkl'
    prediction = online_inference(row_dict, model_path, CAT_FEATURES)

    return {"income class": prediction}