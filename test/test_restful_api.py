import os, sys
from fastapi.testclient import TestClient
import inspect

# Load app from parent folder:
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from restful_api import app

client = TestClient(app)

def test_home():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {"Ok!": "Status code success."}


def test_post_1():
    row1 = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    r = client.post('/inference', json=row1)
    assert r.status_code == 200
    assert r.json() == {"income class": '<=50K'}


def test_post_2():
    row2 = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }
    r = client.post('/inference', json=row2)
    assert r.status_code == 200
    assert r.json() == {"income class": '>50K'}
