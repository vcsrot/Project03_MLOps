import requests

sample = {
    "age": 54,
    "workclass": "Private",
    "fnlwgt": 51835,
    "education": "Prof-school",
    "education_num": 16,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 60,
    "native_country": "United-States"
}

response = requests.post(
    url='https://vin-project3-app.herokuapp.com/inference',
    json=sample
)

print(response.status_code)
print(response.json())