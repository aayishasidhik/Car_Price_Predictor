import sys
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

print("Python:", sys.executable)

app = Flask(__name__)

# ----------------------------
# LOAD MODEL (FIXED)
# ----------------------------
MODEL_PATH = "LinearRegressionModel.pkl"

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model file not found:", MODEL_PATH)
    sys.exit()

try:
    model = joblib.load(MODEL_PATH)
    print("MODEL LOADED SUCCESSFULLY")
except Exception as e:
    print("MODEL LOAD ERROR:", e)
    sys.exit()

# ----------------------------
# LOAD DATASET
# ----------------------------
DATA_PATH = "Cleaned Car.csv"

if not os.path.exists(DATA_PATH):
    print("ERROR: Dataset not found:", DATA_PATH)
    sys.exit()

car = pd.read_csv(DATA_PATH)

# ----------------------------
# HOME PAGE
# ----------------------------
@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )

# ----------------------------
# PREDICTION ROUTE
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        kms_driven = int(request.form.get('kilo_driven'))

        input_data = pd.DataFrame(
            [[car_model, company, year, kms_driven, fuel_type]],
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
        )

        prediction = model.predict(input_data)

        return str(round(prediction[0], 2))

    except Exception as e:
        return f"ERROR DURING PREDICTION: {str(e)}"

# ----------------------------
# RUN APP
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)