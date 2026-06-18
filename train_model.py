import pandas as pd

df = pd.read_csv("Cleaned Car.csv")
print(df.columns)




print("TRAINING SCRIPT STARTED")



import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("Cleaned Car.csv")

print("Dataset loaded")

# Features and target
X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['name', 'company', 'fuel_type'])
    ],
    remainder='passthrough'
)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model.fit(X, y)

print("Model training completed")

# Save model
joblib.dump(model, "LinearRegressionModel.pkl")

print("MODEL TRAINED AND SAVED SUCCESSFULLY")