import joblib

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

print(type(model))
print(type(scaler))