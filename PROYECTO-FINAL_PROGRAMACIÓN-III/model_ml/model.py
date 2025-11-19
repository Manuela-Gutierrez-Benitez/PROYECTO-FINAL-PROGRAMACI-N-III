import joblib
import pandas as pd

MODEL_PATH = "modelo_random_forest.pkl"

model = joblib.load(MODEL_PATH)

def predict_label(K, ph, humidity):
    data = pd.DataFrame([{
        "K": K,
        "ph": ph,
        "humidity": humidity
    }])

    return model.predict(data)[0]