from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("model.joblib")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: list):
    pred = model.predict([features])
    return {"prediction": int(pred[0])}
