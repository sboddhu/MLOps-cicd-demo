import joblib

def test_model_prediction():
    model = joblib.load("model.joblib")
    pred = model.predict([[1.0, 0.2, 3.0]])
    assert pred[0] in [0, 1]
