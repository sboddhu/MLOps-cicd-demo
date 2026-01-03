import joblib
import sys

model = joblib.load("model.joblib")
features = list(map(float, sys.argv[1:]))
prediction = model.predict([features])

print("Prediction:", prediction[0])
