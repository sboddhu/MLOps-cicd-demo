import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/sample.csv")
X = df.drop("label", axis=1)
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model.joblib")
print("Model trained and saved."
