from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/")
def home():
    return "Credit Risk Model is Live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run()