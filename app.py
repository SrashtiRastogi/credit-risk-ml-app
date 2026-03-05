from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    loan_amnt=float(request.form["loan_amnt"])
    term=float(request.form["term"])
    int_rate=float(request.form["int_rate"])
    installment=float(request.form["installment"])
    annual_inc=float(request.form["annual_inc"])
    dti=float(request.form["dti"])
    grade=request.form["grade"]

    data={
        "loan_amnt":loan_amnt,
        "term":term,
        "int_rate":int_rate,
        "installment":installment,
        "annual_inc":annual_inc,
        "dti":dti
    }

    for g in ["grade_B","grade_C","grade_D","grade_E","grade_F","grade_G"]:
        data[g]=0

    data["grade_"+grade]=1

    df=pd.DataFrame([data])
    df=df.reindex(columns=columns,fill_value=0)

    df_scaled=scaler.transform(df)

    prediction=model.predict(df_scaled)

    if prediction[0]==1:
        result="High Credit Risk"
    else:
        result="Low Credit Risk"

    return render_template("result.html",prediction_text=result)

if __name__=="__main__":
    app.run()