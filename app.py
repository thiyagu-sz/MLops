from flask import Flask, request, render_template
import pickle
import pandas as pd
import os

# Train model if not present
if not os.path.exists("model.pkl"):
    import train

model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()
        print("FORM DATA:", data)

        df = pd.DataFrame([data])

        # Convert numeric fields
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")

        # Predict
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0]

        print("Probabilities:", proba)
        print("Raw Prediction:", prediction)

        # Convert prediction to label
        label = "CHURN" if prediction == "Yes" else "STAY"

        # Confidence score
        confidence = round(max(proba) * 100, 2)

        return render_template(
            "index.html",
            prediction=label,
            confidence=confidence
        )

    except Exception as e:
        print("ERROR:", e)
        return render_template("index.html", prediction="Error occurred")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)