from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__, template_folder="templates")

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

# HDI category function
def classify_hdi(value):
    if value >= 0.8:
        return "Very High Human Development"
    elif value >= 0.7:
        return "High Human Development"
    elif value >= 0.55:
        return "Medium Human Development"
    else:
        return "Low Human Development"

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        rank = float(request.form.get("rank"))

        # IMPORTANT: ONLY 1 FEATURE
        input_data = np.array([[rank]])

        prediction = model.predict(input_data)[0]
        category = classify_hdi(prediction)

        return render_template(
            "index.html",
            prediction_text=f"HDI: {prediction:.3f} | {category}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)