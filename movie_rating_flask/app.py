# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoder
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        genre = request.form["genre"]
        director = request.form["director"]
        actor = request.form["actor"]

        input_data = encoder.transform([[genre, director, actor]])
        prediction = model.predict(input_data)[0]

        return render_template("index.html", prediction=round(prediction, 2))

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
