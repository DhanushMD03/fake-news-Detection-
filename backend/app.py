from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Create app FIRST
app = Flask(__name__)
CORS(app)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return "Fake News API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    news = data["text"]

    transformed = vectorizer.transform([news])

    prediction = model.predict(transformed)[0]
    probability = model.predict_proba(transformed)[0]

    confidence = max(probability) * 100

    result = "Fake News ❌" if prediction == 0 else "Real News ✅"

    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)