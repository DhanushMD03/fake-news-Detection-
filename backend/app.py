from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

# ✅ Train model if not present
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    print("Training model...")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # Small dataset (for deployment)
    texts = [
        "Government passed a new law",
        "Scientists discovered a new vaccine",
        "Aliens landed on earth secretly",
        "Magic cure hidden by authorities",
        "Official report confirms economic growth",
        "Secret society controls the world",
    ]

    labels = [1, 1, 0, 0, 1, 0]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    # Save model
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    print("Model trained and saved!")

# ✅ Load model
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