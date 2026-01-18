from flask import Flask, render_template, request
import numpy as np
import cv2

from model_loader import CLASS_NAMES, load_trained_model, preprocess_bgr_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

model = load_trained_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return render_template("index.html", error="Please choose an image file.")

    try:
        raw = uploaded.read()
        if not raw:
            return render_template("index.html", error="Uploaded file is empty.")

        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return render_template("index.html", error="Could not decode image. Use JPG/PNG.")

        input_tensor = preprocess_bgr_image(img)

        probs = model.predict(input_tensor, verbose=0)[0]
        probs = np.asarray(probs, dtype=float)

        predicted_index = int(np.argmax(probs))
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = float(probs[predicted_index]) * 100.0

        pretty_label = (
            "No Tumor"
            if predicted_label in {"notumor", "no_tumor"}
            else predicted_label.title()
        )
        probabilities = [
            {
                "label": (
                    "No Tumor"
                    if name in {"notumor", "no_tumor"}
                    else name.title()
                ),
                "prob": float(p) * 100.0,
            }
            for name, p in zip(CLASS_NAMES, probs)
        ]

        probabilities.sort(key=lambda x: x["prob"], reverse=True)

        return render_template(
            "index.html",
            predicted_label=pretty_label,
            confidence=confidence,
            probabilities=probabilities,
        )
    except Exception as e:
        return render_template("index.html", error=f"Prediction failed: {e}")

if __name__ == "__main__":
    app.run(debug=True)
