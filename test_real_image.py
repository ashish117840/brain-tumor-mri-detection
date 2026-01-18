import os

import cv2
import numpy as np

from model_loader import CLASS_NAMES, load_trained_model, preprocess_bgr_image

# ===============================
# CONFIGURATION
# ===============================
IMAGE_PATH = r"sample MRI Images/notumor/Te-no_0013.jpg"  # change to your MRI image path if needed

# ===============================
# LOAD MODEL
# ===============================
print("✅ Loading model...")
model = load_trained_model()
print("🎉 Model loaded successfully!")

# ===============================
# LOAD & PREPROCESS IMAGE
# ===============================
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"❌ Image not found: {IMAGE_PATH}")

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"❌ Could not read image: {IMAGE_PATH}")

img_array = preprocess_bgr_image(img)

print("🖼 Image loaded & preprocessed")
print("Shape:", img_array.shape)

# ===============================
# RUN PREDICTION
# ===============================
predictions = model.predict(img_array, verbose=0)[0]

predicted_index = int(np.argmax(predictions))
predicted_label = CLASS_NAMES[predicted_index]
confidence = predictions[predicted_index] * 100

# ===============================
# RESULTS
# ===============================
print("\n🔍 Class Probabilities:")
for i, prob in enumerate(predictions):
    print(f"{CLASS_NAMES[i]:12}: {prob:.4f}")

print("\n🎯 FINAL PREDICTION")
pretty_label = "No Tumor" if predicted_label in {"notumor", "no_tumor"} else predicted_label
print(f"🧠 Tumor Type : {pretty_label}")
print(f"📊 Confidence : {confidence:.2f}%")
