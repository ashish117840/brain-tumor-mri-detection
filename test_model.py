import numpy as np

from model_loader import CLASS_NAMES, load_trained_model

print("✅ Loading model...")
model = load_trained_model()
print("🎉 Model loaded successfully!")

dummy_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
predictions = model.predict(dummy_input, verbose=0)[0]
class_index = int(np.argmax(predictions))

print("\n🔍 Probabilities:", predictions)
print("🎯 Predicted Class Index:", class_index)
label = CLASS_NAMES[class_index]
pretty_label = "No Tumor" if label in {"notumor", "no_tumor"} else label
print("🧠 Predicted Tumor Type:", pretty_label)
