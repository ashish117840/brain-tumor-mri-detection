import tensorflow as tf
from tensorflow.keras.models import load_model

# Load using legacy HDF5 loader
model = load_model(
    "models/mri_vgg16_model.h5",
    compile=False
)

# Save again in TF 2.12 compatible format
model.save("models/mri_vgg16_model_fixed.h5")

print("✅ Model successfully converted!")
