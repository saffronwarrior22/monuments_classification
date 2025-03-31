from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import save_model

# Load MobileNetV2 with pre-trained ImageNet weights
model = MobileNetV2(weights="imagenet")

# Save it as an H5 file
model.save("monument_recognition_model.h5")

print("✅ Model saved as 'monument_recognition_model.h5'")
