import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
MODEL_PATH = r"C:\Users\rampr\Desktop\project\monument_recognition_model.h5"
model = load_model(MODEL_PATH)

# Monument labels
monument_classes = ["Statue of Unity", "Taj Mahal", "India Gate", "Qutub Minar", "Charminar"]

# Custom Monument Information
monument_info_dict = {
    "Statue of Unity": "Located in Gujarat, India, this is the world's tallest statue at 182 meters, built in honor of Sardar Vallabhbhai Patel.",
    "Taj Mahal": "A white marble mausoleum in Agra, India, built by Mughal Emperor Shah Jahan for his wife Mumtaz Mahal.",
    "Qutub Minar": "A UNESCO World Heritage Site in Delhi, standing at 73 meters, built in 1193 by Qutb-ud-din Aibak.",
    "India Gate": "A 42-meter high war memorial in New Delhi, honoring the soldiers of the British Indian Army who fought in World War I.",
    "Charminar": "An iconic monument in Hyderabad, built in 1591, featuring four grand arches and a mosque on its top floor."
}

def detect_monument(image):
    """Predict the monument name using the deep learning model."""
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    predictions = model.predict(image)
    class_index = np.argmax(predictions)
    
    if class_index >= len(monument_classes):
        return "Unknown", "No information available."

    monument_name = monument_classes[class_index]
    monument_info = monument_info_dict.get(monument_name, "No information available.")
    
    return monument_name, monument_info

def upload_image():
    """Open a file dialog to let the user upload an image and process it."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if not file_path:
        print("No file selected.")
        return
    
    image = cv2.imread(file_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    monument_name, monument_info = detect_monument(image)

    # Overlay text on the image
    cv2.putText(image, monument_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize the image before displaying
    image = cv2.resize(image, (800, 600))

    # Display the image with detected monument details
    cv2.imshow("Monument Recognition", image)
    
    # Show custom information in a message box
    messagebox.showinfo("Monument Detected", f"Detected: {monument_name}\n\n{monument_info}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# GUI for Image Upload
root = tk.Tk()
root.title("Monument Recognition")

label = tk.Label(root, text="Click the button to upload an image", font=("Arial", 14))
label.pack(pady=10)

upload_button = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 12))
upload_button.pack(pady=10)

root.mainloop()
