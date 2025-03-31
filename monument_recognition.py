import cv2
import numpy as np
# import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk  
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
MODEL_PATH = r"C:\Users\rampr\Desktop\project\monument_recognition_model.h5"
model = load_model(MODEL_PATH)

# Monument labels
monument_classes = ["Statue of Unity", "Taj Mahal", "India Gate", "Qutub Minar", "Charminar", 
                    "Raigad", "Shivneri", "Ellora caves", "Ajanta caves", "Sinhgad","Rajgad"]

# Monument Information
monument_info_dict = {
    "Statue of Unity": "Located in Gujarat, India, this is the world's tallest statue at 182 meters...",
    "Taj Mahal": "A white marble mausoleum in Agra, India, built by Mughal Emperor Shah Jahan...",
    "India Gate": "A 42-meter high war memorial in New Delhi, honoring soldiers of World War I...",
    "Qutub Minar": "A UNESCO World Heritage Site in Delhi, standing at 73 meters...",
    "Charminar": "An iconic monument in Hyderabad, built in 1591...",
    "Raigad": "Raigad is a hill fort in Maharashtra, India...",
    "Shivneri": "Shivneri is the birthplace of Chatrapati Shivaji Maharaj...",
    "Ellora caves": "Ellora Caves are a UNESCO World Heritage Site...",
    "Ajanta caves": "The Ajanta Caves are 30 rock-cut Buddhist cave monuments...",
    "Sinhgad": "The main fortress of Maharashtra formerly known as Kondhana...",
    "Rajgad": "The frist captical of Maratha Empire."
}

def detect_monument(image):
    """Predict the monument name using the deep learning model."""
    image = cv2.resize(image, (224, 224))  
    image = img_to_array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  

    predictions = model.predict(image)
    class_index = np.argmax(predictions)

    print("Raw Predictions:", predictions)  # Debugging output
    print("Predicted Class Index:", class_index)  # Check which class is being predicted

    if class_index >= len(monument_classes):
        return "Unknown", "No information available."

    monument_name = monument_classes[class_index]
    monument_info = monument_info_dict.get(monument_name, "No information available.")

    return monument_name, monument_info

def upload_image():
    """Open a file dialog to let the user upload an image and display results."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if not file_path:
        return

    # Read and process the image
    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Unable to load image.")
        return

    monument_name, monument_info = detect_monument(image)

    # Convert image to display in Tkinter
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = Image.fromarray(image)
    image = image.resize((500, 350), Image.LANCZOS)  # Resize for better display
    img_tk = ImageTk.PhotoImage(image)

    # Update image display in GUI
    img_label.config(image=img_tk)
    img_label.image = img_tk  # Keep reference to avoid garbage collection

    # Update detected name and info labels below the image
    result_label.config(text=f"Detected Monument: {monument_name}", font=("Arial", 16, "bold"), bg="white")
    info_label.config(text=f"Info: {monument_info}", wraplength=600, justify="center", font=("Arial", 12), bg="white")

# Create GUI
root = tk.Tk()
root.title("Monument Recognition")
root.geometry("1600x900")  # Increased window size for better layout

# Load background image
bg_image_path = r"C:\Users\rampr\Desktop\project\monuments_dataset\bgnew.webp"
bg_image = Image.open(bg_image_path)
bg_image = bg_image.resize((1600, 900), Image.LANCZOS)  # Make background fit full window
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a Canvas for the background
canvas = tk.Canvas(root, width=1600, height=900)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Add a title label
title_label = tk.Label(root, text="Monument Recognition", font=("Arial", 18, "bold"), bg="white")
canvas.create_window(800, 200, window=title_label)

# Image display area (initially blank)
img_label = tk.Label(root, bg="white")
canvas.create_window(800, 230, window=img_label)

# Saffron-colored Upload Button
upload_button = tk.Button(
    root, text="Upload Image", command=upload_image, 
    font=("Arial", 14, "bold"), bg="#FF9933", fg="white", 
    padx=15, pady=5, relief="raised", borderwidth=3
)
canvas.create_window(800, 600, window=upload_button)

# Label for displaying recognized monument name
result_label = tk.Label(root, text="", font=("Arial", 14), bg="white")
canvas.create_window(800, 470, window=result_label)

# Label for displaying monument information
info_label = tk.Label(root, text="", font=("Arial", 12), bg="white", wraplength=700, justify="center")
canvas.create_window(800, 520, window=info_label)

# Start GUI loop
root.mainloop()
4

# 

