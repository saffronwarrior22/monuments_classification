# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os

# # Load Pretrained MobileNetV2 Model
# base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# # Freeze the base model layers
# base_model.trainable = False

# # Path to dataset
# DATASET_PATH = r"C:\Users\rampr\Desktop\project\monuments_dataset"

# # Load dataset dynamically
# train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

# train_generator = train_datagen.flow_from_directory(
#     r"C:\Users\rampr\Desktop\project\monuments_dataset",
#     target_size=(224, 224),
#     batch_size=4,
#     class_mode='categorical',
#     subset="training"  # Ensure it's getting the training set
# )

# val_generator = train_datagen.flow_from_directory(
#     r"C:\Users\rampr\Desktop\project\monuments_dataset",
#     target_size=(224, 224),
#     batch_size=4,
#     class_mode='categorical',
#     subset="validation"  # Ensure it's getting the validation set
# )


# # Dynamically determine the number of classes
# num_classes = len(train_generator.class_indices)
# print(f"✅ Detected {num_classes} monument classes.")
# print(f"✅ Training images found: {train_generator.samples}")
# print(f"✅ Validation images found: {val_generator.samples}")


# # Build the model dynamically based on detected classes
# model = tf.keras.Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     Dense(128, activation='relu'),
#     Dense(num_classes, activation='softmax')  # Dynamic number of classes
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(train_generator, validation_data=val_generator, epochs=10)

# # Save the trained model
# model.save("monument_recognition_model.h5")
# print("✅ Model training complete! Saved as monument_recognition_model.h5")
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Set dataset directory
DATASET_DIR = r"C:\Users\rampr\Desktop\project\monuments_dataset\image"  # Update with your actual dataset path

# Image parameters
IMG_SIZE = (224, 224)  # Image size for MobileNetV2
BATCH_SIZE = 32  # Adjust batch size based on system memory

# Data Augmentation & Loading
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% training, 20% validation
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
print("Actual Class Mapping:", train_generator.class_indices)
# {'image1': 0, 'image10': 1, 'image11': 2, 'image2': 3, 'image3': 4, 'image4': 5, 'image5': 6, 'image6': 7, 'image7': 8, 'image8': 9, 'image9': 10}

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Print class indices (Mapping of folder names to labels)
print("Class Labels Mapping:", train_generator.class_indices)


# Load MobileNetV2 without the top classification layer
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze base model layers (only train last layers)
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Freeze only early layers
    layer.trainable = False


# Build Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Dynamically set to 11 classes
])

# Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("monument_recognition_model.h5", save_best_only=True, monitor='val_loss', mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    callbacks=[checkpoint, early_stop]
)

# Save Final Model
model.save("monument_recognition_model_final.h5")

print("Model training completed and saved successfully!")
