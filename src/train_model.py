import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 10

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Data Preprocessing
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# -----------------------------
# CNN Model Architecture
# -----------------------------
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu',
                 input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

# -----------------------------
# Save Model
# -----------------------------
model.save("models/emotion_model.h5")

print("✅ Model training completed and saved successfully!")
