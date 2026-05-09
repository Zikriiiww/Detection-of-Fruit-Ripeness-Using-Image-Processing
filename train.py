import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# =========================
# PATH DATASET
# =========================
TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"

# =========================
# IMAGE GENERATOR
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(
    rescale=1./255
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

valid_data = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# =========================
# LOAD VGG16
# =========================
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze layer VGG16
for layer in base_model.layers:
    layer.trainable = False

# =========================
# BUILD MODEL
# =========================
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN
# =========================
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=10
)

# =========================
# SAVE MODEL
# =========================
os.makedirs("models", exist_ok=True)

model.save("models/fruit_ripeness_model.h5")

print("\nMODEL BERHASIL DISIMPAN!")

# =========================
# PLOT ACCURACY
# =========================
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.show()