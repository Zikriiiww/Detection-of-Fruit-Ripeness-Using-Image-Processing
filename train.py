import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================
# DATASET PATH
# ==========================
TRAIN_DIR = "dataset/train/train_tomat"
VALID_DIR = "dataset/valid/valid_tomat"

# ==========================
# IMAGE PREPROCESSING
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,

    rotation_range=25,

    width_shift_range=0.2,
    height_shift_range=0.2,

    zoom_range=0.2,

    shear_range=0.2,

    horizontal_flip=True,

    brightness_range=[0.8, 1.2],

    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(
    rescale=1./255
)

# ==========================
# LOAD DATASET
# ==========================
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=(224,224),

    batch_size=32,

    class_mode='binary',

    shuffle=True
)

valid_data = valid_datagen.flow_from_directory(
    VALID_DIR,

    target_size=(224,224),

    batch_size=32,

    class_mode='binary',

    shuffle=False
)

# ==========================
# CHECK LABEL
# ==========================
print("\nCLASS LABELS")
print(train_data.class_indices)

print(
    "\nJumlah Train:",
    train_data.samples
)

print(
    "Jumlah Valid:",
    valid_data.samples
)

# ==========================
# LOAD VGG16
# ==========================
base_model = VGG16(
    weights='imagenet',

    include_top=False,

    input_shape=(224,224,3)
)

# Freeze layer awal
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze layer akhir
for layer in base_model.layers[-4:]:

    layer.trainable = True

# ==========================
# BUILD MODEL
# ==========================
model = Sequential([

    base_model,

    Flatten(),

    Dense(
        256,
        activation='relu'
    ),

    BatchNormalization(),

    Dropout(0.5),

    Dense(
        128,
        activation='relu'
    ),

    BatchNormalization(),

    Dropout(0.3),

    Dense(
        1,
        activation='sigmoid'
    )

])

# ==========================
# COMPILE
# ==========================
model.compile(

    optimizer=Adam(
        learning_rate=0.0001
    ),

    loss='binary_crossentropy',

    metrics=['accuracy']
)

model.summary()

# ==========================
# CALLBACKS
# ==========================
os.makedirs(
    "models",
    exist_ok=True
)

early_stop = EarlyStopping(

    monitor='val_loss',

    patience=5,

    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.5,

    patience=2,

    verbose=1
)

checkpoint = ModelCheckpoint(

    "models/tomato_ripeness_model.h5",

    monitor='val_accuracy',

    save_best_only=True,

    verbose=1
)

# ==========================
# TRAINING
# ==========================
history = model.fit(

    train_data,

    validation_data=valid_data,

    epochs=25,

    callbacks=[

        early_stop,

        reduce_lr,

        checkpoint
    ]
)

# ==========================
# EVALUATION
# ==========================
valid_data.reset()

predictions = model.predict(
    valid_data
)

predicted_classes = (

    predictions > 0.5

).astype(int)

print("\nCLASSIFICATION REPORT\n")

print(

    classification_report(

        valid_data.classes,

        predicted_classes,

        target_names=list(
            valid_data.class_indices.keys()
        )

    )

)

print("\nCONFUSION MATRIX")

print(

    confusion_matrix(

        valid_data.classes,

        predicted_classes

    )

)

# ==========================
# FINAL ACCURACY
# ==========================
train_acc = history.history[
    'accuracy'
][-1]

val_acc = history.history[
    'val_accuracy'
][-1]

print(
    "\nTraining Accuracy:",
    train_acc
)

print(
    "Validation Accuracy:",
    val_acc
)

# ==========================
# PLOT ACCURACY
# ==========================
plt.figure(
    figsize=(8,5)
)

plt.plot(
    history.history['accuracy']
)

plt.plot(
    history.history['val_accuracy']
)

plt.title(
    "Tomato Ripeness Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend([
    "Train",
    "Validation"
])

plt.savefig(
    "models/accuracy_graph.png"
)

plt.show()