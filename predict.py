import tensorflow as tf

from tensorflow.keras.utils import (
    load_img,
    img_to_array
)

from tensorflow.keras.applications.vgg16 import (
    preprocess_input
)

import numpy as np
import cv2
import os


# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "models/tomato_ripeness_model.h5"

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("MODEL BERHASIL DIMUAT!")


# =========================
# LABEL
# =========================
CLASS_NAMES = {

    0:"MATANG",

    1:"MENTAH"

}


# =========================
# IMAGE PATH
# =========================
img_path = "tomatmentah.jpg"


# =========================
# CHECK FILE
# =========================
if not os.path.exists(img_path):

    print(
        "Gambar tidak ditemukan"
    )

    exit()


# =========================
# LOAD IMAGE
# =========================
img = load_img(

    img_path,

    target_size=(224,224)

)

img_array = img_to_array(
    img
)

# penting
img_array = preprocess_input(
    img_array
)

img_array = np.expand_dims(

    img_array,

    axis=0

)


# =========================
# PREDICT
# =========================
prediction = model.predict(

    img_array,

    verbose=0

)

prediction_value = float(

    prediction[0][0]

)

print(

    "\nRAW:",

    prediction_value

)


# =========================
# CLASSIFICATION
# =========================
predicted_class = (

    1

    if prediction_value > 0.5

    else 0

)

label = CLASS_NAMES[
    predicted_class
]


if predicted_class == 1:

    confidence = (

        prediction_value

        *100

    )

else:

    confidence = (

        (1-prediction_value)

        *100

    )


print(
    "\nHASIL:",
    label
)

print(
    f"CONF: {confidence:.2f}%"
)


# =========================
# DISPLAY IMAGE
# =========================
img_cv = cv2.imread(
    img_path
)

img_cv = cv2.resize(

    img_cv,

    (700,500)

)

color = (

    (0,255,0)

    if label=="MATANG"

    else (0,0,255)

)

cv2.putText(

    img_cv,

    f"HASIL : {label}",

    (20,50),

    cv2.FONT_HERSHEY_SIMPLEX,

    1,

    color,

    2

)

cv2.putText(

    img_cv,

    f"CONF : {confidence:.2f}%",

    (20,100),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.8,

    (255,0,0),

    2

)

cv2.putText(

    img_cv,

    f"RAW : {prediction_value:.4f}",

    (20,150),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.8,

    (255,255,0),

    2

)

cv2.imshow(

    "Tomato Ripeness Detection",

    img_cv

)

cv2.waitKey(0)

cv2.destroyAllWindows()