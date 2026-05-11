import tensorflow as tf
from tensorflow.keras.utils import image
import numpy as np
import cv2

# =========================
# LOAD MODELDetection-of-Fruit-Ripeness-Using-Image-Processing/predict.py
# =========================
model = tf.keras.models.load_model(
    "models/fruit_ripeness_model.h5"
)

# =========================
# IMAGE PATH
# =========================
img_path = "test_images/test.jpg"

# =========================
# LOAD IMAGE
# =========================
img = image.load_img(
    img_path,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

# =========================
# PREDICTION
# =========================
prediction = model.predict(img_array)

print("Nilai Prediksi:", prediction[0][0])

if prediction[0][0] > 0.5:
    label = "MENTAH"
else:
    label = "MATANG"

print("Hasil:", label)

# =========================
# SHOW IMAGE
# =========================
img_cv = cv2.imread(img_path)

cv2.putText(
    img_cv,
    label,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow("Prediction", img_cv)

cv2.waitKey(0)
cv2.destroyAllWindows()