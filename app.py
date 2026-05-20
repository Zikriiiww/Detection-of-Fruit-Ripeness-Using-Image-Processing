from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model('models/tomato_ripeness_model.h5')

CLASS_NAMES = {
    0: 'Matang',
    1: 'Mentah'
}


# Route khusus agar Flask bisa serve foto yang diupload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    raw = float(pred[0][0])

    cls = 1 if raw > 0.5 else 0
    label = CLASS_NAMES[cls]
    conf = raw * 100 if cls == 1 else (1 - raw) * 100

    # Kirim path URL (bukan path file system) ke template
    filepath_url = f'/uploads/{file.filename}'

    return render_template(
        'index.html',
        prediction=label,
        confidence=f"{conf:.2f}",
        filepath=filepath_url
    )


if __name__ == '__main__':
    app.run(debug=True)