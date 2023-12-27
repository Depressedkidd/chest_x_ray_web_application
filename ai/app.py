from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Загрузка сохраненной модели
model = load_model('./x_ray_model.h5')
# Функция предсказания для загруженных изображений
def preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction_text="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction_text="No selected file")
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            image_array = preprocess_image(file_path)
            prediction = model.predict(image_array)
            predicted_class = np.argmax(prediction[0])
            probabilities = tf.nn.softmax(prediction[0])
        if predicted_class == 0:
            result = "NORMAL"
        else:
            result = "PNEUMONIA"
        probabilities = [f'{p:.4f}' for p in probabilities.numpy()]  # Преобразование вероятностей в строки с 4 знаками после запятой
        return render_template('index.html', prediction_text=f'The image is predicted as: {result}', probabilities=probabilities)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)