from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from keras.models import load_model
import json
import io
from PIL import Image
import base64

app = Flask(__name__)

MODEL_URL = "https://github.com/AirtonStuwar/FRUTAS/releases/tag/v1"
MODEL_PATH = "modelo_frutas.h5"

with open('etiquetas_frutas.json') as f:
    fruits = json.load(f)

# Preprocesamiento de la imagen
def prepare_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('RGB')
    img = img.resize((50, 50))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Añadir batch size
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Recibir la imagen en base64
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    
    # Preparar la imagen para la predicción
    img = prepare_image(img_data)

    # Hacer la predicción
    pred = model.predict(img)
    predicted_label = np.argmax(pred[0])
    predicted_class = list(fruits.keys())[predicted_label]
    predicted_prob = pred[0][predicted_label] * 100

    return jsonify({
        'label': predicted_class,
        'probability': f"{predicted_prob:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)
