import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Modell und Labels laden
@st.cache_resource
def load_my_model():
    # Wir laden es ohne compile, um Fehler mit alten Optimierern zu vermeiden
    return tf.keras.models.load_model('keras_model.h5', compile=False)

model = load_my_model()

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

class_names = load_labels('labels.txt')

st.title("T-Shirt Farbenerkennung")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)

    # Preprocessing (Exakt so wie Teachable Machine es vorgibt)
    size = (224, 224)
    image = image.resize(size)
    image_array = np.asarray(image).astype('float32')
    normalized_image_array = (image_array / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Anzeige (Säubere "0 rot" zu "rot")
    display_name = class_name.split(" ", 1)[1] if " " in class_name else class_name
    
    st.success(f"Farbe: {display_name} (Sicherheit: {confidence_score:.2%})")
