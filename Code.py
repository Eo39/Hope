import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Modell laden (Pfad anpassen, falls anders gespeichert)
model_path = 'model/your_model_directory'  # Ersetze durch deinen Pfad
model = tf.keras.models.load_model(model_path)

# Klassenbezeichnungen, die dein Modell erkennt
class_names = ['rot', 'blau', 'schwarz']

def preprocess_image(image):
    # Bild auf die Eingabegröße des Modells skalieren
    target_size = (224, 224)  # Passe die Größe an dein Modell an
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalisieren
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    top_idx = np.argmax(predictions)
    confidence = predictions[0][top_idx] * 100
    class_name = class_names[top_idx]
    return class_name, confidence

st.title("T-Shirt Farbenerkennung für Blinde")

uploaded_file = st.file_uploader("Lade ein Bild des T-Shirts hoch", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild öffnen
    image = Image.open(uploaded_file)

    st.image(image, caption='Hochgeladenes Bild', use_column_width=True)

    # Vorhersage durchführen
    class_name, confidence = predict_image(image)

    # Ergebnis anzeigen
    if confidence < 50:
        st.write("Die Farbe des T-Shirts wurde nicht erkannt.")
    else:
        st.write(f"Die erkannte Farbe ist: {class_name} ({confidence:.2f}%)")
