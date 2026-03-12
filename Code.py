import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Modell und Labels laden
model_path = 'keras_model.h5'
model = tf.keras.models.load_model(model_path)

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

class_names = load_labels('labels.txt')

def preprocess_image(image):
    target_size = (224, 224) 
    image = image.resize(target_size)
    image_array = np.array(image).astype('float32')
    image_array = image_array / 255.0 
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    top_idx = np.argmax(predictions[0])
    confidence = predictions[0][top_idx] * 100
    class_name = class_names[top_idx]
    return class_name, confidence

st.title("T-Shirt Farbenerkennung für Blinde")

uploaded_file = st.file_uploader("Lade ein Bild des T-Shirts hoch", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Hochgeladenes Bild', use_container_width=True)

    try:
        class_name, confidence = predict_image(image)

        if confidence < 50:
            st.warning(f"Das Ergebnis ist unsicher ({confidence:.2f}%). Vermutung: {class_name}")
        else:
            st.success(f"Die erkannte Farbe ist: **{class_name}** ({confidence:.2f}%)")
            st.button("Ergebnis vorlesen", on_click=lambda: st.write(f"Spreche: {class_name}"))
    except Exception as e:
        st.error(f"Fehler bei der Vorhersage: {e}")
