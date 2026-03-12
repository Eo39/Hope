import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. Modell laden mit TFSMLayer (für Keras 3 Kompatibilität)
# WICHTIG: Ersetze 'model/your_model_directory' durch den echten Ordnernamen in deinem Repo!
model_path = 'model/your_model_directory' 
model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

# Klassenbezeichnungen
class_names = ['rot', 'blau', 'schwarz']

def preprocess_image(image):
    # Bild auf die Eingabegröße des Modells skalieren (224x224 ist Standard für viele Modelle)
    target_size = (224, 224) 
    image = image.resize(target_size)
    image_array = np.array(image).astype('float32') # Konvertierung zu float32 für TF
    image_array = image_array / 255.0  # Normalisieren
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(image):
    processed_img = preprocess_image(image)
    
    # TFSMLayer wird wie eine Funktion aufgerufen, nicht mit .predict()
    predictions_dict = model(processed_img)
    
    # TFSMLayer gibt oft ein Dictionary zurück (z.B. {'output_0': tensor})
    # Wir nehmen den ersten verfügbaren Output-Wert
    output_key = list(predictions_dict.keys())[0]
    predictions = predictions_dict[output_key].numpy()
    
    top_idx = np.argmax(predictions[0])
    confidence = predictions[0][top_idx] * 100
    class_name = class_names[top_idx]
    return class_name, confidence

st.title("T-Shirt Farbenerkennung für Blinde")

uploaded_file = st.file_uploader("Lade ein Bild des T-Shirts hoch", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild öffnen
    image = Image.open(uploaded_file).convert('RGB') # Sicherstellen, dass es RGB ist

    st.image(image, caption='Hochgeladenes Bild', use_container_width=True)

    # Vorhersage durchführen
    try:
        class_name, confidence = predict_image(image)

        # Ergebnis anzeigen
        if confidence < 50:
            st.warning(f"Unsicher: Die Farbe könnte '{class_name}' sein, aber die Sicherheit ist zu gering ({confidence:.2f}%).")
        else:
            st.success(f"Die erkannte Farbe ist: **{class_name}** ({confidence:.2f}%)")
    except Exception as e:
        st.error(f"Fehler bei der Vorhersage: {e}")
