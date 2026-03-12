import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from supabase import create_client, Client

# Supabase Verbindung (Ersetze die Werte durch deine echten Daten!)
SUPABASE_URL = "DEINE_SUPABASE_URL"
SUPABASE_KEY = "DEIN_SUPABASE_ANON_KEY"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def load_my_model():
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

    # Preprocessing
    size = (224, 224)
    image_resized = image.resize(size)
    image_array = np.asarray(image_resized).astype('float32')
    normalized_image_array = (image_array / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    display_name = class_name.split(" ", 1)[1] if " " in class_name else class_name
    
    st.success(f"Farbe: {display_name} ({confidence_score:.2%})")

    # DATEN IN SUPABASE SPEICHERN
    try:
        data_to_save = {"farbe": display_name, "konfidenz": confidence_score}
        supabase.table("farberkennungen").insert(data_to_save).execute()
        st.info("Ergebnis wurde in der Datenbank gespeichert!")
    except Exception as e:
        st.error(f"Datenbankfehler: {e}")
