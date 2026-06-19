import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Configuration ---
IMG_SIZE = (128, 128)
CLASS_NAMES = {0: "Chat 🐱", 1: "Chien 🐶", 2: "Autre 🌍"}
COLORS = {0: "#FF6B6B", 1: "#4ECDC4", 2: "#95E1D3"}

# --- Chargement des modèles ---
@st.cache_resource
def load_models():
    model_scratch = tf.keras.models.load_model(r"D:\IA\Projet_Perso_IA\best_cnn_scratch.h5")
    model_tl      = tf.keras.models.load_model(r"D:\IA\Projet_Perso_IA\best_model_tl.h5")
    return model_scratch, model_tl

model_scratch, model_tl = load_models()

# --- Interface ---
st.title("🐾 Chat, Chien ou Autre ?")
st.write("Uploade une image et compare les deux modèles !")

# Choix du modèle
model_choice = st.radio(
    "Choisis le modèle :",
    ["CNN From Scratch", "Transfer Learning", "Les deux"]
)

# Upload image
uploaded_file = st.file_uploader(
    "Glisse une image ici",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Afficher l'image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image uploadée", width=300)

    # Préparer l'image
    img_resized = img.resize(IMG_SIZE)
    img_array  = np.array(img_resized) / 255.0
    img_array  = np.expand_dims(img_array, axis=0)

    # Fonction de prédiction
    def predict(model, img_array):
        probs = model.predict(img_array, verbose=0)[0]
        classe = np.argmax(probs)
        return classe, probs

    st.divider()

    # Affichage des résultats
    if model_choice in ["CNN From Scratch", "Les deux"]:
        st.subheader("🔵 CNN From Scratch")
        classe, probs = predict(model_scratch, img_array)
        st.success(f"Résultat : **{CLASS_NAMES[classe]}** ({probs[classe]*100:.1f}%)")
        for i, name in CLASS_NAMES.items():
            st.progress(float(probs[i]), text=f"{name} : {probs[i]*100:.1f}%")

    if model_choice in ["Transfer Learning", "Les deux"]:
        st.subheader("🟢 Transfer Learning")
        classe, probs = predict(model_tl, img_array)
        st.success(f"Résultat : **{CLASS_NAMES[classe]}** ({probs[classe]*100:.1f}%)")
        for i, name in CLASS_NAMES.items():
            st.progress(float(probs[i]), text=f"{name} : {probs[i]*100:.1f}%")
            