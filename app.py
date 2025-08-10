import os
import gdown
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ======== Google Drive Model Links ========
MODEL_FILES = {
    "base": "1UBfUF_kOWY_mnl9hm4BzRprBvERCb2kq",
    "efficientnet": "1WYpyrr7ggtfylOvfP_axoNN5FbmlxFkv",
    "custom_cnn": "1cH51G8OpMQxarIlH7Xie2tS2K1TQ6dNE",
    "vgg": "1DhPJQD6Xs8NPkYdGrlV7G31-iS5aXVZf",
    "resnet": "1rmtdeJMbVYL4UoAK1BLcl-DRoe0mJwMs",
    "mobilenet": "1uJIIw6F2C3ARwSZeaJ9-eSijTtS9dOkT",
    "inception": "1K2C6mRcf6VJtO124rcvzn4OUHUtuji4x"
}

# ======== Download & Load Models ========
def download_models():
    os.makedirs("models", exist_ok=True)
    for name, file_id in MODEL_FILES.items():
        file_path = f"models/{name}.h5"
        if not os.path.exists(file_path):
            st.write(f"📥 Downloading {name} model...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_path, quiet=False)
        else:
            st.write(f"✅ {name} model already exists.")

@st.cache_resource
def load_models():
    models = {}
    for name in MODEL_FILES.keys():
        models[name] = load_model(f"models/{name}.h5")
    return models

# ======== Image Preprocessing ========
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ======== Streamlit App ========
st.title("🐟 Fish Classifier App")
st.write("Select a model and upload an image to classify the fish.")

# Download models if not available
download_models()

# Load all models
models = load_models()

# UI elements
model_choice = st.selectbox("Choose a model:", list(MODEL_FILES.keys()))
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array = preprocess_image(uploaded_file)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        model = models[model_choice]
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        st.success(f"Predicted class: {predicted_class}")



