import os
import numpy as np
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the best performing model. VGG16 is a good candidate.
MODEL_PATH = r'C:\Users\DELL\PycharmProjects\multi_fish_classifier\best_model_VGG16.h5'
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train_models.py first.")

# The class labels for your dataset
class_labels = [
    'Black Sea Sprat', 'Gilt-Head Bream', 'Hake', 'Horse Mackerel',
    'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp',
    'Striped Red Mullet', 'Trout'
]

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses an image for model prediction.
    """
    img = Image.open(image_path).resize(target_size)
    img_array = np.array(img) / 255.0  # Rescale to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess and predict
            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class = class_labels[predicted_class_index]
            confidence = predictions[0][predicted_class_index] * 100

            return render_template_string(HTML_TEMPLATE,
                                          prediction=f'Predicted class: {predicted_class}',
                                          confidence=f'Confidence: {confidence:.2f}%',
                                          image_path=f'uploads/{filename}')

    return render_template_string(HTML_TEMPLATE, prediction="", confidence="", image_path="")

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Fish Classifier</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f7fafc;
        }
        .container {
            max-width: 800px;
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
    <div class="container mx-auto p-8 bg-white rounded-xl shadow-lg">
        <h1 class="text-3xl font-bold text-center mb-6 text-gray-800">Fish Classifier</h1>
        <p class="text-center text-gray-600 mb-8">Upload an image to get a prediction.</p>

        <form method="post" enctype="multipart/form-data" class="flex flex-col items-center space-y-4">
            <input type="file" name="file" class="text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-full file:border-0
                file:text-sm file:font-semibold
                file:bg-blue-50 file:text-blue-700
                hover:file:bg-blue-100" />
            <button type="submit" class="w-full sm:w-auto px-6 py-2 rounded-full bg-blue-600 text-white font-semibold shadow-md hover:bg-blue-700 transition duration-300">
                Predict
            </button>
        </form>

        {% if prediction %}
        <div class="mt-8 p-6 bg-gray-50 rounded-lg text-center">
            <h2 class="text-xl font-semibold mb-2 text-gray-700">{{ prediction }}</h2>
            <p class="text-lg text-gray-600 mb-4">{{ confidence }}</p>
            {% if image_path %}
            <img src="{{ image_path }}" alt="Uploaded Image" class="mx-auto mt-4 rounded-lg shadow-sm" style="max-width: 300px; max-height: 300px;">
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    # Add a simple route for serving uploaded images
    from flask import send_from_directory
    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    app.run(debug=True)
