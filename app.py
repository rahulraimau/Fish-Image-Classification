import os
import numpy as np
from flask import Flask, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess
import pickle

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
UPLOAD_FOLDER = 'Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Model configurations
MODELS = {
    'inceptionv3': {
        'path': r'C:\Users\DELL\PycharmProjects\multi_fish_classifier\best_model_InceptionV3.h5',
        'preprocess': inceptionv3_preprocess,
        'history': 'inceptionv3_history.pkl',
        'report': 'inceptionv3_classification_report.txt',
        'confusion_matrix': 'inceptionv3_confusion_matrix.png',
        'training_history': 'inceptionv3_training_history.png',
        'display_name': 'InceptionV3'
    },
    'custom_cnn': {
        'path': r'C:\Users\DELL\PycharmProjects\multi_fish_classifier\custom_cnn_best.h5',
        'preprocess': lambda x: x / 255.0,  # Simple rescaling for custom CNN
        'history': 'custom_cnn_history.pkl',
        'report': 'custom_cnn_classification_report.txt',
        'confusion_matrix': 'custom_cnn_confusion_matrix.png',
        'training_history': 'custom_cnn_training_history.png',
        'display_name': 'Custom CNN'
    },
    'vgg16': {
        'path': r'C:\Users\DELL\PycharmProjects\multi_fish_classifier\best_model_VGG16.h5',
        'preprocess': vgg16_preprocess,
        'history': 'vgg16_history.pkl',
        'report': 'vgg16_classification_report.txt',
        'confusion_matrix': 'vgg16_confusion_matrix.png',
        'training_history': 'vgg16_training_history.png',
        'display_name': 'VGG16'
    },
    'resnet50': {
        'path': r'C:\Users\DELL\PycharmProjects\multi_fish_classifier\best_model_ResNet50.h5',
        'preprocess': resnet50_preprocess,
        'history': 'resnet50_history.pkl',
        'report': 'resnet50_classification_report.txt',
        'confusion_matrix': 'resnet50_confusion_matrix.png',
        'training_history': 'resnet50_training_history.png',
        'display_name': 'ResNet50'
    },
    'mobilenetv2':{
        'path': r'C:\Users\DELL\PycharmProjects\multi_fish_classifier\best_model_MobileNetV2.h5',
        'preprocess': mobilenetv2_preprocess,
        'history': 'mobilenetv2_history.pkl',
        'report': 'mobilenetv2_classification_report.txt',
        'confusion_matrix': 'mobilenetv2_confusion_matrix.png',
        'training_history': 'mobilenetv2_training_history.png',
        'display_name': 'MobileNetV2'
    },
    'efficientnetb0': {
        'path': r'C:\Users\DELL\PycharmProjects\multi_fish_classifier\efficientnetb0_best.h5',
        'preprocess': efficientnet_preprocess,
        'history': 'efficientnetb0_history.pkl',
        'report': 'efficientnetb0_classification_report.txt',
        'confusion_matrix': 'efficientnetb0_confusion_matrix.png',
        'training_history': 'efficientnetb0_training_history.png',
        'display_name': 'EfficientNetB0'
    }
}

# Load models and validation metrics
loaded_models = {}
val_metrics = {}
for model_name, config in MODELS.items():
    if os.path.exists(config['path']):
        loaded_models[model_name] = tf.keras.models.load_model(config['path'])
    else:
        print(f"Warning: Model file {config['path']} not found. Run training script for {model_name}.")
    if os.path.exists(config['history']):
        with open(config['history'], 'rb') as f:
            history = pickle.load(f)
        val_metrics[model_name] = {
            'val_accuracy': history['val_accuracy'][-1],
            'val_loss': history['val_loss'][-1]
        }
    else:
        val_metrics[model_name] = {'val_accuracy': None, 'val_loss': None}

# Class labels matching training dataset
class_labels = [
    'animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet', 'fish sea_food red_sea_bream',
    'fish sea_food sea_bass', 'fish sea_food shrimp',
    'fish sea_food striped_red_mullet', 'fish sea_food trout'
]

def preprocess_image(image_path, model_name, target_size=(224, 224)):
    """Preprocess an image for the selected model."""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            img.save(image_path)
        img = img.resize(target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = MODELS[model_name]['preprocess'](img_array)
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    confidence = ""
    image_path = ""
    warning = ""
    classification_report = ""
    confusion_matrix_path = ""
    training_history_path = ""
    val_accuracy = None
    val_loss = None
    selected_model = request.form.get('model', 'inceptionv3') if request.method == 'POST' else 'inceptionv3'

    # Load evaluation metrics for selected model
    if selected_model in MODELS:
        report_path = MODELS[selected_model]['report']
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                classification_report = f.read()
        confusion_matrix_path = MODELS[selected_model]['confusion_matrix'] if os.path.exists(MODELS[selected_model]['confusion_matrix']) else ""
        training_history_path = MODELS[selected_model]['training_history'] if os.path.exists(MODELS[selected_model]['training_history']) else ""
        val_accuracy = val_metrics[selected_model]['val_accuracy']
        val_loss = val_metrics[selected_model]['val_loss']

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template_string(HTML_TEMPLATE,
                                        prediction="Error: No file uploaded.",
                                        confidence="",
                                        image_path="",
                                        warning="Please upload a valid image file.",
                                        classification_report=classification_report,
                                        confusion_matrix_path=confusion_matrix_path,
                                        training_history_path=training_history_path,
                                        val_accuracy=val_accuracy,
                                        val_loss=val_loss,
                                        selected_model=selected_model,
                                        models=MODELS)
        file = request.files['file']
        if file.filename == '':
            return render_template_string(HTML_TEMPLATE,
                                        prediction="Error: No file selected.",
                                        confidence="",
                                        image_path="",
                                        warning="Please select an image file.",
                                        classification_report=classification_report,
                                        confusion_matrix_path=confusion_matrix_path,
                                        training_history_path=training_history_path,
                                        val_accuracy=val_accuracy,
                                        val_loss=val_loss,
                                        selected_model=selected_model,
                                        models=MODELS)
        if file:
            try:
                if selected_model not in loaded_models:
                    return render_template_string(HTML_TEMPLATE,
                                                prediction=f"Error: Model {MODELS[selected_model]['display_name']} not available.",
                                                confidence="",
                                                image_path="",
                                                warning="Please train the model or select another.",
                                                classification_report=classification_report,
                                                confusion_matrix_path=confusion_matrix_path,
                                                training_history_path=training_history_path,
                                                val_accuracy=val_accuracy,
                                                val_loss=val_loss,
                                                selected_model=selected_model,
                                                models=MODELS)
                filename = secure_filename(file.filename)
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return render_template_string(HTML_TEMPLATE,
                                                prediction="Error: Invalid file format.",
                                                confidence="",
                                                image_path="",
                                                warning="Only .jpg, .jpeg, or .png files are allowed.",
                                                classification_report=classification_report,
                                                confusion_matrix_path=confusion_matrix_path,
                                                training_history_path=training_history_path,
                                                val_accuracy=val_accuracy,
                                                val_loss=val_loss,
                                                selected_model=selected_model,
                                                models=MODELS)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Preprocess and predict
                img_array = preprocess_image(filepath, selected_model)
                predictions = loaded_models[selected_model].predict(img_array, verbose=0)
                predicted_class_index = np.argmax(predictions, axis=1)[0]
                predicted_class = class_labels[predicted_class_index]
                confidence_score = predictions[0][predicted_class_index]

                prediction = f'Predicted Fish Category: {predicted_class}'
                confidence = f'Confidence: {confidence_score*100:.2f}%'
                image_path = f'Uploads/{filename}'

                # Warnings for low confidence, animal fish bass, or low accuracy
                if confidence_score < 0.5:
                    warning = f"Low confidence prediction with {MODELS[selected_model]['display_name']}. The model may be uncertain."
                if predicted_class == 'animal fish bass':
                    warning += " Note: The 'animal fish bass' category has low support (13 samples), which may affect reliability."
                if val_metrics[selected_model]['val_accuracy'] and val_metrics[selected_model]['val_accuracy'] < 0.5:
                    warning += f" Note: {MODELS[selected_model]['display_name']} has low validation accuracy ({val_metrics[selected_model]['val_accuracy']:.4f}), which may reduce reliability."

            except Exception as e:
                return render_template_string(HTML_TEMPLATE,
                                            prediction="Error: Failed to process image.",
                                            confidence="",
                                            image_path="",
                                            warning=f"Processing error: {str(e)}",
                                            classification_report=classification_report,
                                            confusion_matrix_path=confusion_matrix_path,
                                            training_history_path=training_history_path,
                                            val_accuracy=val_accuracy,
                                            val_loss=val_loss,
                                            selected_model=selected_model,
                                            models=MODELS)

    return render_template_string(HTML_TEMPLATE,
                                  prediction=prediction,
                                  confidence=confidence,
                                  image_path=image_path,
                                  warning=warning,
                                  classification_report=classification_report,
                                  confusion_matrix_path=confusion_matrix_path,
                                  training_history_path=training_history_path,
                                  val_accuracy=val_accuracy,
                                  val_loss=val_loss,
                                  selected_model=selected_model,
                                  models=MODELS)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Fish Species Classifier</title>
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
        <h1 class="text-3xl font-bold text-center mb-6 text-gray-800">Fish Species Classifier</h1>
        <p class="text-center text-gray-600 mb-8">Select a model and upload a fish image to predict its species.</p>

        <form method="post" enctype="multipart/form-data" class="flex flex-col items-center space-y-4">
            <select name="model" class="text-sm text-gray-500 border rounded-md p-2">
                {% for model_name, config in models.items() %}
                <option value="{{ model_name }}" {% if model_name == selected_model %}selected{% endif %}>
                    {{ config.display_name }}
                    {% if model_name in val_metrics and val_metrics[model_name].val_accuracy %}
                    (Accuracy: {{ val_metrics[model_name].val_accuracy|round(4) }})
                    {% endif %}
                </option>
                {% endfor %}
            </select>
            <input type="file" name="file" accept=".jpg,.jpeg,.png" class="text-sm text-gray-500
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
            {% if warning %}
            <p class="text-sm text-red-600 mb-4">{{ warning }}</p>
            {% endif %}
            {% if image_path %}
            <img src="{{ image_path }}" alt="Uploaded Image" class="mx-auto mt-4 rounded-lg shadow-sm" style="max-width: 300px; max-height: 300px;">
            {% endif %}
        </div>
        {% endif %}

        {% if classification_report or confusion_matrix_path or training_history_path or val_accuracy %}
        <div class="mt-8 p-6 bg-gray-50 rounded-lg">
            <h2 class="text-xl font-semibold mb-2 text-gray-700">Model Evaluation Metrics ({{ models[selected_model].display_name }})</h2>
            {% if val_accuracy %}
            <p class="text-sm text-gray-600 mb-2">Validation Accuracy: {{ val_accuracy|round(4) }}</p>
            <p class="text-sm text-gray-600 mb-2">Validation Loss: {{ val_loss|round(4) }}</p>
            {% endif %}
            {% if classification_report %}
            <h3 class="text-lg font-semibold mb-2 text-gray-600">Classification Report</h3>
            <pre class="text-sm text-gray-600 bg-white p-4 rounded-lg">{{ classification_report }}</pre>
            {% endif %}
            {% if confusion_matrix_path %}
            <h3 class="text-lg font-semibold mb-2 text-gray-600">Confusion Matrix</h3>
            <img src="{{ confusion_matrix_path }}" alt="Confusion Matrix" class="mx-auto rounded-lg shadow-sm" style="max-width: 500px;">
            {% endif %}
            {% if training_history_path %}
            <h3 class="text-lg font-semibold mb-2 text-gray-600">Training History</h3>
            <img src="{{ training_history_path }}" alt="Training History" class="mx-auto rounded-lg shadow-sm" style="max-width: 500px;">
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    @app.route('/<filename>')
    def static_file(filename):
        return send_from_directory('.', filename)

    @app.route('/Uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    app.run(debug=True)
