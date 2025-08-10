import os
import numpy as np
import gdown
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

# Google Drive file IDs for each model
MODEL_DRIVE_IDS = {
    'inceptionv3': '1K2C6mRcf6VJtO124rcvzn4OUHUtuji4x',
    'custom_cnn': '1cH51G8OpMQxarIlH7Xie2tS2K1TQ6dNE',
    'vgg16': '1DhPJQD6Xs8NPkYdGrlV7G31-iS5aXVZf',
    'resnet50': '1rmtdeJMbVYL4UoAK1BLcl-DRoe0mJwMs',
    'mobilenetv2': '1uJIIw6F2C3ARwSZeaJ9-eSijTtS9dOkT',
    'efficientnetb0': '1WYpyrr7ggtfylOvfP_axoNN5FbmlxFkv'
}

# Model configurations
MODELS = {
    'inceptionv3': {
        'path': r'models/best_model_InceptionV3.h5',
        'preprocess': inceptionv3_preprocess,
        'history': 'inceptionv3_history.pkl',
        'report': 'inceptionv3_classification_report.txt',
        'confusion_matrix': 'inceptionv3_confusion_matrix.png',
        'training_history': 'inceptionv3_training_history.png',
        'display_name': 'InceptionV3'
    },
    'custom_cnn': {
        'path': r'models/custom_cnn_best.h5',
        'preprocess': lambda x: x / 255.0,
        'history': 'custom_cnn_history.pkl',
        'report': 'custom_cnn_classification_report.txt',
        'confusion_matrix': 'custom_cnn_confusion_matrix.png',
        'training_history': 'custom_cnn_training_history.png',
        'display_name': 'Custom CNN'
    },
    'vgg16': {
        'path': r'models/best_model_VGG16.h5',
        'preprocess': vgg16_preprocess,
        'history': 'vgg16_history.pkl',
        'report': 'vgg16_classification_report.txt',
        'confusion_matrix': 'vgg16_confusion_matrix.png',
        'training_history': 'vgg16_training_history.png',
        'display_name': 'VGG16'
    },
    'resnet50': {
        'path': r'models/best_model_ResNet50.h5',
        'preprocess': resnet50_preprocess,
        'history': 'resnet50_history.pkl',
        'report': 'resnet50_classification_report.txt',
        'confusion_matrix': 'resnet50_confusion_matrix.png',
        'training_history': 'resnet50_training_history.png',
        'display_name': 'ResNet50'
    },
    'mobilenetv2': {
        'path': r'models/best_model_MobileNetV2.h5',
        'preprocess': mobilenetv2_preprocess,
        'history': 'mobilenetv2_history.pkl',
        'report': 'mobilenetv2_classification_report.txt',
        'confusion_matrix': 'mobilenetv2_confusion_matrix.png',
        'training_history': 'mobilenetv2_training_history.png',
        'display_name': 'MobileNetV2'
    },
    'efficientnetb0': {
        'path': r'models/efficientnetb0_best.h5',
        'preprocess': efficientnet_preprocess,
        'history': 'efficientnetb0_history.pkl',
        'report': 'efficientnetb0_classification_report.txt',
        'confusion_matrix': 'efficientnetb0_confusion_matrix.png',
        'training_history': 'efficientnetb0_training_history.png',
        'display_name': 'EfficientNetB0'
    }
}

# Function to download a model if it's missing
def ensure_model_file(model_name, file_path):
    if not os.path.exists(file_path):
        if model_name in MODEL_DRIVE_IDS:
            file_id = MODEL_DRIVE_IDS[model_name]
            url = f"https://drive.google.com/uc?id={file_id}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            print(f"📥 Downloading {model_name} model from Google Drive...")
            gdown.download(url, file_path, quiet=False)
        else:
            print(f"No Drive ID found for {model_name}, skipping download.")

# Ensure all model files exist before loading
for m_name, cfg in MODELS.items():
    ensure_model_file(m_name, cfg['path'])

# Load models and validation metrics
loaded_models = {}
val_metrics = {}
for model_name, config in MODELS.items():
    if os.path.exists(config['path']):
        loaded_models[model_name] = tf.keras.models.load_model(config['path'])
    else:
        print(f"⚠️ Warning: Model file {config['path']} not found.")
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

# (rest of your existing Flask routes and HTML_TEMPLATE remain unchanged)

if __name__ == '__main__':
    @app.route('/<filename>')
    def static_file(filename):
        return send_from_directory('.', filename)

    @app.route('/Uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    app.run(debug=True)


