import os
import gdown
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import zipfile

# ==============================
# CONFIGURATION
# ==============================
DATASET_ZIP_ID = "11Q2g3uiotlX0pRDe-Vibn2z3XJLO_Hy_"
ZIP_PATH = "dataset.zip"

MODEL_LINKS = {
    "EfficientNetB0": "1UBfUF_kOWY_mnl9hm4BzRprBvERCb2kq",
    "CustomCNN": "1cH51G8OpMQxarIlH7Xie2tS2K1TQ6dNE",
    "VGG16": "1DhPJQD6Xs8NPkYdGrlV7G31-iS5aXVZf",
    "ResNet50": "1rmtdeJMbVYL4UoAK1BLcl-DRoe0mJwMs",
    "MobileNetV2": "1uJIIw6F2C3ARwSZeaJ9-eSijTtS9dOkT",
    "InceptionV3": "1K2C6mRcf6VJtO124rcvzn4OUHUtuji4x"
}

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

# ==============================
# DOWNLOAD & EXTRACT DATASET
# ==============================
if not os.path.exists(ZIP_PATH):
    print("Downloading dataset zip from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DATASET_ZIP_ID}", ZIP_PATH, quiet=False)

if not (os.path.exists("train") and os.path.exists("val") and os.path.exists("test")):
    print("Extracting dataset zip...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Dataset extracted successfully.")
else:
    print("Dataset folders already exist.")

train_dir = "train"
val_dir = "val"
test_dir = "test"

# ==============================
# IMAGE GENERATORS
# ==============================
datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = datagen.flow_from_directory(
    train_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
val_generator = datagen.flow_from_directory(
    val_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)
test_generator = datagen.flow_from_directory(
    test_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# ==============================
# DOWNLOAD MODELS
# ==============================
def download_model(model_name, file_id):
    output_path = f"{model_name}.h5"
    if not os.path.exists(output_path):
        print(f"Downloading {model_name} model...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{model_name} model already exists.")

for model_name, file_id in MODEL_LINKS.items():
    download_model(model_name, file_id)

# ==============================
# EVALUATION & PLOTTING
# ==============================
def plot_confusion_matrix(y_true, y_pred_classes, target_names, model_name):
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(plots_dir / f'{model_name}_confusion_matrix.png')
    plt.close()

evaluation_metrics = {}

for model_name in MODEL_LINKS.keys():
    try:
        model = load_model(f"{model_name}.h5")
        loss, acc = model.evaluate(test_generator, verbose=0)
        print(f"\n{model_name} - Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")

        y_pred = model.predict(test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes

        report = classification_report(
            y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys())
        )
        print(report)

        plot_confusion_matrix(y_true, y_pred_classes, list(test_generator.class_indices.keys()), model_name)

        evaluation_metrics[model_name] = acc
    except Exception as e:
        print(f"Error loading/evaluating {model_name}: {e}")

# ==============================
# MODEL COMPARISON PLOT
# ==============================
plt.figure(figsize=(10, 6))
sns.barplot(x=list(evaluation_metrics.keys()), y=list(evaluation_metrics.values()), palette="viridis")
plt.title("Model Comparison - Test Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
for i, acc in enumerate(evaluation_metrics.values()):
    plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center')
plt.tight_layout()
plt.savefig(plots_dir / "model_comparison.png")
plt.close()


