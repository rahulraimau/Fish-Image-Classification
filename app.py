import os
import gdown
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0
)
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pathlib import Path

# ==============================================================================
# 0. Google Drive Model Download
# ==============================================================================
drive_links = {
    "EfficientNetB0": "1UBfUF_kOWY_mnl9hm4BzRprBvERCb2kq",
    "CustomCNN": "1cH51G8OpMQxarIlH7Xie2tS2K1TQ6dNE",
    "VGG16": "1DhPJQD6Xs8NPkYdGrlV7G31-iS5aXVZf",
    "ResNet50": "1rmtdeJMbVYL4UoAK1BLcl-DRoe0mJwMs",
    "MobileNetV2": "1uJIIw6F2C3ARwSZeaJ9-eSijTtS9dOkT",
    "InceptionV3": "1K2C6mRcf6VJtO124rcvzn4OUHUtuji4x"
}

def download_model_from_drive(model_name, file_id):
    output_path = f"{model_name}.h5"
    if not os.path.exists(output_path):
        print(f"Downloading {model_name} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        if os.path.getsize(output_path) < 100000:
            raise OSError(f"{model_name} download failed — file too small, likely HTML error page.")
    else:
        print(f"{model_name} already exists, skipping download.")

for model_name, file_id in drive_links.items():
    download_model_from_drive(model_name, file_id)

# ==============================================================================
# 1. Setup and Configuration
# ==============================================================================
train_dir = r"C:\Users\DELL\PycharmProjects\multi_fish_classifier\train"
val_dir = r"C:\Users\DELL\PycharmProjects\multi_fish_classifier\val"
test_dir = r"C:\Users\DELL\PycharmProjects\multi_fish_classifier\test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 11
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
FINE_TUNE_LAYERS = 25

if not os.path.exists(train_dir) or not os.path.exists(val_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError("Dataset directories not found. Update paths in script.")

plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

# ==============================================================================
# 2. Data Preprocessing
# ==============================================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

histories = {}
trained_models = {}
evaluation_metrics = {}

# ==============================================================================
# 3. Build & Train Model Function
# ==============================================================================
def build_and_train_model(base_model_name, base_model_class):
    print("\n" + "="*50)
    print(f"Building and Training {base_model_name} Model")
    print("="*50)

    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + (3,))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    base_model.trainable = False
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    history_initial = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=INITIAL_EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE
    )

    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_filepath = f'best_model_{base_model_name}.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    history_fine_tune = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history_initial.epoch[-1],
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        callbacks=[model_checkpoint_callback]
    )

    history_combined = {
        'accuracy': history_initial.history['accuracy'] + history_fine_tune.history['accuracy'],
        'val_accuracy': history_initial.history['val_accuracy'] + history_fine_tune.history['val_accuracy'],
        'loss': history_initial.history['loss'] + history_fine_tune.history['loss'],
        'val_loss': history_initial.history['val_loss'] + history_fine_tune.history['val_loss']
    }

    return model, history_combined

# ==============================================================================
# 4. Plotting
# ==============================================================================
def plot_training_curves(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / f'{model_name.lower()}_training_curve.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred_classes, target_names, model_name):
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(plots_dir / f'{model_name.lower()}_confusion_matrix.png')
    plt.close()

def plot_model_comparison(metrics):
    model_names = list(metrics.keys())
    accuracies = [metrics[name]['accuracy'] for name in model_names]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=model_names, y=accuracies, palette='viridis')
    plt.title('Model Comparison: Test Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1.0)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig(plots_dir / 'model_comparison_table.png')
    plt.close()

# ==============================================================================
# 5. Evaluate Pre-trained Models (from Drive)
# ==============================================================================
for model_name in drive_links.keys():
    model_path = f"{model_name}.h5"
    print(f"Loading {model_name} from {model_path}")
    model = load_model(model_path)
    trained_models[model_name] = model

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    print(f"{model_name} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    report = classification_report(y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys()))
    print(report)

    plot_confusion_matrix(y_true, y_pred_classes, list(test_generator.class_indices.keys()), model_name)

    metrics_dict = classification_report(
        y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys()), output_dict=True
    )
    evaluation_metrics[model_name] = {
        'accuracy': test_accuracy,
        'precision': metrics_dict['macro avg']['precision'],
        'recall': metrics_dict['macro avg']['recall'],
        'f1-score': metrics_dict['macro avg']['f1-score']
    }

plot_model_comparison(evaluation_metrics)
print("\nModel comparison chart saved.")


