import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# ==============================================================================
# Data and Plotting Utilities
# ==============================================================================

def get_data_generators(train_dir: str, val_dir: str, test_dir: str, image_size: tuple, batch_size: int):
    """
    Creates and returns data generators for training, validation, and testing.

    This function performs data validation to ensure the specified directories exist
    and then configures ImageDataGenerator for data augmentation and rescaling.

    Args:
        train_dir (str): Path to the training data directory.
        val_dir (str): Path to the validation data directory.
        test_dir (str): Path to the test data directory.
        image_size (tuple): The target size for images (e.g., (224, 224)).
        batch_size (int): The number of images per batch.

    Returns:
        tuple: A tuple containing (train_generator, val_generator, test_generator).

    Raises:
        FileNotFoundError: If any of the data directories do not exist.
    """
    # Data validation: Ensure all directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Rescaling only for validation and test sets
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Create data generators from the directories
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return train_generator, val_generator, test_generator


def plot_training_curves(history: dict, model_name: str, plots_dir: Path):
    """
    Plots the training and validation accuracy and loss curves and saves them
    to a file.

    Args:
        history (dict): The history object from Keras model training.
        model_name (str): The name of the model for the plot title and filename.
        plots_dir (Path): The Path object for the directory to save the plots.
    """
    plt.figure(figsize=(12, 4))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / f'{model_name.lower()}_training_curve.png')
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred_classes: np.ndarray, target_names: list, model_name: str,
                          plots_dir: Path):
    """
    Computes and plots the confusion matrix, then saves it as a PNG file.

    Args:
        y_true (np.ndarray): The true class labels.
        y_pred_classes (np.ndarray): The predicted class labels.
        target_names (list): A list of class names.
        model_name (str): The name of the model for the plot title and filename.
        plots_dir (Path): The Path object for the directory to save the plots.
    """
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(plots_dir / f'{model_name.lower()}_confusion_matrix.png')
    plt.close()


def plot_model_comparison(metrics: dict, plots_dir: Path):
    """
    Creates and saves a bar chart comparing the final test accuracies of all models.

    Args:
        metrics (dict): A dictionary containing model names as keys and their
                        evaluation metrics as values.
        plots_dir (Path): The Path object for the directory to save the plots.
    """
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
# Model Building Functions
# ==============================================================================

def create_custom_cnn_model(input_shape: tuple, num_classes: int) -> Sequential:
    """
    Defines and compiles a custom CNN architecture from scratch.

    Args:
        input_shape (tuple): The shape of the input images (e.g., (224, 224, 3)).
        num_classes (int): The number of output classes.

    Returns:
        Sequential: A compiled Keras Sequential model.
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Flatten and Dense Layers for Classification
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),  # Add dropout for regularization
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_transfer_learning_model(base_model_class: tf.keras.Model, image_size: tuple, num_classes: int) -> tuple:
    """
    Builds a transfer learning model with a pre-trained base and a custom head.

    This function creates a new model on top of a frozen pre-trained base model.
    The base model's layers are initially set to non-trainable.

    Args:
        base_model_class (tf.keras.Model): The Keras application class to use
                                          (e.g., VGG16, ResNet50).
        image_size (tuple): The target size for images (e.g., (224, 224)).
        num_classes (int): The number of output classes.

    Returns:
        tuple: A tuple containing the full model and the base model instance.
    """
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=image_size + (3,)
    )

    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model for the first training stage
    base_model.trainable = False

    return model, base_model
