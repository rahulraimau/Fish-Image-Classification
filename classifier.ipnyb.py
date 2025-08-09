#%%
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
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
# 1. Setup and Configuration
# ==============================================================================
# Define the paths to your dataset directories.
# IMPORTANT: Update these paths to match your local file structure.
train_dir = r"C:\Users\DELL\PycharmProjects\multi_fish_classifier\train"
val_dir = r"C:\Users\DELL\PycharmProjects\multi_fish_classifier\val"
test_dir = r"C:\Users\DELL\PycharmProjects\multi_fish_classifier\test"

# Define model parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 11

# The number of epochs for each training stage
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# The number of layers to unfreeze for fine-tuning.
# A higher number means more layers of the base model are trained.
FINE_TUNE_LAYERS = 25

# Verify that the directories exist
if not os.path.exists(train_dir) or not os.path.exists(val_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError(
        "One or more of the dataset directories were not found. "
        "Please update the 'train_dir', 'val_dir', and 'test_dir' variables "
        "at the top of the script with the correct paths."
    )

# Create a directory to save the plots
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

# ==============================================================================
# 2. Data Preprocessing and Augmentation
# ==============================================================================
# Use ImageDataGenerator to load and preprocess images.
# All images will be rescaled by 1/255.
# Data augmentation is applied ONLY to the training data to prevent overfitting.
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

# The validation and test data should NOT be augmented, only rescaled.
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators from the directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Keep data in order for evaluation
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Keep data in order for evaluation
)

# Dictionaries to hold the training history, trained models, and evaluation metrics
histories = {}
trained_models = {}
evaluation_metrics = {}

# ==============================================================================
# 3. Transfer Learning with Fine-Tuning Function
# ==============================================================================
def build_and_train_model(base_model_name, base_model_class):
    """
    Builds a transfer learning model, trains it in two stages (head training
    and fine-tuning), and saves the best model during training.
    """
    print("\n" + "="*50)
    print(f"Building and Training {base_model_name} Model")
    print("="*50)

    # 3.1 Load the pre-trained base model
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=IMAGE_SIZE + (3,)
    )

    # 3.2 Add a new classification head on top of the base model
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 3.3 Stage 1: Train the new classification head only
    # Freeze all layers in the base model
    base_model.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    print("\n--- Starting Stage 1: Training the top layers ---")
    history_initial = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=INITIAL_EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE
    )

    # 3.4 Stage 2: Fine-tuning
    # Unfreeze a portion of the base model's layers for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.00001), # Use a very low learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    print("\n--- Starting Stage 2: Fine-tuning the base model ---")
    # Define a ModelCheckpoint callback to save the best model
    checkpoint_filepath = f'best_model_{base_model_name}.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False, # Save the entire model
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

    # Combine the histories for plotting
    history_combined = {
        'accuracy': history_initial.history['accuracy'] + history_fine_tune.history['accuracy'],
        'val_accuracy': history_initial.history['val_accuracy'] + history_fine_tune.history['val_accuracy'],
        'loss': history_initial.history['loss'] + history_fine_tune.history['loss'],
        'val_loss': history_initial.history['val_loss'] + history_fine_tune.history['val_loss']
    }

    return model, history_combined

# ==============================================================================
# 4. Plotting Functions
# ==============================================================================
def plot_training_curves(history, model_name):
    """
    Plots the training and validation accuracy and loss curves and saves them.
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
    plt.close() # Close the plot to free up memory

def plot_confusion_matrix(y_true, y_pred_classes, target_names, model_name):
    """
    Computes and plots the confusion matrix, then saves it as a PNG file.
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

def plot_model_comparison(metrics):
    """
    Creates and saves a bar chart comparing the final test accuracies of all models.
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
# 5. Main Execution Loop
# ==============================================================================
models_to_train = {
    "VGG16": VGG16,
    "ResNet50": ResNet50,
    "MobileNetV2": MobileNetV2,
    "InceptionV3": InceptionV3,
    "EfficientNetB0": EfficientNetB0
}

# Loop through each model and train it
for name, model_class in models_to_train.items():
    model, history = build_and_train_model(name, model_class)
    trained_models[name] = model
    histories[name] = history

    # After training, plot and save the training curves
    plot_training_curves(history, name)

    print(f"\nTraining curves for {name} saved to plots/{name.lower()}_training_curve.png")

# ==============================================================================
# 6. Model Evaluation and Comparison
# ==============================================================================
print("\n" + "="*50)
print("Starting Final Evaluation on Test Set")
print("="*50)

# Run the evaluation for each trained model
for name, model in trained_models.items():
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

    print(f"\nFinal Evaluation for {name}:")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
and true labels for detailed report an
    #     # Get predictions d confusion matrix
y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys()))
    print(report)

    # Plot and save confusion matrix
    plot_confusion_matrix(y_true, y_pred_classes, list(test_generator.class_indices.keys()), name)
    print(f"Confusion matrix for {name} saved to plots/{name.lower()}_confusion_matrix.png")

    # Store evaluation metrics for later comparison
    metrics_dict = classification_report(y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys()), output_dict=True)
    evaluation_metrics[name] = {
        'accuracy': test_accuracy,
        'precision': metrics_dict['macro avg']['precision'],
        'recall': metrics_dict['macro avg']['recall'],
        'f1-score': metrics_dict['macro avg']['f1-score']
    }

# Finally, plot the comparison table
plot_model_comparison(evaluation_metrics)
print("\nModel comparison bar chart saved to plots/model_comparison_table.png")
#%%

#%%
