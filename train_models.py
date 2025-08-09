import os
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import get_data_generators, create_transfer_learning_model, create_custom_cnn_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0
)
import tensorflow as tf

# ==============================================================================
# Configuration
# ==============================================================================
train_dir = "C:\\Users\\DELL\\Downloads\\Dataset\\images.cv_jzk6llhf18tm3k0kyttxz\\data\\train"
val_dir = "C:\\Users\\DELL\\Downloads\\Dataset\\images.cv_jzk6llhf18tm3k0kyttxz\\data\\val"
test_dir = "C:\\Users\\DELL\\Downloads\\Dataset\\images.cv_jzk6llhf18tm3k0kyttxz\\data\\test"
models_dir = "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 11

INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
FINE_TUNE_LAYERS = 25

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Define models to train
models_to_train = {
    "CustomCNN": None,
    "VGG16": VGG16,
    "ResNet50": ResNet50,
    "MobileNetV2": MobileNetV2,
    "InceptionV3": InceptionV3,
    "EfficientNetB0": EfficientNetB0
}


# ==============================================================================
# Training Function
# ==============================================================================
def train_model(model_name: str, model_class: tf.keras.Model = None):
    """
    Builds, trains, and saves a single model.

    Args:
        model_name (str): The name of the model to train (e.g., "VGG16").
        model_class (tf.keras.Model, optional): The Keras application class for
                                                transfer learning. Defaults to None for CustomCNN.
    """
    print(f"\n{'=' * 50}\nBuilding and Training {model_name} Model\n{'=' * 50}\n")

    # Custom CNN model training
    if model_name == "CustomCNN":
        model = create_custom_cnn_model(IMAGE_SIZE + (3,), NUM_CLASSES)

        checkpoint_filepath = os.path.join(models_dir, 'cnn_model.h5')
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )

        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            epochs=TOTAL_EPOCHS,
            validation_data=val_generator,
            validation_steps=val_generator.samples // BATCH_SIZE,
            callbacks=[model_checkpoint_callback]
        )

    # Transfer learning model training
    else:
        model, base_model = create_transfer_learning_model(
            model_class, IMAGE_SIZE, NUM_CLASSES
        )

        # Stage 1: Train the new classification head only
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("\n--- Starting Stage 1: Training the top layers ---")
        history_initial = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            epochs=INITIAL_EPOCHS,
            validation_data=val_generator,
            validation_steps=val_generator.samples // BATCH_SIZE
        )

        # Stage 2: Fine-tuning
        # Unfreeze a portion of the base model's layers
        base_model.trainable = True
        for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
            layer.trainable = False

        model.compile(optimizer=Adam(learning_rate=0.00001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("\n--- Starting Stage 2: Fine-tuning the base model ---")
        checkpoint_filepath = os.path.join(models_dir, f'{model_name.lower()}_best_model.h5')
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )

        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            epochs=TOTAL_EPOCHS,
            initial_epoch=history_initial.epoch[-1],
            validation_data=val_generator,
            validation_steps=val_generator.samples // BATCH_SIZE,
            callbacks=[model_checkpoint_callback]
        )


if __name__ == "__main__":
    try:
        # Load data generators once for all models
        train_generator, val_generator, _ = get_data_generators(
            train_dir, val_dir, test_dir, IMAGE_SIZE, BATCH_SIZE
        )

        for name, model_class in models_to_train.items():
            train_model(name, model_class)

        print("\nTraining complete. Best models saved to the 'models/' directory.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that your data directories exist and are correctly specified.")