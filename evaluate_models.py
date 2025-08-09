import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from utils import get_data_generators, plot_confusion_matrix, plot_training_curves, plot_model_comparison
from pathlib import Path
import json

# ==============================================================================
# Configuration
# ==============================================================================
train_dir = "C:\\Users\\DELL\\Downloads\\Dataset\\images.cv_jzk6llhf18tm3k0kyttxz\\data\\train"
val_dir = "C:\\Users\\DELL\\Downloads\\Dataset\\images.cv_jzk6llhf18tm3k0kyttxz\\data\\val"
test_dir = "C:\\Users\\DELL\\Downloads\\Dataset\\images.cv_jzk6llhf18tm3k0kyttxz\\data\\test"
models_dir = "models"
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ==============================================================================
# Load data and models
# ==============================================================================
_, _, test_generator = get_data_generators(
    train_dir, val_dir, test_dir, IMAGE_SIZE, BATCH_SIZE
)

models_to_evaluate = {
    "CustomCNN": "cnn_model.h5",
    "VGG16": "vgg16_best_model.h5",
    "ResNet50": "resnet50_best_model.h5",
    "MobileNetV2": "mobilenetv2_best_model.h5",
    "InceptionV3": "inceptionv3_best_model.h5",
    "EfficientNetB0": "efficientnetb0_best_model.h5"
}

evaluation_metrics = {}

# ==============================================================================
# Evaluation Loop
# ==============================================================================
print(f"\n{'=' * 50}\nStarting Final Evaluation on Test Set\n{'=' * 50}\n")
for name, filename in models_to_evaluate.items():
    model_path = os.path.join(models_dir, filename)
    if not os.path.exists(model_path):
        print(f"Skipping evaluation for {name}: model file not found at {model_path}")
        continue

    print(f"Loading and evaluating {name} from {model_path}")
    model = tf.keras.models.load_model(model_path)

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

    print(f"\nFinal Evaluation for {name}:")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys()))
    print(report)

    # Plot and save confusion matrix
    plot_confusion_matrix(y_true, y_pred_classes, list(test_generator.class_indices.keys()), name, plots_dir)
    print(f"Confusion matrix for {name} saved to plots/{name.lower()}_confusion_matrix.png")

    # Store evaluation metrics
    metrics_dict = classification_report(y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys()),
                                         output_dict=True)
    evaluation_metrics[name] = {
        'accuracy': test_accuracy,
        'precision': metrics_dict['macro avg']['precision'],
        'recall': metrics_dict['macro avg']['recall'],
        'f1-score': metrics_dict['macro avg']['f1-score']
    }

# Finally, plot the comparison table
plot_model_comparison(evaluation_metrics, plots_dir)
print("\nModel comparison bar chart saved to plots/model_comparison_table.png")

# Save metrics to a file
with open(os.path.join(plots_dir, 'evaluation_metrics.json'), 'w') as f:
    json.dump(evaluation_metrics, f, indent=4)
print("\nEvaluation metrics saved to plots/evaluation_metrics.json")
