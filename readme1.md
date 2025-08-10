Found 1092 images belonging to 11 classes.
2025-08-10 09:55:22.378356: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. model.compile_metrics will be empty until you train or evaluate the model.

Evaluating CUSTOM_CNN...
C:\Users\DELL\PycharmProjects\multi_fish_classifier\.venv\Lib\site-packages\keras\src\trainers\data_adapters\py_dataset_adapter.py:121: UserWarning: Your PyDataset class should call super().__init__(**kwargs) in its constructor. **kwargs can include workers, use_multiprocessing, max_queue_size. Do not pass these arguments to fit(), as they will be ignored.
  self._warn_if_super_not_called()
Accuracy for CUSTOM_CNN: 0.7592
Confusion Matrix:
[[125  60   0   0   0   0   1   1   0   0   0]
 [  4   6   0   0   0   0   0   0   0   0   0]
 [  0   0 100   0   0   0   0   5   0   0   0]
 [  0   0   0  45   7   0  14  14   0   4  10]
 [  0   0  16   3  64   0   0   8   0   6   0]
 [  0   0   0   0   0  87   0   0   3   0   0]
 [  0   0   0   8   7   0  94   2   0   1   1]
 [  0   0   9   9  14   0   4  56   0   4   1]
 [  0   0   0   0   0   3   0   0  96   1   0]
 [  0   0   0   0   3   4   0   2  30  62   0]
 [  0   0   0   1   0   0   3   0   0   0  94]]
Classification Report:
                                  precision    recall  f1-score   support

                     animal fish       0.97      0.67      0.79       187
                animal fish bass       0.09      0.60      0.16        10
   fish sea_food black_sea_sprat       0.80      0.95      0.87       105
   fish sea_food gilt_head_bream       0.68      0.48      0.56        94
   fish sea_food hourse_mackerel       0.67      0.66      0.67        97
        fish sea_food red_mullet       0.93      0.97      0.95        90
     fish sea_food red_sea_bream       0.81      0.83      0.82       113
          fish sea_food sea_bass       0.64      0.58      0.61        97
            fish sea_food shrimp       0.74      0.96      0.84       100
fish sea_food striped_red_mullet       0.79      0.61      0.69       101
             fish sea_food trout       0.89      0.96      0.92        98

                        accuracy                           0.76      1092
                       macro avg       0.73      0.75      0.72      1092
                    weighted avg       0.80      0.76      0.77      1092

WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. model.compile_metrics will be empty until you train or evaluate the model.

Evaluating VGG16...
Accuracy for VGG16: 0.9844
Confusion Matrix:
[[185   0   0   0   0   0   0   1   0   1   0]
 [ 10   0   0   0   0   0   0   0   0   0   0]
 [  0   0 105   0   0   0   0   0   0   0   0]
 [  0   0   0  92   0   0   2   0   0   0   0]
 [  0   0   0   0  97   0   0   0   0   0   0]
 [  0   0   0   0   0  87   0   0   0   3   0]
 [  0   0   0   0   0   0 113   0   0   0   0]
 [  0   0   0   0   0   0   0  97   0   0   0]
 [  0   0   0   0   0   0   0   0 100   0   0]
 [  0   0   0   0   0   0   0   0   0 101   0]
 [  0   0   0   0   0   0   0   0   0   0  98]]
C:\Users\DELL\PycharmProjects\multi_fish_classifier\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use zero_division parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\DELL\PycharmProjects\multi_fish_classifier\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use zero_division parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\DELL\PycharmProjects\multi_fish_classifier\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use zero_division parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Classification Report:
                                  precision    recall  f1-score   support

                     animal fish       0.95      0.99      0.97       187
                animal fish bass       0.00      0.00      0.00        10
   fish sea_food black_sea_sprat       1.00      1.00      1.00       105
   fish sea_food gilt_head_bream       1.00      0.98      0.99        94
   fish sea_food hourse_mackerel       1.00      1.00      1.00        97
        fish sea_food red_mullet       1.00      0.97      0.98        90
     fish sea_food red_sea_bream       0.98      1.00      0.99       113
          fish sea_food sea_bass       0.99      1.00      0.99        97
            fish sea_food shrimp       1.00      1.00      1.00       100
fish sea_food striped_red_mullet       0.96      1.00      0.98       101
             fish sea_food trout       1.00      1.00      1.00        98

                        accuracy                           0.98      1092
                       macro avg       0.90      0.90      0.90      1092
                    weighted avg       0.98      0.98      0.98      1092

WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. model.compile_metrics will be empty until you train or evaluate the model.

Evaluating RESNET50...
C:\Users\DELL\PycharmProjects\multi_fish_classifier\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use zero_division parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\DELL\PycharmProjects\multi_fish_classifier\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use zero_division parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\DELL\PycharmProjects\multi_fish_classifier\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use zero_division parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Accuracy for RESNET50: 0.5714
Confusion Matrix:
[[169   0   2   2   2   0   5   1   3   0   3]
 [  9   0   0   0   0   0   0   0   0   0   1]
 [  0   0  53   0  38   3   2   6   0   1   2]
 [ 12   0   1  10   9   0  11   1   0   1  49]
 [  0   0   6   0  78   0   1   2   0   0  10]
 [  7   0   3   0  24  31   3   4  15   1   2]
 [  1   0   0   2  14   1  61   0   1   1  32]
 [  0   0  12   4  13   7   0  38   0   2  21]
 [  1   0   0   0   9   1   0   0  89   0   0]
 [  0   0  13   3  18   5   6   7  16  11  22]
 [  0   0   0   1   8   0   2   3   0   0  84]]
Classification Report:
                                  precision    recall  f1-score   support

                     animal fish       0.85      0.90      0.88       187
                animal fish bass       0.00      0.00      0.00        10
   fish sea_food black_sea_sprat       0.59      0.50      0.54       105
   fish sea_food gilt_head_bream       0.45      0.11      0.17        94
   fish sea_food hourse_mackerel       0.37      0.80      0.50        97
        fish sea_food red_mullet       0.65      0.34      0.45        90
     fish sea_food red_sea_bream       0.67      0.54      0.60       113
          fish sea_food sea_bass       0.61      0.39      0.48        97
            fish sea_food shrimp       0.72      0.89      0.79       100
fish sea_food striped_red_mullet       0.65      0.11      0.19       101
             fish sea_food trout       0.37      0.86      0.52        98

                        accuracy                           0.57      1092
                       macro avg       0.54      0.50      0.47      1092
                    weighted avg       0.61      0.57      0.54      1092

WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. model.compile_metrics will be empty until you train or evaluate the model.

Evaluating MOBILENETV2...
Accuracy for MOBILENETV2: 0.9698
Confusion Matrix:
[[180   0   2   0   0   0   0   4   1   0   0]
 [  9   1   0   0   0   0   0   0   0   0   0]
 [  0   0 103   0   0   0   0   0   0   2   0]
 [  0   0   0  92   0   0   2   0   0   0   0]
 [  0   0   1   0  95   0   0   0   0   1   0]
 [  0   0   0   0   0  84   0   0   0   6   0]
 [  0   0   0   1   0   0 112   0   0   0   0]
 [  0   0   0   0   0   0   0  97   0   0   0]
 [  0   0   0   0   0   0   0   0 100   0   0]
 [  0   0   1   0   0   1   0   1   0  98   0]
 [  0   0   0   0   0   0   0   1   0   0  97]]
Classification Report:
                                  precision    recall  f1-score   support

                     animal fish       0.95      0.96      0.96       187
                animal fish bass       1.00      0.10      0.18        10
   fish sea_food black_sea_sprat       0.96      0.98      0.97       105
   fish sea_food gilt_head_bream       0.99      0.98      0.98        94
   fish sea_food hourse_mackerel       1.00      0.98      0.99        97
        fish sea_food red_mullet       0.99      0.93      0.96        90
     fish sea_food red_sea_bream       0.98      0.99      0.99       113
          fish sea_food sea_bass       0.94      1.00      0.97        97
            fish sea_food shrimp       0.99      1.00      1.00       100
fish sea_food striped_red_mullet       0.92      0.97      0.94       101
             fish sea_food trout       1.00      0.99      0.99        98

                        accuracy                           0.97      1092
                       macro avg       0.97      0.90      0.90      1092
                    weighted avg       0.97      0.97      0.97      1092

WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. model.compile_metrics will be empty until you train or evaluate the model.

Evaluating INCEPTIONV3...
Accuracy for INCEPTIONV3: 0.9460
Confusion Matrix:
[[185   0   0   0   0   1   0   0   0   0   1]
 [ 10   0   0   0   0   0   0   0   0   0   0]
 [  0   0  99   0   3   0   0   1   0   2   0]
 [  0   0   0  90   0   0   1   1   0   0   2]
 [  0   0   0   0  96   0   0   0   0   1   0]
 [  0   0   0   0   0  89   0   0   0   1   0]
 [  0   0   0   8   0   0 105   0   0   0   0]
 [  0   0   1   1   0   0   0  90   0   0   5]
 [  0   0   0   0   0   0   0   0 100   0   0]
 [  0   0   3   0   2  14   0   0   0  82   0]
 [  1   0   0   0   0   0   0   0   0   0  97]]
Classification Report:
                                  precision    recall  f1-score   support

                     animal fish       0.94      0.99      0.97       187
                animal fish bass       0.00      0.00      0.00        10
   fish sea_food black_sea_sprat       0.96      0.94      0.95       105
   fish sea_food gilt_head_bream       0.91      0.96      0.93        94
C:\Users\DELL\PycharmProjects\multi_fish_classifier\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use zero_division parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\DELL\PycharmProjects\multi_fish_classifier\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use zero_division parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\DELL\PycharmProjects\multi_fish_classifier\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use zero_division parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
   fish sea_food hourse_mackerel       0.95      0.99      0.97        97
        fish sea_food red_mullet       0.86      0.99      0.92        90
     fish sea_food red_sea_bream       0.99      0.93      0.96       113
          fish sea_food sea_bass       0.98      0.93      0.95        97
            fish sea_food shrimp       1.00      1.00      1.00       100
fish sea_food striped_red_mullet       0.95      0.81      0.88       101
             fish sea_food trout       0.92      0.99      0.96        98

                        accuracy                           0.95      1092
                       macro avg       0.86      0.87      0.86      1092
                    weighted avg       0.94      0.95      0.94      1092

WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. model.compile_metrics will be empty until you train or evaluate the model.

Evaluating EFFICIENTNETB0...
Accuracy for EFFICIENTNETB0: 0.1987
Confusion Matrix:
[[134   0   0  10   0   0   0   0   0   0  43]
 [  9   0   0   0   0   0   0   0   0   0   1]
 [ 55   0   0  17   0   0   0   0   0   0  33]
 [ 40   0   0  15   0   0   0   0   0   0  39]
 [ 35   0   0  24   0   0   0   0   0   0  38]
 [ 59   0   0  14   0   0   0   0   0   0  17]
 [ 46   0   0  23   0   0   0   0   0   0  44]
 [ 54   0   0  23   0   0   0   0   0   0  20]
 [ 95   0   0   5   0   0   0   0   0   0   0]
 [ 58   0   0  13   0   0   0   0   0   0  30]
 [ 21   0   0   9   0   0   0   0   0   0  68]]
Classification Report:
                                  precision    recall  f1-score   support

                     animal fish       0.22      0.72      0.34       187
                animal fish bass       0.00      0.00      0.00        10
   fish sea_food black_sea_sprat       0.00      0.00      0.00       105
   fish sea_food gilt_head_bream       0.10      0.16      0.12        94
   fish sea_food hourse_mackerel       0.00      0.00      0.00        97
        fish sea_food red_mullet       0.00      0.00      0.00        90
     fish sea_food red_sea_bream       0.00      0.00      0.00       113
          fish sea_food sea_bass       0.00      0.00      0.00        97
            fish sea_food shrimp       0.00      0.00      0.00       100
fish sea_food striped_red_mullet       0.00      0.00      0.00       101
             fish sea_food trout       0.20      0.69      0.32        98

                        accuracy                           0.20      1092
                       macro avg       0.05      0.14      0.07      1092
                    weighted avg       0.06      0.20      0.10      1092

C:\Users\DELL\PycharmProjects\multi_fish_classifier\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use zero_division parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\DELL\PycharmProjects\multi_fish_classifier\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use zero_division parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\DELL\PycharmProjects\multi_fish_classifier\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use zero_division parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Evaluation completed for all models.

Process finished with exit code 0

Performance Summary (Accuracy)
Model	Accuracy
VGG16	98.44%
MobileNetV2	96.98%
InceptionV3	94.60%
Custom CNN	75.92%
ResNet50	57.14%
EfficientNetB0	19.87%

Observations
VGG16 is the clear winner – Extremely high accuracy, but it’s missing predictions for “animal fish bass” (precision = 0). This suggests class imbalance or overfitting for rare classes.

MobileNetV2 is very close in accuracy to VGG16, but it does make predictions for all classes (though “animal fish bass” recall is still low at 10%).

InceptionV3 performs well but struggles with rare classes like “animal fish bass” and “striped red mullet.”

Custom CNN is far behind transfer learning models, showing the advantage of pre-trained feature extractors.

ResNet50 struggles here—possibly due to wrong preprocessing (image size, normalization) or inadequate fine-tuning.

EfficientNetB0 completely underperforms—likely due to wrong input preprocessing (needs tf.keras.applications.efficientnet.preprocess_input) or model architecture mismatch.

Why “animal fish bass” is so bad for many models
Only 10 images in that class.

Many models completely fail to learn features from such a small sample → they predict other similar-looking fish.

VGG16 & MobileNetV2 get almost everything else right, so this class imbalance becomes more visible.

What you should try next
Class imbalance fixes:

Data augmentation only for low-sample classes.

Class weights during training (class_weight in model.fit).

Oversampling rare classes.

Model choice for deployment:

MobileNetV2 for speed + good accuracy.

VGG16 for maximum accuracy if speed is less critical.

ResNet50 & EfficientNetB0 debugging:

Make sure images are resized correctly (ResNet50 expects 224×224, EfficientNetB0 expects 224×224 but with its own preprocessing).

Apply correct preprocessing functions:

python
Copy
Edit
from tensorflow.keras.applications import efficientnet
x = efficientnet.preprocess_input(x)
Try fine-tuning last few layers instead of only using as fixed feature extractor.

Evaluation improvement:

Use Top-3 accuracy to measure whether the model is “almost” correct in rare classes.

Analyze misclassified images visually.
