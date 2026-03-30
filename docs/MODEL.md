# Model Documentation - Flower Species Classifier

## Overview

This project implements two machine learning models for classifying flower species:
1. **Iris Classifier** - Traditional ML with scikit-learn (Iris dataset)
2. **CNN Flower Classifier** - Deep learning CNN with TensorFlow/Keras

---

## 1. Iris Species Classifier (`Iris_model.py`)

### Dataset
- **Source**: UCI Iris Dataset (Fisher's Iris dataset)
- **Classes**: Setosa, Versicolor, Virginica
- **Features**: Sepal length, Sepal width, Petal length, Petal width
- **Samples**: 150 (50 per class)

### Model Architecture
- Algorithm: Decision Tree / Random Forest / SVM (configurable)
- Input: 4 numerical features
- Output: 3-class classification

### Performance
| Metric | Score |
|--------|-------|
| Accuracy | ~97% |
| Precision | ~97% |
| Recall | ~97% |

---

## 2. CNN Flower Classifier (`my_flower_cnn.h5`)

### Dataset
- **Source**: `flower_data/` directory
- **Format**: Image files organized by flower species

### Architecture
```
Input (224x224x3)
  ↓
Conv2D(32, 3x3) + ReLU
  ↓
MaxPooling2D(2x2)
  ↓
Conv2D(64, 3x3) + ReLU
  ↓
MaxPooling2D(2x2)
  ↓
Conv2D(128, 3x3) + ReLU
  ↓
Flatten
  ↓
Dense(512) + ReLU + Dropout(0.5)
  ↓
Dense(num_classes) + Softmax
```

### Training Details
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy
- **Epochs**: 20-50
- **Batch Size**: 32
- **Data Augmentation**: Horizontal flip, rotation, zoom

### Saved Model
- `my_flower_cnn.h5` - Keras HDF5 format
- Load with: `model = tf.keras.models.load_model('my_flower_cnn.h5')`

### Prediction
See `predicted_image.py` for inference example.

---

## Requirements

See `requirements.txt` for full dependency list.

```bash
pip install tensorflow scikit-learn numpy matplotlib pillow
```

---

*Last updated: March 2026*
