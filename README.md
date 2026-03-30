---
title: Flower Species Classifier
emoji: 🌸
colorFrom: yellow
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Flower Species Classifier 🌸

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-FFD600?style=flat)](https://huggingface.co/spaces/shantoshdurai/flower-species-classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)

A deep learning-based image classification model to identify flower species using **MobileNetV2 transfer learning**. This project demonstrates practical machine learning techniques including data preprocessing, CNN model training,
 and real-time image prediction.

## 📋 Project Overview

This project implements a **Convolutional Neural Network (CNN)** using transfer learning with MobileNetV2 pre-trained on ImageNet. The model classifies 10 different flower species with high accuracy and demonstrates end-to-end machine learning workflow from data loading to inference.

### Model Performance
- **Model Architecture**: MobileNetV2 (frozen base) + Global Average Pooling + Dense output layer
- **Training Data**: 587 images (80%)
- **Validation Data**: 146 images (20%)
- **Total Classes**: 10 flower species

## 🌺 Supported Flower Species

1. Bougainvillea
2. Daisies
3. Garden Roses
4. Gardenias
5. Hibiscus
6. Hydrangeas
7. Lilies
8. Orchids
9. Peonies
10. Tulips

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/shantoshdurai/flower-species-classifier.git
cd flower-species-classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📚 Usage

### Training the Model

```python
from train import train_model

train_model(epochs=10, batch_size=32)
```

### Making Predictions

```python
from predict import predict_flower_species

species, confidence = predict_flower_species('path_to_flower_image.jpg')
print(f"Predicted Species: {species}")
print(f"Confidence: {confidence:.2f}%")
```

## 🏗️ Project Structure

```
flower-species-classifier/
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── requirements.txt              # Project dependencies
├── my_flower_cnn.h5             # Trained model      
└── README.md                     # This file
```

## 🔧 Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **MobileNetV2**: Pre-trained CNN architecture for transfer learning
- **NumPy**: Numerical computations
- **Pillow**: Image processing
- **Matplotlib**: Visualization

## 📖 How It Works

### Transfer Learning Approach

1. **Load Pre-trained MobileNetV2**: Uses weights trained on ImageNet dataset
2. **Freeze Base Model**: Keeps pre-trained weights fixed to leverage learned features
3. **Add Custom Layers**: Global Average Pooling + Dense layer for flower classification
4. **Train on Flower Data**: Fine-tune only the new layers on your specific dataset
5. **Inference**: Use trained model to predict flower species from new images

### Model Architecture
```
MobileNetV2 (frozen)
        ↓
Global Average Pooling
        ↓
Dense(512, activation='relu')
        ↓
Dense(10, activation='softmax') → Output (10 flower classes)
```

## 📊 Results

- **Training Accuracy**: 95%+
- **Validation Accuracy**: 90%+
- **Inference Time**: <200ms per image (CPU)
- **Model Size**: ~11 MB

## 🎓 Learning Outcomes

This project covers:

- ✅ Image dataset organization and preprocessing
- ✅ Transfer learning with pre-trained models
- ✅ Data splitting (train/validation)
- ✅ CNN model training and evaluation
- ✅ Image prediction and inference
- ✅ Model serialization (saving/loading)
- ✅ Professional GitHub repository structure

## 🔮 Future Improvements

- [ ] Add web interface using Flask/Streamlit
- [ ] Deploy model as REST API
- [ ] Implement data augmentation for better generalization
- [ ] Add more flower species
- [ ] Mobile app deployment (TensorFlow Lite)
- [ ] Real-time camera prediction
- [ ] Advanced performance metrics and confusion matrix

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- **Author**: Shantosh Durai
- **GitHub**: [@shantoshdurai](https://github.com/shantoshdurai)
- **Email**: santoshp123steam@gmail.com
  

## 🙏 Acknowledgments

- TensorFlow/Keras for the amazing deep learning framework
- MobileNetV2 authors for the efficient architecture
- Kaggle for the flower dataset

---

**Made with ❤️ for flower lovers and ML enthusiasts!**
