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

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-FFD600?style=flat)](https://santoshp123-flower-species-classifiers.hf.space)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://tensorflow.org)

A deep learning-based image classification model to identify flower species using **MobileNetV2 transfer learning**. Upload a photo and instantly get the flower species with confidence scores. Deployed as an interactive web app using Streamlit and Docker.

## 📋 Project Overview

This project implements a **Convolutional Neural Network (CNN)** using transfer learning with MobileNetV2 pre-trained on ImageNet. The model classifies 10 different flower species with high accuracy and demonstrates end-to-end machine learning workflow from data loading to inference.

### Model Performance
- **Model Architecture**: MobileNetV2 (frozen base) + Global Average Pooling + Dense output layer
- **Training Data**: 587 images (80%)
- **Validation Data**: 146 images (20%)
- **Total Classes**: 10 flower species

## 🎬 Live Demo

**Try it now**: [🌸 Flower Species Classifier](https://santoshp123-flower-species-classifiers.hf.space)

Upload any flower photo and get instant classification with confidence scores!

### Demo Screenshots

**Idle State:**
![App Interface](images/app-idle.png)

**With Prediction:**
![App with Prediction](images/app-prediction.png)

---

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

### Run the Web App Locally

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### Run with Docker

```bash
docker build -t flower-classifier .
docker run -p 7860:7860 flower-classifier
```

Visit `http://localhost:7860` in your browser.

## 🏗️ Project Structure

```
flower-species-classifier/
├── streamlit_app.py              # Interactive web app (Streamlit)
├── Dockerfile                    # Docker containerization
├── requirements.txt              # Python dependencies
├── my_flower_cnn.h5             # Trained MobileNetV2 model (24MB)
├── images/                       # Demo screenshots
│   ├── app-idle.png
│   └── app-prediction.png
└── README.md                     # This file
```

## 🔧 Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **MobileNetV2**: Pre-trained CNN architecture for transfer learning
- **Streamlit**: Interactive web application framework
- **Docker**: Containerization for reliable deployment
- **Hugging Face Spaces**: Cloud hosting platform
- **NumPy**: Numerical computations
- **Pillow**: Image processing

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
- **Model Size**: ~24 MB

## 🎓 Learning Outcomes

This project covers:

- ✅ Image dataset organization and preprocessing
- ✅ Transfer learning with pre-trained models
- ✅ Data splitting (train/validation)
- ✅ CNN model training and evaluation
- ✅ Image prediction and inference
- ✅ Model serialization (saving/loading)
- ✅ Streamlit web application development
- ✅ Docker containerization
- ✅ Cloud deployment (Hugging Face Spaces)

## 🔮 Future Improvements

- [ ] Add more flower species to training data
- [ ] Implement data augmentation for better generalization
- [ ] Mobile app deployment (TensorFlow Lite)
- [ ] Real-time camera prediction
- [ ] Advanced performance metrics and confusion matrix
- [ ] Batch processing for multiple images
- [ ] API endpoint for programmatic access

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
- Hugging Face for Spaces hosting

---

**Made with ❤️ for flower lovers and ML enthusiasts!**
