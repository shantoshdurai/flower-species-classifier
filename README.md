
# Flower Species Classifier 🌸

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-FFD600?style=flat)](https://santoshp123-flower-species-classifiers.hf.space)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning model that identifies flower species from photos using **MobileNetV2 transfer learning**. Upload an image and get instant predictions with confidence scores.

**[Try the live demo →](https://santoshp123-flower-species-classifiers.hf.space)**

## 🎬 Demo

| Idle State | With Prediction |
|---|---|
| ![App Interface](images/image.png) | ![Prediction Result](images/image%20result.png) |

## ✨ Features

- 🌸 Classifies 10 flower species
- ⚡ Fast predictions (<200ms)
- 🎨 Beautiful Streamlit interface
- 🐳 Docker containerized
- ☁️ Deployed on Hugging Face Spaces

## 🌺 Supported Flowers

Bougainvillea • Daisies • Garden Roses • Gardenias • Hibiscus • Hydrangeas • Lilies • Orchids • Peonies • Tulips

## 🚀 Quick Start

### Run Locally

```bash
git clone https://github.com/shantoshdurai/flower-species-classifier.git
cd flower-species-classifier
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Run with Docker

```bash
docker build -t flower-classifier .
docker run -p 7860:7860 flower-classifier
```

## 🏗️ Project Structure

```
├── streamlit_app.py       # Web app
├── Dockerfile             # Containerization
├── requirements.txt       # Dependencies
├── my_flower_cnn.h5      # Model (24MB)
└── images/               # Demo screenshots
```

## 🔧 Tech Stack

- **TensorFlow/Keras** — Deep learning
- **MobileNetV2** — Efficient CNN architecture
- **Streamlit** — Web interface
- **Docker** — Deployment

## 📊 Model Info

- **Architecture**: MobileNetV2 + Global Average Pooling + Dense layers
- **Training Accuracy**: 95%+
- **Validation Accuracy**: 90%+
- **Model Size**: 24 MB

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

Made with ❤️ by [Shantosh Durai](https://github.com/shantoshdurai)
