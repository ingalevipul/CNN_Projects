# 🌿 Plant Disease Prediction Web Application

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)

This web application leverages deep learning to predict diseases in plant leaves in real-time. Using a trained CNN model, it can identify various diseases in Tomato, Potato, and Pepper Bell plants, as well as distinguish healthy plants.

## ✨ Features


- 🌱 Support for multiple plant types:
  - 🍅 Tomato
  - 🥔 Potato
  - 🫑 Pepper Bell
- 💻 Easy-to-use web interface
- ⚡ Instant visual feedback
- 📸 Support for common image formats (JPG, JPEG, PNG)

## 📊 Dataset

The model is trained on the Plant village  dataset available on Kaggle:
[Plant village Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

## 🔍 Disease Categories

The model can predict the following conditions:

### 🫑 Pepper Bell
- Bacterial spot
- Healthy

### 🥔 Potato
- Early blight
- Late blight
- Healthy

### 🍅 Tomato
- Bacterial spot
- Early blight
- Late blight
- Leaf Mold
- Septoria leaf spot
- Spider mites
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic virus
- Healthy

## 🛠️ Technical Stack

- **Backend**: `Python Flask`
- **Deep Learning**: `TensorFlow/Keras`
- **Frontend**: `HTML`, `Bootstrap`
- **Model**: `CNN (Convolutional Neural Network)`

## 📥 Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
```

2. Install the required dependencies:
```bash
pip install flask tensorflow pillow numpy werkzeug
```

3. Run the application:
```bash
python app.py
```

4. Open your web browser and navigate to:
```
http://localhost:5000
```

## 🚀 Usage

1. 🌐 Open the web application in your browser
2. 📂 Click on "Choose File" to select an image of a plant leaf
3. 🔄 Click "Predict Disease" to get the prediction
4. 📊 View the results:
   - The uploaded image
   - The predicted disease or health status

## 🧠 Model Information

The application uses a pre-trained CNN model (`plant_disease_classifier.h5`) that has been trained on a large dataset of plant leaf images. The model can be trained using our [Colab Notebook](https://colab.research.google.com/drive/1YRVHoyzTM4VudGWhl98JRP-UG9dRWdNq?usp=sharing).

## 📁 Directory Structure

```
📦 CNN_Projects\Leaf disease prediction
 ┣ 📜 app.py                     # Main Flask application
 ┣ 📜 plant_disease_classifier.h5 # Trained model
 ┣ 📂 static/
 ┃ ┗ 📂 uploads/                # Folder for uploaded images
 ┣ 📂 templates/
 ┃ ┗ 📜 index.html             # Web interface template
 ┗ 📜 README.md                # Documentation
```

## 📋 Requirements

- `Python 3.x`
- `Flask`
- `TensorFlow`
- `Pillow (PIL)`
- `NumPy`
- `Werkzeug`

