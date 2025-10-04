from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure the upload folder exists and has proper permissions
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
# Configure maximum content length for file uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the model
model = load_model('plant_disease_classifier.h5')

# Classes dictionary
classes = {
    0: 'Pepper bell Bacterial spot',
    1: 'Pepper bell healthy',
    2: 'Potato Early blight',
    3: 'Potato Late blight',
    4: 'Potato healthy',
    5: 'Tomato Bacterial spot',
    6: 'Tomato Early blight',
    7: 'Tomato Late blight',
    8: 'Tomato Leaf Mold',
    9: 'Tomato Septoria leaf_spot',
    10: 'Tomato Spider mites Two spotted spider mite',
    11: 'Tomato Target Spot',
    12: 'Tomato Tomato YellowLeaf Curl Virus',
    13: 'Tomato Tomato mosaic virus',
    14: 'Tomato healthy'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Get prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    return classes[predicted_class]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    image_path = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            prediction = predict_image(filepath)
            # Make the image path relative to the static folder
            image_path = 'uploads/' + filename
    
    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)