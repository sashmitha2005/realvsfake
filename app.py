from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import gc  # Import garbage collection

app = Flask(__name__)

# Load the trained model
model = load_model(os.path.join(os.getcwd(), 'final_model.h5'))

# Set the folder for saving uploaded files
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

THRESHOLD = 0.44

def preprocess_image(filepath):
    image = load_img(filepath, target_size=(300, 300))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize to [0,1]
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image = preprocess_image(filepath)
            prediction = model.predict(image)

            distance = calculate_euclidean_distance(prediction[0], np.array([0.5]))
            label = 'Real' if distance <= THRESHOLD else 'Fake'

            # Clean up to free memory
            del image, prediction
            gc.collect()

            return render_template('result.html', label=label, filepath=filepath)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return "An error occurred during prediction. Please try again."

if __name__ == '__main__':
    app.run(debug=False)  # Use Gunicorn for production
