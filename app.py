from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import os
from tensorflow.lite.python.interpreter import Interpreter
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import euclidean
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

# Load the TensorFlow Lite model
TFLITE_MODEL_PATH = os.path.join(os.getcwd(), 'final_model.tflite')

# Initialize the TFLite interpreter
interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the folder for saving uploaded files
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set the prediction threshold (adjust this based on your model's evaluation)
THRESHOLD = 0.44  # Pre-calculated threshold from your previous code

# Preprocess the image for prediction
def preprocess_image(filepath):
    image = load_img(filepath, target_size=(300, 300))  # Adjust size if necessary
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize to [0,1]
    return image

# Calculate Euclidean distance between two vectors
def calculate_euclidean_distance(pred, centroid):
    return euclidean(pred, centroid)

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

            # Preprocess the uploaded image
            image = preprocess_image(filepath)

            # Set the image as input to the TensorFlow Lite model
            interpreter.set_tensor(input_details[0]['index'], image)

            # Run inference
            interpreter.invoke()

            # Get the model's output
            prediction = interpreter.get_tensor(output_details[0]['index'])

            # For now, we compare the first value of prediction against threshold
            # You may modify this if your model outputs multiple values
            distance = calculate_euclidean_distance(prediction[0], np.array([0.5]))  # Assuming centroid is around [0.5]
            
            # Apply threshold to determine if 'Real' or 'Fake'
            if distance <= THRESHOLD:
                label = 'Real'
            else:
                label = 'Fake'

            return render_template('result.html', label=label, filepath=filepath)

        except Exception as e:
            # Log the error and return an error message
            print(f"Error during prediction: {e}")
            return "An error occurred during prediction. Please try again."

# Evaluation function to help improve threshold tuning
def evaluate_model(interpreter, validation_data):
    y_true = []  # Actual labels
    y_pred = []  # Predicted labels

    for image_path, label in validation_data:
        # Preprocess the image
        image = preprocess_image(image_path)

        # Set the image as input to the TensorFlow Lite model
        interpreter.set_tensor(input_details[0]['index'], image)

        # Run inference
        interpreter.invoke()

        # Get the model's output
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Calculate Euclidean distance from the prediction
        distance = calculate_euclidean_distance(prediction[0], np.array([0.5]))  # Assuming centroid [0.5]

        # Compare distance to threshold
        if distance <= THRESHOLD:
            y_pred.append(0)  # 'Real'
        else:
            y_pred.append(1)  # 'Fake'
        
        y_true.append(label)

    # Confusion matrix and classification report
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

if __name__ == '__main__':
    
    app.run(debug=True)
