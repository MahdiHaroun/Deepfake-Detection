from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import os
import tempfile
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # limit uploads to 16MB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load model exactly like in the notebook
def load_model_from_notebook():
    try:
        model_path = 'final_inceptionresnetv2_deepfake_model_99_new/final_inceptionresnetv2_deepfake_model_99_new.keras'
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            print(f"Current working directory: {os.getcwd()}")
            return None
            
        model = tf.keras.models.load_model(model_path)
        
        print("Model loaded successfully")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Test prediction
        test_input = np.random.random((1, 224, 224, 3))
        test_output = model.predict(test_input)
        print(f"Test prediction shape: {test_output.shape}")
        print(f"Test prediction values: {test_output}")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model once at startup
model = load_model_from_notebook()

def preprocess_image(image_path):
    """Process a single image exactly as in the notebook"""
    try:
        # Load and resize image
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        
        # Convert to array and normalize
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
        img_array = img_array / 255.0  # Normalize
        
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict_image(image_path):
    """Predict if an image is real or fake"""
    if not model:
        return {"error": "Model not loaded"}, 500
    
    # Preprocess the image
    img_array = preprocess_image(image_path)
    if img_array is None:
        return {"error": "Failed to process image"}, 400
    
    # Get prediction
    predictions = model.predict(img_array)
    prediction_value = float(predictions[0][0])
    
    print(f"Raw prediction: {prediction_value}")
    
    # Classify using same threshold as notebook
    label = "Real" if prediction_value >= 0.5 else "Fake"
    is_deepfake = label == "Fake"
    
    # Calculate confidence
    confidence = prediction_value if label == "Real" else (1.0 - prediction_value)
    
    return {
        "is_deepfake": is_deepfake,
        "result": label,
        "confidence": round(confidence * 100, 2),
        "raw_prediction": round(prediction_value, 4)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'photo' not in request.files:
        return jsonify({"error": "No photo file provided"}), 400
    
    photo_file = request.files['photo']
    if photo_file.filename == '':
        return jsonify({"error": "Empty photo file name"}), 400
    
    if not allowed_file(photo_file.filename):
        return jsonify({"error": "File type not allowed. Please upload a JPG or PNG image"}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(photo_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo_file.save(filepath)
        
        # Process the image
        result = predict_image(filepath)
        
        # Clean up temp file
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing photo: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)