from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image  # Add this line
import os
import tempfile
import io
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # limit uploads to 16MB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()  # Use temp directory

# Load CNN model - using the same approach as in the notebook
def load_model():
    try:
        model_path = 'final_inceptionresnetv2_deepfake_model_99_new/final_inceptionresnetv2_deepfake_model_99_new.keras'
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            print(f"Current working directory: {os.getcwd()}")
            return None
            
        # Load the model exactly as in notebook
        model = tf.keras.models.load_model(model_path)
        
        # Verify model loaded correctly
        print("Model loaded successfully")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Test prediction with random data
        test_input = np.random.random((1, 224, 224, 3))
        test_output = model.predict(test_input)
        print(f"Test prediction shape: {test_output.shape}")
        print(f"Test prediction values: {test_output}")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Global variable for model
model = load_model()

def extract_frames(video_path, num_frames=10):
    """Extract evenly spaced frames from video"""
    frames = []
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        return frames
    
    # Calculate frame indices to extract
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for i in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = video.read()
        if success:
            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    video.release()
    return frames

# Replace the existing preprocess_frame function with this:
def preprocess_frame(frame):
    """Preprocess a frame exactly as in the notebook"""
    # Convert to PIL Image format first
    img_pil = Image.fromarray(frame)
    
    # Save as temporary file to use image.load_img
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_frame.jpg')
    img_pil.save(temp_path)
    
    # Use the exact same preprocessing as in the notebook
    img = image.load_img(temp_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    
    # Remove temp file
    os.remove(temp_path)
    
    return img_array

def predict_deepfake(frames):
    """Process frames through model and return prediction"""
    if not model:
        return {"error": "Model not loaded"}, 500
    
    # Preprocess frames for the model
    processed_frames = []
    for frame in frames:
        processed = preprocess_frame(frame)
        processed_frames.append(processed)
    
    # Convert to numpy array
    batch = np.array(processed_frames)
    
    # Add explicit model verification
    print(f"Model input shape: {batch.shape}")
    print(f"Model input range: min={np.min(batch)}, max={np.max(batch)}")
    
    # Get predictions for all frames
    predictions = model.predict(batch)
    print(f"Raw predictions shape: {predictions.shape}")
    print(f"Raw predictions values: {predictions}")
    
    # Flatten predictions if needed
    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions.flatten()
    
    print(f"Flattened predictions: {predictions}")    
    
    # REVERSED CLASSIFICATION THRESHOLD - FIX THE LOGIC
    frame_results = ['Fake' if p > 0.5 else 'Real' for p in predictions]  # Changed < to >
    print(f"Frame-by-frame results: {frame_results}")
    
    # Average the predictions across frames
    avg_prediction = float(np.mean(predictions))
    print(f"Average prediction: {avg_prediction}")
    
    # REVERSED CLASSIFICATION THRESHOLD - FIX THE LOGIC
    is_deepfake = bool(avg_prediction > 0.5)  # Changed < to >
    confidence = avg_prediction if is_deepfake else (1.0 - avg_prediction)  # Adjusted confidence calculation
    
    return {
        "is_deepfake": is_deepfake,
        "result": "Fake" if is_deepfake else "Real",
        "confidence": round(confidence * 100, 2),
        "frame_results": frame_results
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "Empty video file name"}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)
        
        # Extract frames
        frames = extract_frames(filepath, num_frames=10)
        
        if not frames:
            return jsonify({"error": "Could not extract frames from video"}), 400
        
        # Process frames through the model
        result = predict_deepfake(frames)
        
        # Clean up temp file
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)