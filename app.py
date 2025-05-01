import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import io
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # limit uploads to 16MB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()  # Use temp directory

# Load CNN model - dummy model for now, replace with your actual model
def load_model():
    try:
        # Replace this with your actual model loading code
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])  
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # In production, you would load weights here
        # model.load_weights('path_to_your_weights.h5')
        print("Model loaded successfully")
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

def compress_image(frame, quality=85):
    """Compress image using PIL while maintaining quality"""
    # Convert numpy array to PIL Image
    img = Image.fromarray(frame)
    
    # Resize to standard input size for the model (224x224 is common)
    img = img.resize((224, 224), Image.LANCZOS)
    
    # Compress using a BytesIO buffer
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    
    # Convert back to numpy array for model input
    buffer.seek(0)
    compressed_img = Image.open(buffer)
    return np.array(compressed_img)

def predict_deepfake(frames):
    """Process frames through model and return prediction"""
    if not model:
        return {"error": "Model not loaded"}, 500
    
    # Preprocess frames for the model
    processed_frames = []
    for frame in frames:
        # Compress and normalize
        compressed = compress_image(frame)
        # Normalize pixel values to [0,1]
        normalized = compressed / 255.0
        processed_frames.append(normalized)
    
    # Convert to numpy array
    batch = np.array(processed_frames)
    
    # Get predictions for all frames
    predictions = model.predict(batch)
    
    # Average the predictions across frames
    avg_prediction = np.mean(predictions)
    
    # Threshold for classification
    is_deepfake = bool(avg_prediction > 0.5)
    confidence = float(avg_prediction) if is_deepfake else float(1.0 - avg_prediction)
    
    return {
        "is_deepfake": is_deepfake,
        "confidence": round(confidence * 100, 2)
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
        return jsonify({"error": "No video selected"}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)
        
        # Extract frames
        frames = extract_frames(filepath, num_frames=10)
        
        if not frames:
            os.remove(filepath)
            return jsonify({"error": "Could not extract frames from video"}), 400
        
        # Process frames through the model
        result = predict_deepfake(frames)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)