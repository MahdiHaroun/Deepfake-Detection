# Deepfake Photo Detector

A web application that detects deepfake content in videos by extracting frames and analyzing them with a CNN model.

## Features

- Upload short video files through a user-friendly interface
- Process videos by extracting 10 representative frames
- Compress frames efficiently without quality loss
- Analyze frames using a CNN model to detect deepfakes
- Display results with confidence level
- Optimized for performance with no database or permanent storage requirements

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/DeepFake-Photos-Genrator.git
   cd DeepFake-Photos-Genrator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

Start the application with:
```
python app.py
```

The application will be available at http://localhost:5000

## How It Works

1. **Video Upload**: Users upload a short video file through the web interface
2. **Frame Extraction**: The system extracts 10 evenly spaced frames from the video
3. **Image Compression**: Frames are compressed to reduce size while maintaining quality
4. **Model Analysis**: 
   - Frames are normalized and fed into a CNN model
   - Each frame receives a prediction score
   - Scores are averaged to determine final classification
5. **Result Display**: The system shows whether the video is authentic or manipulated with a confidence score

## Technical Implementation

- **Framework**: Flask web server
- **Video Processing**: OpenCV for frame extraction
- **Image Processing**: PIL/Pillow for efficient compression
- **Model**: TensorFlow CNN model for deepfake detection
- **In-memory processing**: No database or storage requirements

## Performance Optimizations

- Only 10 frames are extracted to minimize processing time
- Images are resized to 224x224 pixels for the model
- Efficient JPEG compression reduces memory usage without losing quality
- Temporary files are cleaned up after processing
- All processing happens in memory for better performance
- Model is loaded once at startup to save time during analysis

## Customization

Replace the dummy model in `app.py` with your trained CNN model:

```python
# Replace with your actual model loading code
model = tf.keras.models.load_model('path_to_your_model')
```

## License

[MIT License](LICENSE)
