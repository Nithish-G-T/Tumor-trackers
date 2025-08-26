from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import io
from PIL import Image
import numpy as np
import traceback
import logging
from ml_utils import TumorPredictor
from tts_utils import TextToSpeech

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and TTS
tumor_predictor = None
tts_engine = None

def initialize_services():
    """
    Initialize the ML model and TTS engine.
    """
    global tumor_predictor, tts_engine
    
    try:
        # Initialize tumor predictor
        tumor_predictor = TumorPredictor(model_path="tumor_model.pth")
        logger.info("Tumor predictor initialized successfully")
        
        # Initialize TTS engine
        tts_engine = TextToSpeech()
        logger.info("TTS engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Brain Tumor Prediction API is running',
        'model_loaded': tumor_predictor is not None,
        'tts_available': tts_engine is not None
    })

@app.route('/predict', methods=['POST'])
def predict_tumor():
    """
    Main prediction endpoint that accepts an image and returns tumor analysis.
    
    Expected input:
    - Multipart form data with 'image' field containing the image file
    
    Returns:
    - JSON response with prediction results, medical summary, and visualizations
    """
    try:
        # Check if image file is present in request
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'message': 'Please upload an image file with the key "image"'
            }), 400
        
        image_file = request.files['image']
        
        # Validate file
        if image_file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a valid image file'
            }), 400
        
        # Check if model is loaded
        if tumor_predictor is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Tumor prediction model is not available'
            }), 500
        
        # Load and validate image
        try:
            image = Image.open(image_file.stream)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validate image dimensions
            if image.size[0] < 50 or image.size[1] < 50:
                return jsonify({
                    'error': 'Image too small',
                    'message': 'Image dimensions must be at least 50x50 pixels'
                }), 400
            
        except Exception as e:
            return jsonify({
                'error': 'Invalid image file',
                'message': f'Could not process image: {str(e)}'
            }), 400
        
        # Perform tumor prediction
        logger.info("Starting tumor prediction...")
        prediction = tumor_predictor.predict(image)
        
        # Map tumor type to benign/malignant classification
        tumor_type = prediction['class']
        if tumor_type == 'glioma':
            malignancy = 'malignant'
        elif tumor_type in ['meningioma', 'pituitary', 'no_tumor']:
            malignancy = 'benign'
        else:
            malignancy = 'unknown'
        
        # Generate medical summary
        medical_summary = tumor_predictor.generate_medical_summary(prediction)
        
        # Generate Grad-CAM visualization
        logger.info("Generating Grad-CAM visualization...")
        gradcam_image = tumor_predictor.generate_gradcam(image)
        
        # Generate audio summary
        audio_base64 = None
        if tts_engine is not None:
            try:
                logger.info("Generating audio summary...")
                audio_base64 = tts_engine.text_to_speech_base64(medical_summary)
            except Exception as e:
                logger.warning(f"Failed to generate audio: {e}")
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'tumor_type': prediction['class'],
                'malignancy': malignancy,
                'confidence': prediction['confidence'],
                'confidence_percentage': f"{prediction['confidence']:.1%}",
                'probabilities': dict(zip(prediction['class_names'], prediction['probabilities']))
            },
            'medical_summary': medical_summary,
            'visualization': {
                'gradcam_image': gradcam_image,
                'image_format': 'base64_png'
            },
            'audio': {
                'available': audio_base64 is not None,
                'audio_data': audio_base64,
                'format': 'base64_mp3' if audio_base64 else None
            },
            'metadata': {
                'model_info': 'EfficientNet-B3 Brain Tumor Classifier',
                'image_processed': True,
                'timestamp': str(np.datetime64('now'))
            }
        }
        
        logger.info(f"Prediction completed successfully: {prediction['class']} ({prediction['confidence']:.1%})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': 'Prediction failed',
            'message': f'An error occurred during prediction: {str(e)}',
            'success': False
        }), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """
    Get information about the loaded model.
    """
    if tumor_predictor is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Tumor prediction model is not available'
        }), 500
    
    return jsonify({
        'model_type': 'EfficientNet-B3',
        'classes': tumor_predictor.class_names,
        'input_size': '300x300',
        'device': str(tumor_predictor.device),
        'model_loaded': True
    })

@app.route('/audio/<tumor_type>', methods=['POST'])
def generate_audio_for_tumor(tumor_type):
    """
    Generate audio summary for a specific tumor type (for testing purposes).
    """
    try:
        if tts_engine is None:
            return jsonify({
                'error': 'TTS not available',
                'message': 'Text-to-speech engine is not available'
            }), 500
        
        # Create a sample medical summary for the tumor type
        sample_summary = f"""
        MEDICAL ANALYSIS REPORT
        
        DIAGNOSIS: {tumor_type.upper()}
        CONFIDENCE: 95.0% (very high confidence)
        
        CLINICAL FINDINGS:
        This is a sample medical summary for {tumor_type} tumor type.
        
        RECOMMENDATIONS:
        - Further clinical correlation is recommended
        - Consult with a neurosurgeon for treatment planning
        """
        
        # Generate audio
        audio_base64 = tts_engine.text_to_speech_base64(sample_summary)
        
        return jsonify({
            'success': True,
            'tumor_type': tumor_type,
            'audio_data': audio_base64,
            'format': 'base64_mp3'
        })
        
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return jsonify({
            'error': 'Audio generation failed',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors.
    """
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 errors.
    """
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    # Initialize services before starting the server
    print("Initializing brain tumor prediction services...")
    initialize_services()
    
    # Start Flask server
    print("Starting Flask server...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
