from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import logging
import numpy as np
import base64
from PIL import Image
import io
from .services import SignLanguageTranslator

logger = logging.getLogger(__name__)

# Initialize the translator
try:
    translator = SignLanguageTranslator()
    logger.info("Successfully initialized SignLanguageTranslator")
except Exception as e:
    logger.error(f"Failed to initialize translator: {str(e)}")
    translator = None

@csrf_exempt
def process_sign_language(request):
    """Endpoint for processing sign language images"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    if translator is None:
        return JsonResponse({"error": "Translator service is not available"}, status=503)

    logger.info("Received request to process_sign_language")
    try:
        data = json.loads(request.body)
        logger.debug(f"Received data keys: {data.keys()}")
        
        # Handle base64 encoded image
        if 'image_base64' in data:
            image_data = base64.b64decode(data['image_base64'])
            image = Image.open(io.BytesIO(image_data))
        # Handle raw frame data
        elif 'frame' in data:
            frame = np.array(data['frame'])
            image = Image.fromarray(frame)
        else:
            return JsonResponse({"error": "No image data provided"}, status=400)

        # Process the image
        results = translator.process_sign_frame(image)
        
        # Convert to speech if requested
        if data.get('convert_to_speech', False):
            # Get the text from the top prediction
            text = results[0]['label']
            audio = translator.text_to_speech(text, data.get('target_language', 'eng'))
            results = {
                'sign_detection': results,
                'audio_output': audio.tolist()
            }
        
        return JsonResponse({
            "results": results,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error processing sign language: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def process_speech(request):
    """Endpoint for processing speech"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    if translator is None:
        return JsonResponse({"error": "Translator service is not available"}, status=503)
        
    try:
        data = json.loads(request.body)
        audio_data = np.array(data['audio'])
        source_language = data.get('source_language', 'eng')
        
        # Convert speech to text
        text = translator.speech_to_text(audio_data, source_language)
        
        return JsonResponse({
            "text": text,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error processing speech: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)