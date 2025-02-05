from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import logging
import numpy as np
from .services import TranslationService

logger = logging.getLogger(__name__)

# Initialize the translation service
try:
    translator = TranslationService()
    logger.info("Successfully initialized TranslationService")
except Exception as e:
    logger.error(f"Failed to initialize translator: {str(e)}")
    translator = None

@csrf_exempt
def process_speech(request):
    """Endpoint for processing speech to text"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    if translator is None:
        return JsonResponse({"error": "Translation service is not available"}, status=503)
        
    try:
        data = json.loads(request.body)
        audio_data = np.array(data['audio'])
        source_language = data.get('source_language', 'en')
        
        text = translator.speech_to_text(audio_data, source_language)
        
        return JsonResponse({
            "text": text,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error processing speech: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def translate_text(request):
    """Endpoint for text-to-text translation"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    if translator is None:
        return JsonResponse({"error": "Translation service is not available"}, status=503)
        
    try:
        data = json.loads(request.body)
        text = data['text']
        source_language = data.get('source_language', 'en')
        target_language = data.get('target_language', 'fr')
        tone = data.get('tone')
        context = data.get('context', [])
        
        result = translator.translate_text(
            text=text,
            src_lang=source_language,
            tgt_lang=target_language,
            tone=tone,
            context=context
        )
        
        return JsonResponse({
            **result,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def update_glossary(request):
    """Endpoint for updating custom glossary"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
        
    try:
        data = json.loads(request.body)
        source_language = data['source_language']
        target_language = data['target_language']
        terms = data['terms']
        
        translator.update_glossary(source_language, target_language, terms)
        
        return JsonResponse({
            "status": "success",
            "message": "Glossary updated successfully"
        })
        
    except Exception as e:
        logger.error(f"Error updating glossary: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)