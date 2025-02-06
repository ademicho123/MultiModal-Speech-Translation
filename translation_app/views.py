from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .services import TranslationService

translator = TranslationService()

@csrf_exempt
def process_speech(request):
    """ Endpoint for Speech-to-Text (STT) """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    try:
        data = json.loads(request.body)
        audio_path = data.get("audio_path")
        text = translator.speech_to_text(audio_path)
        return JsonResponse({"text": text, "status": "success"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def translate_text(request):
    """ Endpoint for Text-to-Text (TTT) """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    try:
        data = json.loads(request.body)
        text = data.get("text")
        src_lang = data.get("source_language", "en")  # Use short codes in API
        tgt_lang = data.get("target_language", "fr")  # Use short codes in API
        
        if not text:
            return JsonResponse({"error": "Text is required"}, status=400)
            
        translation = translator.translate_text(text, src_lang, tgt_lang)
        return JsonResponse({"translation": translation, "status": "success"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
