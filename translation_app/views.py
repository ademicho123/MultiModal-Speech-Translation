from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .services import MultimodalTranslator

translator = MultimodalTranslator()  # Initialize the translator once

@csrf_exempt
def translate_view(request):
    """ API endpoint for multimodal translation """
    if request.method == "POST":
        data = json.loads(request.body)
        audio_path = data.get("audio_path")
        text = data.get("text")
        video_path = data.get("video_path")
        tgt_lang = data.get("tgt_lang", "fr")

        result = {}

        if audio_path:
            result["speech_to_text"] = translator.speech_to_text(audio_path)
            result["translated_text"] = translator.text_translation(result["speech_to_text"], tgt_lang=tgt_lang)

        if text:
            result["translated_text"] = translator.text_translation(text, tgt_lang=tgt_lang)
            translator.text_to_speech(result["translated_text"])  # Speak output

        if video_path:
            result["sign_language_text"] = translator.recognize_sign_language(video_path)
            translator.text_to_speech(result["sign_language_text"])  # Speak recognized sign

        return JsonResponse(result)

    return JsonResponse({"error": "Invalid request"}, status=400)
