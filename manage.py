# Runs models for real-time transcription and translation  

from models.stt import transcribe_speech
from models.ttt import translate_text
from models.sign_language import recognize_sign_language

def multimodal_translation(audio_path=None, text=None, video_path=None, tgt_lang="fr"):
    result = {}
    if audio_path:
        result["speech_to_text"] = transcribe_speech(audio_path)
        result["translated_text"] = translate_text(result["speech_to_text"], tgt_lang=tgt_lang)
    if text:
        result["translated_text"] = translate_text(text, tgt_lang=tgt_lang)
    if video_path:
        result["sign_language"] = recognize_sign_language(video_path)
    return result
