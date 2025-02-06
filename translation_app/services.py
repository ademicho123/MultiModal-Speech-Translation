import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import librosa
import os

class TranslationService:
    """ Uses Whisper for STT and NLLB for TTT """

    def __init__(self):
        #Load Whisper Model for Speech-to-Text
        self.whisper_model_name = "openai/whisper-large-v3"
        self.whisper_processor = WhisperProcessor.from_pretrained(self.whisper_model_name)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(self.whisper_model_name)

        # Load NLLB Model for Text Translation
        self.nllb_model_name = "facebook/nllb-200-distilled-600M"
        self.nllb_tokenizer = AutoTokenizer.from_pretrained(self.nllb_model_name)
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(self.nllb_model_name)

    def speech_to_text(self, audio_path):
        """ Converts speech to text using Whisper """
        
        if not os.path.exists(audio_path):
            return "Error: Audio file not found."

        audio, sr = librosa.load(audio_path, sr=16000)
        
        if len(audio) == 0:
            return "Error: Audio file is empty."

        if sr != 16000:
            audio = librosa.resample(audio, sr, 16000)

        inputs = self.whisper_processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(inputs.input_features)
            
        return self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    def translate_text(self, text, src_lang="en", tgt_lang="fr"):
        """ Translates text using NLLB with correct language codes """
        # Language code mapping
        LANG_CODES = {
            "en": "eng_Latn",  # English
            "fr": "fra_Latn",  # French
            "es": "spa_Latn",  # Spanish
            "ar": "ara_Arab",  # Arabic
            "zh": "zho_Hans",  # Chinese (Simplified)
        }

        src_lang_code = LANG_CODES.get(src_lang, "eng_Latn")
        tgt_lang_code = LANG_CODES.get(tgt_lang, "fra_Latn")

        # Prepare inputs with truncation
        inputs = self.nllb_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate translation with enhanced parameters (similar to your translator_V1.py)
        with torch.no_grad():
            outputs = self.nllb_model.generate(
                **inputs,
                forced_bos_token_id=self.nllb_tokenizer.convert_tokens_to_ids(tgt_lang_code),
                max_length=512,
                num_beams=8,
                length_penalty=1.2,
                no_repeat_ngram_size=2,
                early_stopping=True,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )

        # Decode the translation
        translation = self.nllb_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return translation