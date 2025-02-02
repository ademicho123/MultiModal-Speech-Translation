# translation_app/services.py
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, MarianMTModel, MarianTokenizer
import mediapipe as mp
import cv2
from gtts import gTTS
import os

class MultimodalTranslator:
    """ Unified processing for speech, text, and sign language """

    def __init__(self):
        # Load models once to avoid reloading per request
        self.stt_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        self.stt_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
        self.mp_hands = mp.solutions.hands.Hands()

    def speech_to_text(self, audio_path):
        """ Converts speech to text """
        waveform, rate = torchaudio.load(audio_path)
        inputs = self.stt_processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=rate)
        with torch.no_grad():
            logits = self.stt_model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.stt_processor.decode(predicted_ids[0])

    def text_translation(self, text, src_lang="en", tgt_lang="fr"):
        """ Translates text between languages """
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**tokens)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def text_to_speech(self, text, lang="en", output_file="output.mp3"):
        """ Converts text to speech """
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_file)
        os.system(f"mpg321 {output_file}")  # Play the speech

    def recognize_sign_language(self, video_path):
        """ Recognizes sign language from video input """
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                return "Recognized Sign Gesture"  # Placeholder for real recognition logic
        cap.release()
        return "No Gesture Detected"
