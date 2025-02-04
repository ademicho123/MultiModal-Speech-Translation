from transformers import (
    pipeline,
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoProcessor, 
    AutoModelForSpeechSeq2Seq
)
import torch
import numpy as np
import cv2
from PIL import Image
import io

class SignLanguageTranslator:
    def __init__(self):
        # Initialize sign language classification pipeline
        self.sign_pipeline = pipeline(
            "image-classification", 
            model="Heem2/sign-language-classification"
        )
        
        # Initialize speech model
        self.speech_model_name = "facebook/seamless-m4t-v2-large"
        self.speech_processor = AutoProcessor.from_pretrained(self.speech_model_name)
        self.speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(self.speech_model_name)

    def process_sign_frame(self, frame):
        """
        Process a video frame containing sign language
        frame: numpy array of the image
        """
        # Convert numpy array to PIL Image
        if isinstance(frame, np.ndarray):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
        else:
            pil_image = frame

        # Get classification results
        results = self.sign_pipeline(pil_image)
        return results

    def text_to_speech(self, text, tgt_lang="eng"):
        """Convert text to speech"""
        inputs = self.speech_processor(text, src_lang=tgt_lang, return_tensors="pt")
        with torch.no_grad():
            audio_output = self.speech_model.generate(**inputs, tgt_lang=tgt_lang)
        return audio_output.cpu().numpy()

    def speech_to_text(self, audio_data, src_lang="eng"):
        """Convert speech to text"""
        inputs = self.speech_processor(audio_data, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            output = self.speech_model.generate(**inputs, tgt_lang=src_lang)
        return self.speech_processor.decode(output[0])