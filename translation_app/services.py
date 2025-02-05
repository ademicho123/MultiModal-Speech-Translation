from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
    MBartForConditionalGeneration
)
import torch
from typing import List, Dict, Optional
import spacy
import numpy as np

class TranslationService:
    def __init__(self):
        # Speech to text model
        self.speech_model_name = "openai/whisper-large-v3"
        self.speech_processor = AutoProcessor.from_pretrained(self.speech_model_name)
        self.speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(self.speech_model_name)
        
        # Text to text translation model
        self.translation_model_name = "facebook/mbart-large-50-many-to-many-mmt"
        self.translation_model = MBartForConditionalGeneration.from_pretrained(self.translation_model_name)
        self.translation_tokenizer = AutoTokenizer.from_pretrained(self.translation_model_name)
        
        # NER model for named entity preservation
        self.ner_model = spacy.load("en_core_web_trf")
        
        # Context window for document-level translation
        self.context_window_size = 3
        self.translation_history = []
        
        # User-defined glossary
        self.custom_glossary = {}
        
        # Tone mapping for style control
        self.tone_markers = {
            "casual": "<casual>",
            "professional": "<professional>",
            "formal": "<formal>"
        }

    def speech_to_text(self, audio_data: np.ndarray, src_lang: str = "en") -> str:
        """Convert speech to text with improved accuracy"""
        inputs = self.speech_processor(
            audio_data, 
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            output = self.speech_model.generate(
                **inputs,
                language=src_lang,
                task="transcribe"
            )
            
        transcription = self.speech_processor.decode(output[0])
        return transcription

    def translate_text(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        tone: Optional[str] = None,
        context: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Translate text with context awareness and style control"""
        # Apply NER to preserve named entities
        doc = self.ner_model(text)
        entities = {ent.text: ent.label_ for ent in doc.ents}
        
        # Apply custom glossary
        for term, translation in self.custom_glossary.get(f"{src_lang}-{tgt_lang}", {}).items():
            text = text.replace(term, f"<gloss>{translation}</gloss>")
        
        # Add context from previous translations
        if context:
            self.translation_history = context[-self.context_window_size:]
        context_text = " ".join(self.translation_history + [text])
        
        # Add tone marker if specified
        if tone and tone in self.tone_markers:
            context_text = f"{self.tone_markers[tone]} {context_text}"

        # Perform translation
        inputs = self.translation_tokenizer(
            context_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.translation_model.generate(
                **inputs,
                forced_bos_token_id=self.translation_tokenizer.lang_code_to_id[tgt_lang],
                max_length=1024
            )
        
        translation = self.translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Restore named entities
        for entity, label in entities.items():
            translation = translation.replace(f"<{label}>{entity}</{label}>", entity)
        
        # Store translation in history
        self.translation_history.append(text)
        if len(self.translation_history) > self.context_window_size:
            self.translation_history.pop(0)
            
        return {
            "translation": translation,
            "entities": entities,
            "context_used": bool(context),
            "tone_applied": tone
        }

    def update_glossary(self, src_lang: str, tgt_lang: str, terms: Dict[str, str]):
        """Update custom glossary for specific language pair"""
        lang_pair = f"{src_lang}-{tgt_lang}"
        if lang_pair not in self.custom_glossary:
            self.custom_glossary[lang_pair] = {}
        self.custom_glossary[lang_pair].update(terms)

    def clear_translation_history(self):
        """Clear context history"""
        self.translation_history = []