from django.urls import path
from .views import process_speech, translate_text

urlpatterns = [
    path('speech-to-text/', process_speech, name='speech_to_text'),
    path('translate/', translate_text, name='translate_text'),
]