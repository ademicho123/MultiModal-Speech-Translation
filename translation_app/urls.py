from django.urls import path
from . import views

urlpatterns = [
    path('speech-to-text/', views.process_speech, name='speech_to_text'),
    path('translate/', views.translate_text, name='translate_text'),
    path('glossary/update/', views.update_glossary, name='update_glossary'),
]