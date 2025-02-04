from django.urls import path
from . import views

urlpatterns = [
    path('sign-to-speech/', views.process_sign_language, name='sign_to_speech'),
    path('speech-to-sign/', views.process_speech, name='speech_to_sign'),
]