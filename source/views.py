from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from .forms import UserRegistrationForm,UserUpdateForm, ProfileUpdateForm,customerChurn,churning
from .models import annModel,churn_model
from django.contrib.auth.decorators import login_required
from . import signals
from django.template.loader import render_to_string
from django.contrib.auth.models import User
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin

from .models import Profile
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth import authenticate, login
from django.http import HttpResponse
from sklearn.preprocessing import StandardScaler
import joblib 
import os
from keras.models import load_model
from django.shortcuts import render
from django.http import JsonResponse
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import librosa
import numpy as np
from pydub import AudioSegment
import io




#============================AI final project=========================

def predictEmotion(request):

    template_name = "user/emotions.html"

    return render(request, template_name)

@csrf_exempt
def analyze(request):
    print("I am yet to post")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "modeltodeploy/")
    SCALER_PATH = os.path.join(os.path.dirname(__file__), "emotionscaler.joblib")
    ENCODER_PATH = os.path.join(os.path.dirname(__file__), "cemotionencoder.joblib")
    if request.method == 'POST':
       

        audio_file = request.FILES.get('audio_file')

       if audio_file:
             Convert to WAV format using pydub
            audio = AudioSegment.from_file(audio_file)
            audio_path = 'user/audio.wav'
            audio.export(audio_path, format="wav")

           
        

        
        import base64
        import io

        audio = io.BytesIO(audio_binary)

        audio = AudioSegment.from_file(audio)

      
        output_audio_path = 'output_audio.wav'
        finalaudio = audio.export(audio, format='wav')

        feature = get_features(audio_path)
        scaler = joblib.load(SCALER_PATH)
        scaledfeature = scaler.transform(feature)
        encoder = joblib.load(ENCODER_PATH)
        model = load_model(MODEL_PATH)
        prediction = model.predict(scaledfeature)

        output = encoder.inverse_transform(prediction)

       
        result = f"You know, your customer is  :==>  {output]}"

        # Return the result as JSON
        return JsonResponse({'emotion': result})

    # If the request is not a POST request, render an empty page
    return render(request, 'user/emotions.html')





def extract_features(data,sr):
        # ZCR
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result=np.hstack((result, zcr)) # stacking horizontally

        # Chroma_stft
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma_stft)) # stacking horizontally

        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr).T, axis=0)
        result = np.hstack((result, mfcc)) # stacking horizontally

        # Root Mean Square Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms)) # stacking horizontally
        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
        result = np.hstack((result, mel)) # stacking horizontally

        return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, sr = None)
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)
    return result



    
  









        
        


    


        
