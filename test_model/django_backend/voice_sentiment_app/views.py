import wave
import io
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split



DURATION = 5
Features = pd.read_csv(r"C:\Users\User\Videos\ai_sdp\test_model\features_sdp(acersnew).csv")  # 24sdpnewversion
Features.head()
X = Features.iloc[:, :-1].values
Y = Features['labels'].values
# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
# splitting data, 95% train and 5 % test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
# x_train.shape, y_train.shape, x_test.shape, y_test.shape
# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# x_train.shape, y_train.shape, x_test.shape, y_test.shape

# making our data compatible to model.
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
# x_train.shape, y_train.shape, x_test.shape, y_test.shape


sample_rate = 22500


def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


fs = 22500
def predict_emotion(audio_file_name):
    # Load the pre-trained CNN model
    loaded_model = tf.keras.models.load_model(
        r'C:\Users\User\Videos\ai_sdp\test_model\cnn_model(20%test).h5')  # 24sdpnewversion
    # Load the recorded audio file
    # audio, sr = librosa.load(r"C:\Users\User\Videos\ai sdp\test_model\newSong0_23.wav", sr=fs)

    # path_ = r"C:\Users\hp\Downloads\test_model\recording.wav"
    path_ = fr"C:\Users\User\Videos\ai_sdp\voices\{audio_file_name}"
    data_, sample_rate_ = librosa.load(path_)
    X_ = np.array(extract_features(data_))
    X_ = scaler.transform(X_.reshape(1, -1))
    pred_test_ = loaded_model.predict(np.expand_dims(X_, axis=2))
    prediction = encoder.inverse_transform(pred_test_)
    prediction = prediction[0][0]
    return prediction




@csrf_exempt
def audio_upload(request):
    if request.method == 'POST':
        audio_file = request.FILES['file']
        print(audio_file)
        # Assuming the function `predict_emotion` takes an audio file and returns a prediction
        result = predict_emotion(audio_file.name)
        return JsonResponse({'emotion': result})
    return JsonResponse({'error': 'Invalid request'}, status=400)











# def give_record_seconds(audio_file):
#     with wave.open(audio_file, 'rb') as wf:
#         # Get the number of frames and the frame rate
#         num_frames = wf.getnframes()
#         frame_rate = wf.getframerate()
#         # Calculate the duration in seconds
#         duration_seconds = num_frames / float(frame_rate)
#     return duration_seconds
















def home(request):
    return render(request, 'index.html')


def about_us(request):
    return render(request, 'about_us.html')


def contact(request):
    return render(request, 'contact.html')


def tryser(request):
    return render(request, 'tryser.html')
















