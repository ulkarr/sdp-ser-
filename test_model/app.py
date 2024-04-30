import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

import librosa
import librosa.display
# import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split



# to play the audio files
# from IPython.display import Audio
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization,Activation
# from tensorflow.keras.utils import to_categorical

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<header class="css-18ni7ap e8zbici2">
  <a href="#" class="about-btn">About Us</a>
  <a href="#" class="contact-btn">Contact</a>
  <h1></h1>
</header>
""",
 
unsafe_allow_html=True) 


# SAMPLE_RATE = 22500
DURATION = 5
#add features.csv file here
#Features = pd.read_csv(r"C:\Users\User\Videos\ai sdp\test_model\features_final_version_sdp.csv") #23sdp
#Features = pd.read_csv(r"C:\Users\User\Videos\ai sdp\test_model\features_sdp.csv") #24sdp
Features = pd.read_csv(r"C:\Users\User\Videos\ai sdp\test_model\features_sdp(acersnew).csv") #24sdpnewversion
Features.head()
X = Features.iloc[: ,:-1].values
Y = Features['labels'].values
# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
# splitting data, 95% train and 5 % test
x_train, x_test, y_train, y_test = train_test_split(X, Y,test_size=0.05, random_state=0, shuffle=True)
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
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result



def record_audio():
    with st.spinner("Recording..."):
        audio = sd.rec(int(sample_rate * DURATION), samplerate=sample_rate, channels=1)
        sd.wait()
        z = audio.reshape(-1)
        return z

# Define the Streamlit app




def app():

    # Add CSS to change the background color


# Define your app logic below

    # Add a title
    st.markdown("<h1 style='text-align: center; color: #000000;'>Speech Emotion Recognition</h1>", unsafe_allow_html=True)


    button_style = """
    <style>
    div.stButton > button:first-child {
        background-color: purple;

        color: white;
        border-radius: 50px;
        font-size: 20px;
        padding: 10px 20px;
        margin-right: 20px;
    }

    div.stButton > button:last-child {
        background-color: #008080;
        color: #fff;
        border-radius: 25px;
        font-size: 18px;
        padding: 8px 25px;
        box-shadow: 0px 0px 10px 1px #008080;
        transition: all 0.2s ease-in-out;
    }

    div.stButton > button:last-child:hover {
        background-color: #fff;
        color: #008080;
        box-shadow: 0px 0px 15px 1px #008080;
        cursor: pointer;
    }
    </style>
    """

    st.header("Record Audio")
    # Add the CSS style to the page
    st.markdown(button_style, unsafe_allow_html=True)

    fs = 22500
    # Use beta_columns to display the buttons horizontally
    col1, col2 = st.columns(2)

    start_recording = col1.button("Start Recording")
    stop_recording = col2.button("Stop Recording")

    recording = None

    

    if start_recording:
        is_recording = True
        audio = record_audio()
        audio_file = 'recording.wav'
        sf.write(audio_file, audio, sample_rate)
        st.audio(audio_file, format='audio/wav')
        z = record_audio()

    if stop_recording:
        st.write("Recording stopped.")
        if recording is not None:
            sf.write("recording.wav", recording, fs)

    # Define the UI for predicting the emotion
    st.header("Predict Emotion")
        # Add the CSS style for the "Predict" button
    predict_style = """
    <style>
    .stButton button:last-child {
        background-color: blue;
    }
    </style>
    """
    st.markdown(predict_style, unsafe_allow_html=True)
    if st.button("Predict"):
        if True:
            # Load the model and the encoder
            #add new model link here
            # loaded_model = tf.keras.models.load_model(r'C:\Users\hp\OneDrive\İş masası\SDP_FINAL_ALL\cnn_model.h5')
            #loaded_model = tf.keras.models.load_model(r'C:\Users\User\Videos\ai sdp\test_model\cnn_model.h5') #23sdp
            #loaded_model = tf.keras.models.load_model(r'C:\Users\User\Videos\ai sdp\test_model\cnn_model(acers).h5') #24sdp
            loaded_model = tf.keras.models.load_model(r'C:\Users\User\Videos\ai sdp\test_model\cnn_model(acersnew).h5') #24sdpnewversion
            # Load the recorded audio file
            audio, sr = librosa.load("recording.wav", sr=fs)

            # path_ = r"C:\Users\hp\Downloads\test_model\recording.wav"
            path_ = r"C:\Users\User\Videos\ai sdp\test_model\recording.wav"
            data_, sample_rate_ = librosa.load(path_)
            X_ = np.array(extract_features(data_))
            X_ = scaler.transform(X_.reshape(1,-1))
            pred_test_ = loaded_model.predict(np.expand_dims(X_, axis=2))
            prediction = encoder.inverse_transform(pred_test_)
            prediction = prediction[0][0]


        import matplotlib.pyplot as plt

        # Display the prediction with custom styling and text
        if prediction == "happy":
            st.markdown("<div style='background-color: #DAF7A6; padding: 10px; border-radius: 5px;'>"
                        "<h2 style='text-align: center; color: #5CB85C; font-family: Arial, sans-serif;'>The predicted emotion is: {} 😃</h2>"
                        "<p style='text-align: center; color: #5CB85C; font-family: Arial, sans-serif;'>Wow, that's great to hear! Keep spreading positivity 😊</p>"
                        "</div>".format(prediction), unsafe_allow_html=True)
            audio, sr = librosa.load("recording.wav", sr=fs)
            fig, ax = plt.subplots()
            plt.title('Happy')
            ax.set(xlabel='Time (seconds)', ylabel='Amplitude')
            ax.plot(np.linspace(0, len(audio) / sr, len(audio)), audio)
            st.pyplot(fig)
        elif prediction == "sad":
            st.markdown("<div style='background-color: #AED6F1; padding: 10px; border-radius: 5px;'>"
                        "<h2 style='text-align: center; color: #3498DB; font-family: Arial, sans-serif;'>The predicted emotion is: {} 😢</h2>"
                        "<p style='text-align: center; color: #3498DB; font-family: Arial, sans-serif;'>I'm sorry to hear that. Remember, things will get better soon ❤️</p>"
                        "</div>".format(prediction), unsafe_allow_html=True)
            audio, sr = librosa.load("recording.wav", sr=fs)
            fig, ax = plt.subplots()
            plt.title('Sad')
            ax.set(xlabel='Time (seconds)', ylabel='Amplitude')
            ax.plot(np.linspace(0, len(audio) / sr, len(audio)), audio)
            st.pyplot(fig)
        elif prediction == "angry":
            st.markdown("<div style='background-color: #F5B7B1; padding: 10px; border-radius: 5px;'>"
                        "<h2 style='text-align: center; color: #D9534F; font-family: Arial, sans-serif;'>The predicted emotion is: {} 😠</h2>"
                        "<p style='text-align: center; color: #D9534F; font-family: Arial, sans-serif;'>Take a deep breath and try to calm down. Everything will be okay 😌</p>"
                        "</div>".format(prediction), unsafe_allow_html=True)
            audio, sr = librosa.load("recording.wav", sr=fs)
            fig, ax = plt.subplots()
            plt.title('Angry')
            ax.set(xlabel='Time (seconds)', ylabel='Amplitude')
            ax.plot(np.linspace(0, len(audio) / sr, len(audio)), audio)
            st.pyplot(fig)
        elif prediction == "fear":
            st.markdown("<div style='background-color: #D2B4DE; padding: 10px; border-radius: 5px;'>"
                        "<h2 style='text-align: center; color: #8E44AD; font-family: Arial, sans-serif;'>The predicted emotion is: {} 😨</h2>"
                        "<p style='text-align: center; color: #8E44AD; font-family: Arial, sans-serif;'>It's okay to be scared sometimes. Just remember to take things one step at a time 🙏</p>"
                        "</div>".format(prediction), unsafe_allow_html=True)
            audio, sr = librosa.load("recording.wav", sr=fs)
            fig, ax = plt.subplots()
            plt.title('Fear')
            ax.set(xlabel='Time (seconds)', ylabel='Amplitude')
            ax.plot(np.linspace(0, len(audio) / sr, len(audio)), audio)
            st.pyplot(fig)
        elif prediction == "surprised":
            st.markdown("<div style='background-color: #2E7D32; padding: 10px; border-radius: 5px;'>"
                        "<h2 style='text-align: center; color: #C5E1A5; font-family: Arial, sans-serif;'>The predicted emotion is: {} 😲</h2>"
                        "<p style='text-align: center; color: #C5E1A5; font-family: Arial, sans-serif;'>Wow, you look genuinely surprised!</p>"
                        "</div>".format(prediction), unsafe_allow_html=True)
            audio, sr = librosa.load("recording.wav", sr=fs)
            fig, ax = plt.subplots()
            plt.title('surprised')
            ax.set(xlabel='Time (seconds)', ylabel='Amplitude')
            ax.plot(np.linspace(0, len(audio) / sr, len(audio)), audio)
            st.pyplot(fig)

        else:
            st.markdown("<div style='background-color: #F4F6F6; padding: 10px; border-radius: 5px;'>"
                        "<h2 style='text-align: center; color: #7F8C8D; font-family: Arial, sans-serif;'>The predicted emotion is: {} 😐</h2>"
                        "<p style='text-align: center; color: #7F8C8D; font-family: Arial, sans-serif;'>Looks like you're feeling neutral. That's okay, it's good to take a break and relax 😌</p>"
                        "</div>".format(prediction), unsafe_allow_html=True)
            audio, sr = librosa.load("recording.wav", sr=fs)
            fig, ax = plt.subplots()
            plt.title('Neutral')
            ax.set(xlabel='Time (seconds)', ylabel='Amplitude')
            ax.plot(np.linspace(0, len(audio) / sr, len(audio)), audio)
            st.pyplot(fig)

if __name__ == "__main__":
    app()
