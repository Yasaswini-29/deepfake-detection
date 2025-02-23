import streamlit as st
import numpy as np
import librosa
import joblib
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

# Load trained model and scaler
model = joblib.load("svm_audio_model.pkl")
scaler = joblib.load("scaler.pkl")
EXPECTED_FEATURE_LENGTH = scaler.n_features_in_

def extract_features(audio_path, n_mfcc=20, n_fft=2048, hop_length=512):
    """
    Extracts audio features: MFCCs, chroma, mel spectrogram, spectral contrast, rolloff, zero-crossing rate, and spectrogram.
    """
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        if audio_data.size == 0:
            st.error("Error: Empty audio file.")
            return None
        
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(audio_data)
        
        spec, _ = librosa.magphase(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length))
        spec_mean = np.mean(spec, axis=1)
        
        features = np.hstack((
            np.mean(mfccs, axis=1), np.mean(delta_mfccs, axis=1), np.mean(chroma, axis=1),
            np.mean(mel_spec_db, axis=1), np.mean(spectral_contrast, axis=1),
            np.mean(spectral_rolloff, axis=1), np.mean(zero_crossing, axis=1), spec_mean
        ))
        
        if features.shape[0] < EXPECTED_FEATURE_LENGTH:
            features = np.pad(features, (0, EXPECTED_FEATURE_LENGTH - features.shape[0]), mode='constant')
        elif features.shape[0] > EXPECTED_FEATURE_LENGTH:
            features = features[:EXPECTED_FEATURE_LENGTH]
        
        return features
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

st.title("🔍 Audio Impersonation Detection")
st.write("Upload an audio file to analyze whether it is **Genuine** or **Fake**.")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    features = extract_features(file_path)
    if features is not None:
        features_scaled = scaler.transform([features])
        confidence = model.predict_proba(features_scaled)[0]
        prediction = "Genuine" if confidence[1] > confidence[0] else "Fake"
        
        st.write(f"Prediction: **{prediction}**")
        st.write(f"Confidence: {max(confidence) * 100:.2f}%")
        
        # Plot waveform and spectrogram
        audio_data, sr = librosa.load(file_path, sr=None)
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(audio_data, sr=sr)
        plt.title("Waveform")
        
        plt.subplot(2, 1, 2)
        spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
        plt.title("Spectrogram")
        plt.colorbar(format="%+2.0f dB")
        
        st.pyplot(plt)
