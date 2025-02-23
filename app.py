import streamlit as st
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf

# Load trained model and scaler
model = joblib.load("svm_audio_model.pkl")
scaler = joblib.load("scaler.pkl")

# Expected feature length (check with your training data)
EXPECTED_FEATURE_LENGTH = scaler.n_features_in_

def extract_features(audio_path, n_mfcc=20, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)

        # Extract feature vector
        features = np.hstack((
            np.mean(mfccs, axis=1), 
            np.mean(delta_mfccs, axis=1), 
            np.mean(chroma, axis=1),
            np.mean(mel_spec, axis=1)[:20],  # Ensure only first 20 values
            np.mean(spectral_contrast, axis=1)
        ))

        # Fix feature length
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
    # Save uploaded file temporarily
    audio_path = "temp.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format="audio/wav")

    # Extract features
    features = extract_features(audio_path)

    if features is not None:
        st.write(f"Extracted feature shape: {features.shape[0]}")
        st.write(f"Scaler expects: {EXPECTED_FEATURE_LENGTH} features")

        # Scale features
        features_scaled = scaler.transform([features])

        # Predict
        confidence = model.predict_proba(features_scaled)[0]
        prediction = "Genuine" if confidence[1] > confidence[0] else "Fake"
        confidence_score = max(confidence) * 100

        st.write(f"### 🎯 Prediction: **{prediction}**")
        st.write(f"📊 Confidence: **{confidence_score:.2f}%**")

        # Plot waveform and spectrogram
        audio_data, sr = librosa.load(audio_path, sr=None)
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))

        # Waveform
        librosa.display.waveshow(audio_data, sr=sr, ax=ax[0])
        ax[0].set(title="Waveform")

        # Spectrogram
        spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
        ax[1].set(title="Spectrogram")
        fig.colorbar(ax[1].collections[0], ax=ax[1])

        st.pyplot(fig)
