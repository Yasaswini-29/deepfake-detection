import streamlit as st
import numpy as np
import librosa
import joblib
import librosa.display
import matplotlib.pyplot as plt

# Load pre-trained models and other necessary objects
model = joblib.load("svm_audio_model_pca_rbf_optimized.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

def extract_features(audio_path, n_mfcc=100, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        if audio_data.size == 0:
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

        return features
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def plot_spectrogram(audio_path):
    """Plots waveform and spectrogram of the audio file."""
    audio_data, sr = librosa.load(audio_path, sr=None)

    plt.figure(figsize=(10, 5))

    # **Waveform Plot**
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio_data, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # **Spectrogram Plot**
    plt.subplot(2, 1, 2)
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log', cmap='inferno')
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(format="%+2.0f dB")

    plt.tight_layout()
    st.pyplot(plt)

def predict_audio(file_path):
    """Predicts whether the audio is real or fake, visualizes the spectrogram, and displays model accuracy."""
    features = extract_features(file_path)

    if features is not None:
        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)
        prediction = model.predict(features_pca)[0]
        confidence = model.predict_proba(features_pca)[0]

        label = "🟢 Genuine" if prediction == 1 else "🔴 Fake"
        confidence_score = max(confidence) * 100
        
        st.write(f"Prediction: {label} | Confidence: {confidence_score:.2f}%")
        
        # Plot the spectrogram
        plot_spectrogram(file_path)
    else:
        st.write("Error processing the audio file!")

# Streamlit UI
st.title("Real vs Fake Audio Classifier")
st.write("Upload an audio file to predict whether it is real or fake.")

# File uploader
audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "flac"])

if audio_file is not None:
    # Save the uploaded file temporarily
    temp_audio_path = "temp_audio_file.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.getbuffer())

    # Run prediction
    predict_audio(temp_audio_path)
