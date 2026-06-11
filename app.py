import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import tempfile
import plotly.graph_objects as go
import librosa.display
import matplotlib.pyplot as plt

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Music Emotion Recognition",
    layout="wide"
)

# ==========================
# MODEL
# ==========================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "mer_final_no_lambda.keras"
    )

    return model

model = load_model()

# ==========================
# EMOTION LABELS
# ==========================

EMOTIONS = [
    "HVHA",
    "HVLA",
    "LVHA",
    "LVLA"
]

EMOTION_DESC = {
    "HVHA": "Happy / Excited / Energetic",
    "HVLA": "Relaxed / Peaceful",
    "LVHA": "Angry / Tense",
    "LVLA": "Sad / Melancholic"
}

# ==========================
# FEATURE EXTRACTION
# ==========================

def extract_features(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=22050,
        duration=30
    )

    target_length = sr * 30

    if len(y) < target_length:
        y = np.pad(
            y,
            (0, target_length - len(y))
        )
    else:
        y = y[:target_length]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel = librosa.power_to_db(mel)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    zcr = librosa.feature.zero_crossing_rate(
        y
    )

    rms = librosa.feature.rms(
        y=y
    )

    features = np.vstack([
        mel,
        mfcc,
        chroma,
        centroid,
        bandwidth,
        rolloff,
        zcr,
        rms
    ])

    return features.T.astype(np.float32)

# ==========================
# PREDICTION
# ==========================

def predict_audio(audio_path):

    feat = extract_features(
        audio_path
    )

    feat = np.expand_dims(
        feat,
        axis=0
    )

    pred = model.predict(
        feat,
        verbose=0
    )

    pred_class = np.argmax(pred)

    emotion = EMOTIONS[
        pred_class
    ]

    confidence = float(
        np.max(pred)
    )

    return (
        emotion,
        confidence,
        pred[0]
    )

# ==========================
# WAVEFORM
# ==========================

def plot_waveform(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=None
    )

    max_points = 5000

    if len(y) > max_points:

        indices = np.linspace(
            0,
            len(y)-1,
            max_points,
            dtype=int
        )

        y = y[indices]

    time = np.arange(len(y)) / sr

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time,
            y=y,
            mode='lines'
        )
    )

    fig.update_layout(
        title="Audio Waveform",
        height=400
    )

    return fig

def plot_spectrogram(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=22050
    )

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    fig, ax = plt.subplots(
        figsize=(10,4)
    )

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        ax=ax
    )

    ax.set_title(
        "Mel Spectrogram"
    )

    return fig

# ==========================
# UI
# ==========================

st.title(
    "🎵 Music Emotion Recognition System"
)

st.markdown(
    "Upload a song and compare your emotion with the model prediction."
)

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["mp3","wav"]
)

user_emotion = st.selectbox(
    "Select Your Emotion",
    EMOTIONS
)

if uploaded_file is not None:

    st.audio(
        uploaded_file
    )

    suffix = uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{suffix}"
    ) as tmp:

        tmp.write(
            uploaded_file.getvalue()
        )

        audio_path = tmp.name

    tab1, tab2 = st.tabs(
        [
            "Waveform",
            "Spectrogram"
        ]
    )

    with tab1:

        st.plotly_chart(
            plot_waveform(audio_path),
            use_container_width=True
        )

    with tab2:

        st.pyplot(
            plot_spectrogram(audio_path)
        )

    if st.button(
        "Predict Emotion"
    ):

        emotion, conf, probs = predict_audio(
            audio_path
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Emotion",
                emotion
            )

        with col2:

            st.metric(
                "Confidence",
                f"{conf:.2%}"
            )

        with col3:

            st.metric(
                "User Emotion",
                user_emotion
            )

        st.subheader(
            "Emotion Meaning"
        )

        st.info(
            EMOTION_DESC[emotion]
        )

        st.subheader(
            "Emotion Probability Distribution"
        )

        prob_dict = {
            EMOTIONS[i]:
            float(probs[i])
            for i in range(4)
        }

        st.bar_chart(
            prob_dict
        )

        if emotion == user_emotion:

            st.success(
                "✅ User and Model Agree"
            )

        else:

            st.error(
                "❌ User and Model Disagree"
            )