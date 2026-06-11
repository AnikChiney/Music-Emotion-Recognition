## 👨‍💻 About the Author

I am Anik Chiney, a Computer Science and Engineering student at the Institute of Engineering & Management (IEM), Kolkata, with a strong interest in Artificial Intelligence, Machine Learning, Deep Learning, Quantum Machine Learning, and Full-Stack Development.

My work focuses on building intelligent systems that combine research and practical applications. I have experience developing projects in Music Emotion Recognition, AI-powered applications, MERN stack development, authentication systems, WebRTC-based communication platforms, and Retrieval-Augmented Generation (RAG) systems.

I am passionate about exploring emerging technologies and applying machine learning techniques to solve real-world problems. Through academic projects, research, and internships, I continuously strive to expand my knowledge in AI, Deep Learning, and Quantum Computing while building scalable and impactful software solutions.

### Areas of Interest

* Artificial Intelligence (AI)
* Machine Learning (ML)
* Deep Learning (DL)
* Quantum Machine Learning (QML)
* Music Emotion Recognition
* Natural Language Processing (NLP)
* Generative AI
* Retrieval-Augmented Generation (RAG)
* Full-Stack Web Development
* Computer Vision

### Contact

* Name: Anik Chiney
* Institute: Institute of Engineering & Management (IEM), Kolkata
* Email: anikchiney130@gmail.com
* GitHub: https://github.com/AnikChiney

"Turning research ideas into practical AI solutions."


<img width="1812" height="531" alt="image" src="https://github.com/user-attachments/assets/f2688be5-60b9-4dcd-8e3c-d5af565115d1" />
<img width="1785" height="690" alt="image" src="https://github.com/user-attachments/assets/1fa22057-b721-42da-827e-056e03d0d366" />
<img width="1742" height="795" alt="image" src="https://github.com/user-attachments/assets/223ee694-5f80-41f0-8b49-14f3ea07e235" />
<img width="1775" height="796" alt="image" src="https://github.com/user-attachments/assets/055fce60-9971-476c-9b17-0207e72f8b2b" />
<img width="1828" height="757" alt="image" src="https://github.com/user-attachments/assets/8e5c413a-91cf-44ca-b353-c053c760203f" />


# 🎵 Music Emotion Recognition using CNN-BiLSTM-Attention

## 📌 Overview

This project presents a **Music Emotion Recognition (MER)** system that predicts the emotional category of a music track using Deep Learning. The model is trained on the **DEAM (Database for Emotional Analysis of Music)** dataset and classifies music into four emotional quadrants based on the Valence-Arousal model.

The application is deployed using **Streamlit** and allows users to:

* Upload an audio file (.mp3/.wav)
* Listen to the uploaded song
* Visualize the audio waveform
* Visualize the Mel Spectrogram
* Predict the emotion of the music
* Compare the model prediction with user-selected emotion
* View class probability distribution

---

## 🎯 Emotion Classes

| Emotion Code | Description                 |
| ------------ | --------------------------- |
| HVHA         | Happy / Excited / Energetic |
| HVLA         | Relaxed / Peaceful          |
| LVHA         | Angry / Tense               |
| LVLA         | Sad / Melancholic           |

---

## 🗂 Dataset

### DEAM Dataset

Database for Emotional Analysis of Music

The DEAM dataset contains:

* Audio recordings
* Static valence annotations
* Static arousal annotations
* Dynamic emotional annotations

Official Website:
http://cvml.unige.ch/databases/DEAM/

---

## ⚙️ Feature Extraction

The following handcrafted audio features are extracted using Librosa:

### Spectral Features

* Spectral Centroid
* Spectral Bandwidth
* Spectral Rolloff

### Temporal Features

* Zero Crossing Rate (ZCR)
* RMS Energy

### Frequency Features

* Mel Spectrogram (128 bands)
* MFCC (40 coefficients)
* Chroma Features

Combined feature shape:

```text
(1292, 185)
```

where:

* 1292 = Time frames
* 185 = Feature dimensions

---

## 🧠 Deep Learning Architecture

### CNN + BiLSTM + Attention

```text
Input Features
      │
      ▼
Conv1D (128)
      │
Batch Normalization
      │
Max Pooling
      │
Conv1D (256)
      │
Batch Normalization
      │
Max Pooling
      │
BiLSTM (128)
      │
Attention Mechanism
      │
Global Average Pooling
      │
Dense (256)
      │
Dropout (0.3)
      │
Softmax (4 Classes)
```

### Components

* CNN layers capture local temporal patterns.
* BiLSTM captures long-range temporal dependencies.
* Attention layer focuses on emotionally important segments.
* Softmax layer performs multi-class emotion classification.

---

## 📊 Data Augmentation

To improve generalization, the following augmentations were applied:

1. Original Audio
2. Additive Gaussian Noise
3. Pitch Shift
4. Time Stretch

This increased the dataset size by approximately 4×.

---

## 🚀 Streamlit Application Features

### Audio Upload

Users can upload:

* MP3
* WAV

### Visualizations

* Audio Waveform
* Mel Spectrogram

### Emotion Prediction

Displays:

* Predicted Emotion
* Confidence Score
* Emotion Description
* Probability Distribution

### User Comparison

Allows users to select their perceived emotion and compare it with the model prediction.

---

## 🛠 Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Audio Processing

* Librosa
* SoundFile

### Machine Learning

* Scikit-Learn

### Visualization

* Plotly
* Matplotlib

### Deployment

* Streamlit
* Render

---

## 📁 Project Structure

```text
Music-Emotion-Recognition/
│
├── app.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── README.md
│
├── models/
│   └── mer_final_no_lambda.keras
│
├── extracted_features/
│
├── results/
│
└── dataset/
```

---

## ▶️ Running Locally

### Clone Repository

```bash
git clone https://github.com/AnikChiney/Music-Emotion-Recognition.git

cd Music-Emotion-Recognition
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit

```bash
streamlit run app.py
```

---

## 📈 Future Enhancements

* Real-time emotion recognition
* Quantum Machine Learning models (QSVC, VQC)
* Multi-label emotion prediction
* Dynamic emotion tracking
* Music recommendation system
* Emotion timeline visualization

---

## 👨‍💻 Author

Anik Chiney

Institute of Engineering & Management (IEM)

Email: anikchiney130@gmail.com

GitHub:
https://github.com/AnikChiney

---

## 📜 License & Usage Policy

© 2026 Anik Chiney. All Rights Reserved.

This project has been developed as part of academic research and software development work by **Anik Chiney**.

You are welcome to view, learn from, and reference this project for educational and research purposes. However, modification, redistribution, commercial use, reproduction, or incorporation of any part of this project into another project is **not permitted without prior written permission from the author**.

### Restrictions

* ❌ Do not modify the source code without permission.
* ❌ Do not redistribute this project or its components without permission.
* ❌ Do not use this project for commercial purposes without permission.
* ❌ Do not publish derivative works based on this project without permission.
* ❌ Do not claim this work as your own.

### Permissions

For collaboration, reuse, modification, research partnerships, or commercial licensing, please contact the author for approval.

Author: Anik Chiney
Email: anikchiney130@gmail.com
GitHub: https://github.com/AnikChiney

By using this repository, you agree to respect the above terms and acknowledge the original author.




