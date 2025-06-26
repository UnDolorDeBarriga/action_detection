# 🧏‍♂️ Hope: Real-Time Italian Sign Language Translator

Welcome to the repository for **Hope**, a Minimum Viable Product (MVP) developed as part of the 2025 SUL course at Politecnico di Milano. This project bridges the communication gap between hearing individuals and the Deaf and Hard-of-Hearing (DHH) community through real-time dynamic Italian Sign Language recognition, phrase generation, and audio output.

---

## 🎯 Project Summary

Hope is an end-to-end pipeline that:
- Captures sign gestures via webcam,
- Extracts 3D landmarks using MediaPipe Holistic,
- Classifies dynamic signs using an LSTM model,
- Converts recognized glosses into Italian phrases via a Large Language Model (Gemini),
- Speaks the output with Google Text-to-Speech.

All of this is presented in a real-time graphical interface built with Tkinter.

---

## 🧠 Key Technologies

| Component            | Technology Used                         |
|----------------------|------------------------------------------|
| Pose & Hand Tracking | MediaPipe Holistic                      |
| Classification Model | TensorFlow Keras (LSTM architecture)    |
| Data Augmentation    | Custom techniques inspired by literature|
| GUI                  | Tkinter                                 |
| Language Modeling    | Gemini LLM                              |
| Audio Output         | Google Text-to-Speech (TTS) API         |

---

## 🚀 MVP Pipeline Overview

```text
Webcam Video ▶️ MediaPipe ▶️ Landmark Normalization ▶️ LSTM Classification ▶️
Gloss Output ▶️ Gemini LLM ▶️ Phrase ▶️ Google TTS ▶️ Audio Output
