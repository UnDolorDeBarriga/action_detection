import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow as tf
from utilities import (
    load_model_lables,
    preprocess_landmarks,
    draw_styled_landmarks,
    mediapipe_detection,
    VIDEO_LENGTH
)

class GestureEngine:
    def __init__(self, model_path='model/save_model.keras', draw=True, threshold=0.8):
        self._config_gpu()
        model_path = os.path.abspath(model_path)
        self.model = load_model(model_path)

        self.actions = load_model_lables()
        print(f"[GestureEngine] Loaded model and labels: {self.actions}")

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.threshold = threshold

        self.n_frame_without_hands = 0
        self.n_frame_without_prediction = 0
        self.draw_enabled = draw

        self.N_FRAMES_NO_HANDS = 5
        self.N_FRAMES_NO_PREDICT = 15

    def _config_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[GPU] {len(gpus)} GPU(s) found.")
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"[GPU] Enabled dynamic memory for: {gpu.name}")
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(f"[GPU] Configured {len(logical_gpus)} logical GPU(s).")
            except RuntimeError as e:
                print(f"[GPU] RuntimeError: {e}")
        else:
            print("[GPU] No GPUs found.")

    def toggle_draw(self):
        self.draw_enabled = not self.draw_enabled
        print(f"[GestureEngine] Drawing {'enabled' if self.draw_enabled else 'disabled'}.")

    def reset(self):
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.n_frame_without_hands = 0
        self.n_frame_without_prediction = 0

    def process(self, frame):
        image, results = mediapipe_detection(frame, self.holistic)

        if self.draw_enabled:
            draw_styled_landmarks(image, results)

        keypoints = preprocess_landmarks(results)

        if results.left_hand_landmarks or results.right_hand_landmarks:
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-VIDEO_LENGTH:]
            self.n_frame_without_hands = 0
        else:
            self.n_frame_without_hands += 1
            if self.n_frame_without_hands > self.N_FRAMES_NO_HANDS:
                self.sequence = []
                self.n_frame_without_hands = 0

        if len(self.sequence) == VIDEO_LENGTH:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
            prediction = self.actions[np.argmax(res)]
            confidence = res[np.argmax(res)]
            print(f"[GestureEngine] Prediction: {prediction} ({confidence:.2f})")

            self.predictions.append(np.argmax(res))

            if confidence > self.threshold:
                self.n_frame_without_prediction = 0
                if not self.sentence or prediction != self.sentence[-1]:
                    self.sentence.append(prediction)
                self.sequence = []
            else:
                self.n_frame_without_prediction += 1
                if self.n_frame_without_prediction > self.N_FRAMES_NO_PREDICT:
                    self.sequence = []
                    self.n_frame_without_prediction = 0

            if len(self.sentence) > 5:
                self.sentence = self.sentence[-5:]

            # if self.draw_enabled:
            #     image = self.prob_viz(res, image)

        return image, ' '.join(self.sentence)

    def prob_viz(self, res, image):
        COLORS = [(245,117,16), (117,245,16), (16,117,245), (245,16,117), (16,245,117), (117,16,245), (245,117,16)]
        output = image.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), COLORS[num % len(COLORS)], -1)
            cv2.putText(output, self.actions[num], (0, 85 + num * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return output
