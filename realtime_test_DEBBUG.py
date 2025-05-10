import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
from utilities import load_model_lables, extract_keypoints, draw_styled_landmarks, mediapipe_detection, preprocess_landmarks, KEYPOINTS_CONFIG
from utilities import VIDEO_LENGTH

# Define mediapipe model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
DATA_PATH = os.path.join('model/data') 
COLORS = [(245,117,16), (117,245,16), (16,117,245), (245,16,117), (16,245,117), (117,16,245), (245,117,16)]
N_FRAMES_NO_HANDS = 5 # Number of frames to clean if no hands are detected
N_FRAMES_NO_PREDICT = 15 # Number of frames to clean if no prediction is detected


def main():
    config_gpu()

    actions = load_model_lables()
    print(f"Loaded model labels: {actions}")

    model = load_model('model/save_model.keras')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    mp_holistic_obj = mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
    )

    sequence = []
    sentence = []
    predictions = []
    threshold = 0.8
    n_frame_without_hands = 0
    n_frame_without_prediction = 0

    expected_len = len(KEYPOINTS_CONFIG["POSE_LANDMARKS_USED"]) * KEYPOINTS_CONFIG["POSE_DIM"] + \
                   2 * KEYPOINTS_CONFIG["NUM_HAND_LANDMARKS"] * KEYPOINTS_CONFIG["HAND_DIM"]
    print(f"[INFO] Expected keypoint vector length: {expected_len}")

    while True:
        key = cv2.waitKey(16)
        if key == 27:
            break

        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break

        image, results = mediapipe_detection(frame, mp_holistic_obj)
        draw_styled_landmarks(image, results)

        keypoints = preprocess_landmarks(results)

        if len(keypoints) != expected_len:
            print(f"[ERROR] Got keypoint vector of length {len(keypoints)}, expected {expected_len}")
            continue
        else:
            print(f"[INFO] Keypoints OK: {len(keypoints)}")

        if results.left_hand_landmarks is not None or results.right_hand_landmarks is not None:
            sequence.append(keypoints)
            sequence = sequence[-VIDEO_LENGTH:]
            print(f"[INFO] Sequence length: {len(sequence)}")
            n_frame_without_hands = 0
        else:
            print(f"[WARN] No hands detected")
            n_frame_without_hands += 1
            if n_frame_without_hands > N_FRAMES_NO_HANDS:
                sequence = []
                n_frame_without_hands = 0
                print(f"[INFO] Sequence reset due to no hands detected")

        if len(sequence) == VIDEO_LENGTH:
            input_data = np.expand_dims(sequence, axis=0)
            print(f"[INFO] Predicting on input shape: {input_data.shape}")
            res = model.predict(input_data)[0]
            print(f"[RESULT] {actions[np.argmax(res)]} ({res[np.argmax(res)]:.2f})")

            predictions.append(np.argmax(res))

            if res[np.argmax(res)] > threshold:
                n_frame_without_prediction = 0 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
                sequence = []
            else:
                n_frame_without_prediction += 1
                if n_frame_without_prediction > N_FRAMES_NO_PREDICT:
                    sequence = []
                    n_frame_without_prediction = 0
                    print(f"[INFO] Sequence reset due to no prediction")

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            image = prob_viz(res, actions, image)

            if key == ord('s'):
                np.save("debug_sequence.npy", np.array(sequence))
                print("[DEBUG] Saved sequence to debug_sequence.npy")

        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)

    cap.release()
    cv2.destroyAllWindows()


def prob_viz(res, actions, input_frame) -> np.ndarray:
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), COLORS[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame


def config_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Trovate {len(gpus)} GPU fisiche.")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  - Memoria dinamica abilitata per: {gpu.name}")
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Configurate {len(logical_gpus)} GPU logiche.")
        except RuntimeError as e:
            print(f"Errore durante la configurazione della GPU: {e}")
    else:
        print("Nessuna GPU fisica trovata da TensorFlow.")


if __name__ == '__main__':
    main()
