import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import csv
from tensorflow.keras.models import load_model
import tensorflow as tf

#TODO: 1: Take out face
#TODO: 2: Only take train data if hands are visisble
#TODO: 3: Is possible to train with more weight on the hands?
#TODO: 4: Training data
#TODO: 5: Improve recognise logic
#TODO: 6: Send sentence to LLM to build a proper phrase

# Define mediapipe model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
DATA_PATH = os.path.join('model/data') 
VIDEO_LENGTH = 25                               # Number of frames to save for each action
COLORS = [(245,117,16), (117,245,16), (16,117,245)]

def main():
    config_gpu()
    # Load actions to detect
    actions = load_model_lables()
    print(f"Loaded model labels: {actions}")

    # Load model
    model = load_model('model/save_model.keras')

    # Video capture
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
    threshold = 0.9
    while True:
        # If ESC (27) is pressed, exit the loop
        key = cv2.waitKey(16)
        if key == 27:
            break
        
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break
        
        # Make detections
        image, results = mediapipe_detection(frame, mp_holistic_obj)
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        print(f"RH: {np.all(keypoints[-21*3:])}, LH: {np.all(keypoints[-2*21*3:-21*3])}, Size: {len(sequence)}")
        if np.all(keypoints[-21*3:]) != 0 or np.all(keypoints[-2*21*3:-21*3]) != 0:
            sequence.append(keypoints)
            sequence = sequence[-25:]
        
        if len(sequence) == 25:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            print(res[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
            #3. Viz logic
            # if np.unique(predictions[-10:])[0]==np.argmax(res): 
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
                sequence = []
            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

    cap.release()
    cv2.destroyAllWindows()


def mediapipe_detection(image, model) -> tuple:
    """
    Detects landmarks in the image using the specified model.
    Args:
        image: The input image.
        model: The mediapipe model to use for detection.
    Returns:
        image: The processed image with landmarks drawn.
        results: The detection results containing landmarks.
    """
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results
    
def load_model_lables() -> list:
    """
    Loads the labels for the classifier model from a CSV file.
    Returns:
        model_classifier_labels: A list of labels for the classifier model.
    """
    with open("model/custom_model_label.csv", encoding='utf-8-sig') as f:
            model_classifier_labels = csv.reader(f)
            model_classifier_labels = [
                row[0] for row in model_classifier_labels
            ]
    return model_classifier_labels

def draw_landmarks(image, results) -> None:
    """
    Draws landmarks on the image using the specified results.
    Args:
        image: The input image.
        results: The detection results containing landmarks.
    """
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results) -> None:
    """
    Draws styled landmarks on the image using the specified results.
    Args:
        image: The input image.
        results: The detection results containing landmarks.
    """
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=3), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=1)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=3), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1)
                             )

# TODO: Check if is necesary to do a preprocessing on landmakrs to get the relative position, not the absolute. 
def extract_keypoints(results) -> np.ndarray:
    """
    Extracts keypoints from the results and flattens them into a single array.
    Args:
        results: The detection results containing landmarks.
    Returns:
        A flattened array of keypoints.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame) -> np.ndarray:
    """
    Visualizes the probabilities of the actions.
    Args:
        res: The prediction results.
        actions: The list of actions.
        input_frame: The input frame to draw on.
    Returns:
        output_frame: The processed frame with probabilities drawn.
    """
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), COLORS[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame
    
def config_gpu():
    # --- INIZIO CODICE CONFIGURAZIONE GPU ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Trovate {len(gpus)} GPU fisiche.")
        try:
            # Itera su ogni GPU fisica trovata
            for gpu in gpus:  # <--- Qui viene definita la variabile 'gpu'
                # Imposta la crescita della memoria per QUESTA specifica gpu
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  - Memoria dinamica abilitata per: {gpu.name}") # Stampa info utile
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Configurate {len(logical_gpus)} GPU logiche.")
        except RuntimeError as e:
            # La crescita della memoria deve essere impostata prima dell'inizializzazione!
            print(f"Errore durante la configurazione della GPU: {e}")
    else:
        print("Nessuna GPU fisica trovata da TensorFlow.")
    # --- FINE CODICE CONFIGURAZIONE GPU --


if __name__ == '__main__':
    main()