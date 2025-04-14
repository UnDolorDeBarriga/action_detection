import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
from utilities import load_model_lables, extract_keypoints, draw_styled_landmarks, mediapipe_detection, preprocess_landmarks
from utilities import VIDEO_LENGTH

#TODO: 1: Take out face
    # Normalization
    # Rotations
    # Squeeze
    # Prespective
    # Gausian Noise
#TODO: 2: Only take train data if hands are visisble
#TODO: 3: Is possible to train with more weight on the hands?
#TODO: 4: Training data
#TODO: 5: Improve recognise logic
#TODO: 6: Send sentence to LLM to build a proper phrase

# Define mediapipe model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
DATA_PATH = os.path.join('model/data') 
                          # Number of frames to save for each action
COLORS = [(245,117,16), (117,245,16), (16,117,245), (245,16,117), (16,245,117), (117,16,245), (245,117,16)]

def main():
    # Configure GPU if available
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
    threshold = 0.8
    n_frame_without_hands = 0
    while True:
        # ESC (27) to exit teh loop
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

        # Extract keypoints
        # keypoints = extract_keypoints(results)
        keypoints = preprocess_landmarks(results)

        #TODO: 2
        if  results.left_hand_landmarks is not None or results.right_hand_landmarks is not None:
            sequence.append(keypoints)
            sequence = sequence[-25:]
            print(f"Sequence length: {len(sequence)}")
            n_frame_without_hands = 0
        else:
            print(f"No hands detected")
            n_frame_without_hands += 1
            if n_frame_without_hands > 5:
                sequence = []
                n_frame_without_hands = 0
                print(f"Sequence reset due to no hands detected")
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