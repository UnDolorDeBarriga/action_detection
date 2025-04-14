import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import csv
import math
import copy
# Define mediapipe model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
DATA_PATH = os.path.join('model/data') 
VIDEO_LENGTH = 30                               # Number of frames to save for each action

def main():
    # Load actions to detect
    actions = load_model_lables()
    print(f"Loaded model labels: {actions}")

    last_dir = get_last_directory(actions)
    print(f"Last non-empty directory: {last_dir}")


    # Video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    mp_holistic_obj = mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
    )

    number = -1
    frame_count = 0
    save_sequence = False
    action_num = 0
    while True:
        # If ESC (27) is pressed, exit the loop
        key = cv2.waitKey(16)
        if key == 27:
            break
        number, o_number = select_num(key, number)
        
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break
        
        # Make detections
        image, results = mediapipe_detection(frame, mp_holistic_obj)
        # Draw landmarks
        draw_styled_landmarks(image, results)


        # if number != -1 and not save_sequence:
        #     # Start a new sequence
        #     save_sequence = True
        #     frame_count = 0
        #     action_num = number 
        #     action = actions[number]
        #     last_dir[action] += 1
        #     # Create directory
        #     os.makedirs(os.path.join(DATA_PATH, action, str(last_dir[action])), exist_ok=True)

        #     # Start colecting
        #     cv2.putText(image, 'STARTING COLLECTION', (120,200), 
        #                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
        #     # Show to screen
        #     cv2.imshow('OpenCV Feed', image)
        #     cv2.waitKey(500)


        # else:
        #     if save_sequence:
        #         image = draw_info(image, action, last_dir[action])
        #         # Export keypoints
        #         keypoints = extract_keypoints(results)
        #         npy_path = os.path.join(DATA_PATH, action, str(last_dir[action]), str(frame_count))
        #         np.save(npy_path, keypoints)
        #         # Increment frame count
        #         frame_count += 1
            
        #     if frame_count == VIDEO_LENGTH:
        #         save_sequence = False
        #         frame_count = 0
        #         cv2.putText(image, 'STOP COLECTING {}'.format(action), (120,200), 
        #                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 4, cv2.LINE_AA)
                
        #         print(f"Finished saving sequence for action {action}")
        #         cv2.imshow('OpenCV Feed', image)
        #         cv2.waitKey(500)kp_hands
        p_landmarks = preprocess_landmarks(results)
        # print(kp_hands)       # Show to screen
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
    #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
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
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          ) 
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

# From Here
def calculate_head_unit(pose_landmarks):
    """
    Calcola l'unità di misura 'head unit' basata sulla distanza tra le spalle.
    Args:
        pose_landmarks: Oggetto landmarks della posa da MediaPipe.
    Returns:
        La dimensione della head unit, o None se le spalle non sono visibili.
    """
    mp_pose = mp.solutions.pose.PoseLandmark
    if pose_landmarks:
        landmarks = pose_landmarks.landmark
        try:
            left_shoulder = landmarks[mp_pose.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.RIGHT_SHOULDER.value]
            # Verifica la visibilità (opzionale ma consigliato)
            if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
                 print("Warning: Shoulders not clearly visible.")
                 # Potresti decidere di non calcolare se la visibilità è bassa
                 # return None
            
            shoulder_distance = calculate_distance(left_shoulder, right_shoulder)
            if shoulder_distance < 1e-6: # Evita divisione per zero o valori minuscoli
                 return None
            head_unit = shoulder_distance / 2.0
            return head_unit
        except (IndexError, AttributeError):
            # Landmark non trovati o oggetto non valido
            print("Error: Could not find shoulder landmarks.")
            return None
    return None

def rotate_point(x, y, angle_rad, center_x=0.5, center_y=0.5):
    """Applies 2D rotation to a point about a center."""
    x_shifted = x - center_x
    y_shifted = y - center_y
    x_rotated = x_shifted * math.cos(angle_rad) - y_shifted * math.sin(angle_rad)
    y_rotated = y_shifted * math.cos(angle_rad) + x_shifted * math.sin(angle_rad)
    return x_rotated + center_x, y_rotated + center_y

def squeeze_point_x(x, w1, w2, W=1.0):
    """Applies horizontal squeeze to the x coordinate."""
    # Ensure the denominator is not too close to zero to avoid instability
    denominator = W - (w1 + w2)
    if abs(denominator) < 1e-6: # Avoid division by zero or very small values
        # In this limiting case (w1+w2 almost equal to W), you could return 0.5
        # or handle as you like. Here we return the original value.
        print(f"Warning: Squeeze denominator near zero ({denominator}). Skipping squeeze for this x.")
        return x
    return (x - w1) / denominator

def apply_spatial_augmentations(results, angle_deg, squeeze_w1, squeeze_w2):
    """
        Apply rotation and squeeze to MediaPipe landmarks.

        Args:
            results: The output of the MediaPipe Holistic model.
            angle_deg: The rotation angle in degrees (randomly generated per instance).
            squeeze_w1: Left squeeze proportion (randomly generated per instance).
            squeeze_w2: Right squeeze proportion (randomly generated per instance).

        Returns:
            A concatenated NumPy array of augmented landmarks [pose, lh, rh].
        """
    angle_rad = math.radians(angle_deg)
    W = 1.0 # Larghezza normalizzata del frame

    augmented_pose = []
    augmented_lh = []
    augmented_rh = []

    # Processa Pose Landmarks
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            x, y = lm.x, lm.y
            # 1. Applica Rotazione
            x_rot, y_rot = rotate_point(x, y, angle_rad)
            # 2. Applica Squeeze alla coordinata x ruotata
            x_sq = squeeze_point_x(x_rot, squeeze_w1, squeeze_w2, W)
            # Conserva y ruotata, z e visibilità originali
            augmented_pose.extend([x_sq, y_rot, lm.z, lm.visibility])
    else:
        augmented_pose = np.zeros(33 * 4)

    # Processa Left Hand Landmarks
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            x, y = lm.x, lm.y
            # 1. Applica Rotazione
            x_rot, y_rot = rotate_point(x, y, angle_rad)
            # 2. Applica Squeeze
            x_sq = squeeze_point_x(x_rot, squeeze_w1, squeeze_w2, W)
            augmented_lh.extend([x_sq, y_rot, lm.z])
    else:
        augmented_lh = np.zeros(21 * 3)

    # Processa Right Hand Landmarks
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            x, y = lm.x, lm.y
            # 1. Applica Rotazione
            x_rot, y_rot = rotate_point(x, y, angle_rad)
            # 2. Applica Squeeze
            x_sq = squeeze_point_x(x_rot, squeeze_w1, squeeze_w2, W)
            augmented_rh.extend([x_sq, y_rot, lm.z])
    else:
        augmented_rh = np.zeros(21 * 3)

    # Concatena i risultati (assicurandoti che siano array numpy)
    pose_arr = np.array(augmented_pose).flatten()
    lh_arr = np.array(augmented_lh).flatten()
    rh_arr = np.array(augmented_rh).flatten()

    return np.concatenate([pose_arr, lh_arr, rh_arr])

# --- Esempio di utilizzo nel tuo ciclo di processing ---

def normalize_hands_coordinates(kp_hands) -> np.ndarray:
    """
    Normalizes the coordinates of the left and right hand keypoints. It is normalized by the distance between the wrist and furthest point.
    The coordinates are converted to relative coordinates based on the wrist position.
    Args:
        kp_hands: A tuple containing left and right hand keypoints.
    Returns:
        A numpy array containing the normalized coordinates of both hands.
    """
    temp_lh = copy.deepcopy(kp_hands[0])
    temp_rh = copy.deepcopy(kp_hands[1])

    # Convert to relative coordinates
    lBase = (0, 0, 0)
    rBase = (0, 0, 0)
    l_dist = 0.0
    r_dist = 0.0
    for i in range(20):
        if i == 0:
            lBase = (temp_lh[i][0], temp_lh[i][1], temp_lh[i][2])
            rBase = (temp_rh[i][0], temp_rh[i][1], temp_rh[i][2])

        t_l_dist = calculate_distance(temp_lh[i], lBase)
        t_r_dist = calculate_distance(temp_rh[i], rBase)
        if (t_l_dist > l_dist):
            l_dist = t_l_dist
        if (t_r_dist > r_dist):
            r_dist = t_r_dist

        temp_lh[i] = temp_lh[i] - lBase
        temp_rh[i] = temp_rh[i] - rBase

    # Convert to a one-dimensional list
    temp_lh = temp_lh.flatten()
    temp_rh = temp_rh.flatten()

    # Normalization
    def l_normalize_(n):
        return n / l_dist if l_dist != 0 else n
    def r_normalize_(n):
        return n / r_dist if r_dist != 0 else n

    temp_lh = list(map(l_normalize_, temp_lh))
    temp_rh = list(map(r_normalize_, temp_rh))

    return np.array([temp_lh, temp_rh]).flatten()

def normalize_pose_coordinates(kp_pose) -> np.ndarray:
    """
    Normalizes the coordinates of the pose keypoints. It is normalized by the distance between the shoulders.
    The coordinates are converted to relative coordinates based on the nose position.
    Args:
        kp_pose: A numpy array containing pose keypoints.
    Returns:
        A numpy array containing the normalized coordinates of the pose keypoints.
    """
    shoulder_distance = calculate_distance(kp_pose[11], kp_pose[12])
    temp_pose = kp_pose[:, :-1]
    # Convert to relative coordinates
    base = (0, 0, 0)
    for i in range(33):
        if i == 0:
            base = (temp_pose[i][0], temp_pose[i][1], temp_pose[i][2])
        temp_pose[i] = temp_pose[i] - base

    # Convert to a one-dimensional list
    temp_pose = temp_pose.flatten()

    # Normalization
    def normalize_(n):
        return n / shoulder_distance if shoulder_distance != 0 else n

    temp_pose = list(map(normalize_, temp_pose))

    return np.array(temp_pose)

def calculate_distance(landmark1, landmark2) -> float:
    """
    Calculates the Euclidean distance between two landmarks.
    Args:
        landmark1: First landmark (x, y, z).
        landmark2: Second landmark (x, y, z).
    Returns:
        The Euclidean distance between the two landmarks.
    """
    visibility_th = 0.3
    return math.sqrt((landmark1[0] - landmark2[0])**2 +
                     (landmark1[1] - landmark2[1])**2 +
                     (landmark1[2] - landmark2[2])**2)

def extract_keypots_cordinates(results) -> tuple:
    """
    Extracts keypoints from the results and returns them as separate  matrices.
    Args:
        results: The detection results containing landmarks.
    Returns:
        A tuple containing pose, left hand, and right hand keypoints.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33,4))
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21,3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21,3))
    return (pose, np.array([lh, rh]))

def extract_keypoints(results) -> np.ndarray:
    """
    Extracts keypoints from the results and flattens them into a single array.
    Args:
        results: The detection results containing landmarks.
    Returns:
        A flattened array of keypoints.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    #return np.concatenate([pose, face, lh, rh])
    return np.concatenate([pose,lh, rh])

def select_num(key, number) -> tuple:
    """
    Selects a number based on the key pressed.
    Args:
        key: The key pressed.
        number: The current number.
    Returns:
        number: The selected number.
        o_number: The original number.
    """
    o_number = number
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
        o_number = number
    # if 97 <= key <= 122:
    #     number = key - 87
    #     o_number = number
    return number, o_number

def draw_info(image, action, sequence) -> np.ndarray:
    """
    Draws information on the image.
    Args:
        image: The input image.
        action: The action name.
        sequence: The number of the sequence.
    Returns:
        image: The processed image with information drawn.
    """
    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return image

import os

def get_last_directory(actions) -> dict:
    """
    Gets the directory with the highest numerical name for each action.
    Creates directories for each action if they don't exist.
    
    Args:
        actions: List of action names.
    
    Returns:
        last_dir: Dictionary with action names as keys and the highest numbered directory as values,
                  or -1 if no directories with numeric names exist.
    """
    last_dir = {}
    
    for action in actions: 
        action_path = os.path.join(DATA_PATH, action)
        os.makedirs(action_path, exist_ok=True)

        max_numeric_folder = -1
        for folder in os.listdir(action_path):
            folder_path = os.path.join(action_path, folder)
            if os.path.isdir(folder_path) and folder.isdigit():
                if os.listdir(folder_path):  # ensure it's not empty
                    max_numeric_folder = max(max_numeric_folder, int(folder))
        
        last_dir[action] = max_numeric_folder

    return last_dir

def preprocess_landmarks(results) -> np.ndarray:
    """
    Preprocesses the landmarks by applying spatial augmentations and normalizing the coordinates.
    Args:
        results: The detection results containing landmarks.
    Returns:
        A concatenated numpy array of normalized pose and hand coordinates.
    """
    kp_pose, kp_hands = extract_keypots_cordinates(results)
    norm_hands = normalize_hands_coordinates(kp_hands)
    norm_pose = normalize_pose_coordinates(kp_pose)
    return np.concatenate([norm_pose, norm_hands])


if __name__ == '__main__':
    main()

