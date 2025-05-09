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
from typing import List, Any 
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain 
from gtts import gTTS
from langchain_google_genai import ChatGoogleGenerativeAI
from playsound import playsound

# Define mediapipe model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
DATA_PATH = os.path.join('model/data') 
VIDEO_LENGTH = 15                               # Number of frames to save for each action

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

def squeeze_data(sequences, max_squeeze_proportion) -> np.ndarray:
    """
    Applies squeeze to the x-coordinates of the sequences based on the specified proportion.
    Args:
        sequences: A numpy array containing the sequences of keypoints.
        max_squeeze_proportion: Maximum proportion for squeezing.
    Returns:
        A numpy array containing the squeezed sequences.
    """
    n, n_frames, total_features = sequences.shape
    squeezed_sequences = np.copy(sequences)

    W = 1.0

    for i in range(n):
        w1 = np.random.uniform(0, max_squeeze_proportion)
        w2 = np.random.uniform(0, max_squeeze_proportion)
        new_W = W - (w1 + w2)
        for frame_idx in range(n_frames):
            for j in range(0, total_features, 3):
                x = sequences[i, frame_idx, j]
                squeezed_x = (x - w1) / new_W
                squeezed_sequences[i, frame_idx, j] = squeezed_x

    return squeezed_sequences

def rotate_data(sequences, rotations) -> np.ndarray:
    """
    Rotates the hand coordinates in the sequences based on the specified angles.
    Args:
        sequences: A numpy array containing the sequences of keypoints.
        rotations: A list of angles in degrees for rotation.
    Returns:
        A numpy array containing the rotated sequences."""
    pose_length = 99
    hand_length = 63

    hand_l_start = pose_length
    hand_l_end = pose_length + hand_length
    hand_r_start = pose_length + hand_length
    hand_r_end = pose_length + 2 * hand_length

    _, n_frames, _ = sequences.shape

    rotated_sequences = np.copy(sequences)
    
    for i, rot in enumerate(rotations):
        cos_theta = np.cos(np.radians(rot))
        sin_theta = np.sin(np.radians(rot))
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                  [sin_theta, cos_theta]])
        
        for frame_idx in range(n_frames):
            # Extract x, y coordinates for left and right hands
            hand_l = sequences[i, frame_idx, hand_l_start:hand_l_end].reshape(21, 3)[:, :2]
            hand_r = sequences[i, frame_idx, hand_r_start:hand_r_end].reshape(21, 3)[:, :2]

            # Rotate left hand
            rotated_hand_l = np.dot(hand_l, rotation_matrix)
            rotated_sequences[i, frame_idx, hand_l_start:hand_l_end-1:3] = rotated_hand_l[:, 0]
            rotated_sequences[i, frame_idx, hand_l_start+1:hand_l_end:3] = rotated_hand_l[:, 1]

            # Rotate right hand
            rotated_hand_r = np.dot(hand_r, rotation_matrix)
            rotated_sequences[i, frame_idx, hand_r_start:hand_r_end-1:3] = rotated_hand_r[:, 0]
            rotated_sequences[i, frame_idx, hand_r_start+1:hand_r_end:3] = rotated_hand_r[:, 1]

    return rotated_sequences

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
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark])
        pose[23:] = np.zeros((10, 4))  
    else:
        pose = np.zeros((33, 4))
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

def mirror_pose_sequence(sequences) -> np.ndarray:
    """
    Mirrors the pose sequence by swapping left and right landmarks.
    Args:
        sequences: A numpy array containing the sequences of keypoints.
    Returns:
        A numpy array containing the mirrored sequences.
    """

    n, num_frames, _ = sequences.shape
    mirrored_sequences = np.copy(sequences)

    # Pose landmark pairs to swap
    pose_landmark_pairs = [
        (11, 12),  # left_shoulder <-> right_shoulder
        (13, 14),  # left_elbow    <-> right_elbow
        (15, 16),  # left_wrist    <-> right_wrist
        (17, 18),  # left_pinky    <-> right_pinky
        (19, 20),  # left_index    <-> right_index
        (21, 22),  # left_thumb    <-> right_thumb
        (7, 8)     # left_ear      <-> right_ear
    ]

    pose_length = 33 * 3
    hand_length = 21 * 3

    left_hand_start = pose_length  # 99
    left_hand_end = pose_length + hand_length  # 99 + 63 = 162
    right_hand_start = left_hand_end  # 162
    right_hand_end = left_hand_end + hand_length  # 162 + 63 = 225

    for i in range(n):
        for frame_idx in range(num_frames):
            # Mirror Pose
            for left_idx, right_idx in pose_landmark_pairs:
                left_start = left_idx * 3
                left_end = left_start + 3
                right_start = right_idx * 3
                right_end = right_start + 3

                mirrored_sequences[i, frame_idx, left_start:left_end], \
                    mirrored_sequences[i, frame_idx, right_start:right_end] = \
                    sequences[i, frame_idx, right_start:right_end].copy(), \
                    sequences[i, frame_idx, left_start:left_end].copy()

                mirrored_sequences[i, frame_idx, left_start] *= -1
                mirrored_sequences[i, frame_idx, right_start] *= -1

            # Mirror Hands
            mirrored_sequences[i, frame_idx, left_hand_start:left_hand_end], \
                mirrored_sequences[i, frame_idx, right_hand_start:right_hand_end] = \
                sequences[i, frame_idx, right_hand_start:right_hand_end].copy(), \
                sequences[i, frame_idx, left_hand_start:left_hand_end].copy()

            # Invert the x-coordinate of the hands
            for j in range(21):
                mirrored_sequences[i, frame_idx, left_hand_start + j * 3] *= -1  # Left hand x
                mirrored_sequences[i, frame_idx, right_hand_start + j * 3] *= -1  # Right hand x

    return mirrored_sequences


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
    # print(f"Pose shape: {norm_pose.shape}, Hands shape: {norm_hands.shape}")
    return np.concatenate([norm_pose, norm_hands])

def sequential_arm_rotation(sequences, max_angle_degrees, rotation_probability) -> np.ndarray:
    """
    Sequentially rotates arm joints in a sequence of pose landmarks.
    Arms are rotated with a given probability and angle.
    Args:
        sequences (np.ndarray): Input sequences of shape (n, num_frames, 99).
        max_angle_degrees (float): Maximum angle for rotation in degrees.
        rotation_probability (float): Probability of applying rotation.
    Returns:
        np.ndarray: Rotated sequences of the same shape as input.
    """
    n, num_frames, _ = sequences.shape
    rotated_sequences = np.copy(sequences)

    arm_joints = [11, 12, 13, 14, 15, 16] # Indices for left and right shoulders, elbows, and wrists

    for i in range(n):
        for frame_idx in range(num_frames):
            prev_coords = None  # Store coordinates of the "previous" joint

            for joint_idx in arm_joints:
                start_idx = joint_idx * 3
                end_idx = start_idx + 3

                rotated_coords = sequences[i, frame_idx, start_idx:end_idx].copy() # Define it here

                if np.random.rand() < rotation_probability:
                    theta = np.radians(np.random.uniform(-max_angle_degrees, max_angle_degrees))
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)
                    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                              [sin_theta, cos_theta]])

                    joint_coords = sequences[i, frame_idx, start_idx:end_idx]  # Extract x, y, z

                    # Rotate relative to the previous joint (if available)
                    if prev_coords is not None:
                        offset = joint_coords[:2] - prev_coords[:2]  # Offset of x, y
                        rotated_offset = np.dot(offset, rotation_matrix)
                        rotated_coords[:2] = prev_coords[:2] + rotated_offset
                    else:
                        rotated_coords[:2] = np.dot(joint_coords[:2], rotation_matrix)

                    rotated_sequences[i, frame_idx, start_idx:start_idx + 2] = rotated_coords[:2]  # Update x, y

                # Update prev_coords for the next joint
                
                prev_coords = sequences[i, frame_idx, start_idx:end_idx].copy()

    return rotated_sequences



def gloss_list_to_speech(gloss_text_list: List[str], llm: ChatGoogleGenerativeAI, template_str: str) -> gTTS:
    """
    Converts a sequence of glosses into natural language text and then into speech.

    Args:
        gloss_text_list (List[str]): A list of gloss strings.
        llm (ChatGoogleGenerativeAI): The language model to use to convert glosses to natural language.
                                         (Updated type hint)
        template_str (str): The prompt template string to use for the language model.
                            (Parameter name changed from 'template' to 'template_str' to avoid confusion
                             with a potentially externally defined 'template' variable and to clarify it's a string)

    Returns:
        gTTS: The generated gTTS audio object.
    """
    # The prompt is created using the 'template_str' passed as an argument
    prompt = PromptTemplate(template=template_str, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    gloss_text = ' | '.join(gloss_text_list)

    # The output of invoke with LLMChain is usually a dictionary with a 'text' key
    # or for some chat models it might be a message object.
    # For ChatGoogleGenerativeAI used with LLMChain, it should return {'text': 'response...'}
    response = llm_chain.invoke(gloss_text)
    natural_text = response['text']

    language = 'en'
    natural_audio = gTTS(text=natural_text, lang=language, slow=False)

    # Create a safer file name, replacing invalid characters if necessary
    safe_audio_name = gloss_text.replace(' | ', '_').replace('?', '').replace("'", "")
    audio_name = safe_audio_name + '.mp3'

    natural_audio.save(audio_name)
    print(f"Audio saved as: {audio_name}") # Added print for confirmation
    print(f"Generated natural text: {natural_text}") # Added print for debugging

    try:
        playsound(audio_name)
    except Exception as e:
        print(f"Errore durante la riproduzione dell'audio con playsound: {e}")

    return natural_audio

if __name__ == '__main__':
    main()

