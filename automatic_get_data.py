import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import csv
from main import get_last_directory, load_model_lables, draw_info, select_num, extract_keypoints, draw_styled_landmarks, mediapipe_detection
import sys

# Define mediapipe model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
DATA_PATH = os.path.join('model/data')
DELAY = 0.7


def main():
    # Set up variables
    no_sequences = 5
    sequence_length = 25



    # Load actions to detect
    actions = load_model_lables()
    print(f"Loaded model labels: {actions}")

    last_dir = get_last_directory(actions)
    print(f"Last non-empty directory: {last_dir}")

    for action in actions: 
        for sequence in range(1,no_sequences+1):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(last_dir[action]+sequence)))
            except:
                pass
        last_dir[action] += 1

    # Video capture
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    mp_holistic_obj = mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
    )   

    while cv2.waitKey(30) != 32:
        ret, image_i = cap.read()
        image_i = cv2.flip(image_i, 1)
        cv2.putText(image_i, 'Press SPACE to start collection', (120,200), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 0), 4, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', image_i)
        cv2.waitKey(1)

     
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(last_dir[action], last_dir[action]+no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, mp_holistic_obj)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION {}'.format(action), (120,200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(int(DELAY*1000))
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
    

    cap.release()
    cv2.destroyAllWindows()

    print('Collection Complete')

if __name__ == '__main__':
    main()

