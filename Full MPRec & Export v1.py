"""
Utilizzo modulo MEDIAPIPE per analisi pose mani.
Cattura i landmarks e li salva in coords.csv.
"""

import mediapipe as mp
import cv2
import csv
import os
import numpy as np
from pathlib import Path

# --- Path Setup ---
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
print(path + ' --> ' + filename + "\n")

test_file = Path(path) / 'full_coords.csv'   # file su cui salvare le coordinate landmarks
print(f"Salvataggio su: {test_file}")
print("Esiste già file CSV ?:", test_file.exists())

# --- Constants for landmark counts --- GPT
NUM_POSE_LANDMARKS = 33
NUM_FACE_LANDMARKS = 468
NUM_HAND_LANDMARKS = 21
counter=0

# --- Landmark extraction function --- from GPT per settare a 0 coordinate
def extract_landmark_list(landmarks, expected_len):
    if landmarks:
        return list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten())
    else:
        return [0.0] * (expected_len * 4)


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


print('Inserisci nome classe da registrare:')
class_name = input().strip()
print('Procedo con registrazione classe:', class_name)


cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Disegno landmarks
        #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # estrae landmarks e mette a zero quando nulli
        pose_row = extract_landmark_list(results.pose_landmarks.landmark if results.pose_landmarks else None, NUM_POSE_LANDMARKS)
        face_row = extract_landmark_list(results.face_landmarks.landmark if results.face_landmarks else None, NUM_FACE_LANDMARKS)
        lefthand_row = extract_landmark_list(results.left_hand_landmarks.landmark if results.left_hand_landmarks else None, NUM_HAND_LANDMARKS)
        righthand_row = extract_landmark_list(results.right_hand_landmarks.landmark if results.right_hand_landmarks else None, NUM_HAND_LANDMARKS)

        # unione landmarks
        row = [class_name] + pose_row + face_row + lefthand_row + righthand_row

        # Crea CSV header 
        if not test_file.exists():
            print("Creo header CSV...")
            landmarks = ['class']
            total_landmarks = NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS + 2 * NUM_HAND_LANDMARKS
            for i in range(1, total_landmarks + 1):
                landmarks += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']
            with open(test_file, mode='w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(landmarks)

        # Append dati to CSV
        with open(test_file, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(row)
            counter += 1  
        print(f'Classe : {class_name} x {counter}')   

                
        text_dati = f'Classe : {class_name} x {counter} samples'
        cv2.putText(image, text_dati, (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 220), 3, cv2.LINE_AA)
        cv2.imshow('Registrazione Coordinate', image)

        # Exit 
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
