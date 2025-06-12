
"""
Utilizzo modulo MEDIAPIPE per analisi pose mani , corpo e volto.
Il programma utilizza un modulo di machine learning precedentemente addestrato per
riconoscere i numeri  2 3 4 fatti con la mano destra.
"""

import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import pickle
import pandas as pd
import os
""" 
il file full_language.pkl e' il risultato dell'addetramento fatto usando i programmi 
per registrare e salvare su file csv le coordinate dei landmarks 
e full_training_pose.py per addestrare gli algoritmi di ML .
"""
from pathlib import Path

#print("This file directory and name")
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
#print(path + ' --> ' + filename + "\n")
# definisco nome del file in cui verranno salvate le coordinate dei landmarks 
ai_model=path+'/full_language.pkl'
ai_model=(Path(ai_model))
print(ai_model)
print(ai_model.exists())

# --- Constants for landmark counts --- GPT
NUM_POSE_LANDMARKS = 33
NUM_FACE_LANDMARKS = 468
NUM_HAND_LANDMARKS = 21

# --- Landmark extraction function --- GPT
def extract_landmark_list(landmarks, expected_len):
    if landmarks:
        return list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten())
    else:
        return [0.0] * (expected_len * 4)

with open(ai_model, 'rb') as f:
    model = pickle.load(f)  

cap = cv2.VideoCapture(0)  

# Initiate holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Extract landmarks with fallback to zeros
        pose_row = extract_landmark_list(results.pose_landmarks.landmark if results.pose_landmarks else None, NUM_POSE_LANDMARKS)
        face_row = extract_landmark_list(results.face_landmarks.landmark if results.face_landmarks else None, NUM_FACE_LANDMARKS)
        lefthand_row = extract_landmark_list(results.left_hand_landmarks.landmark if results.left_hand_landmarks else None, NUM_HAND_LANDMARKS)
        righthand_row = extract_landmark_list(results.right_hand_landmarks.landmark if results.right_hand_landmarks else None, NUM_HAND_LANDMARKS)

        
        # Export coordinates
        try:
            
            row=pose_row+face_row+lefthand_row+righthand_row
            #print(row)
            # Make Detections
            X = pd.DataFrame([row])
            #print(X)
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, body_language_prob[1])
            #print(body_language_prob)
            #print(max(body_language_prob))
            # Grab ear coords
            """
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [640,480]).astype(int))
            
            cv2.rectangle(image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(body_language_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            """
            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        except:
            pass
                        
        cv2.imshow('POSE DECODE', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()