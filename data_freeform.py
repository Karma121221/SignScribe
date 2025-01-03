import cv2
import numpy as np
import mediapipe as mp
import os

# Define the directory to save data
DATA_PATH = "data"  # Change this to your desired directory

# Create the data directory if it doesn't exist
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Capture video from your webcam
cap = cv2.VideoCapture(0)

# Action label for the data
action = "freeform"

# Create a subdirectory for the action
action_dir = os.path.join(DATA_PATH, action)
os.makedirs(action_dir, exist_ok=True)

# Initialize variables
frame_num = 0
sequence_num = 0

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Extract hand landmarks for the first hand (assuming only one hand in the frame)
        landmarks = results.multi_hand_landmarks[0].landmark

        # Convert landmarks to a flat list
        landmark_list = [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks]

        # Save the landmarks as a NumPy array
        npy_file = os.path.join(action_dir, f"{sequence_num}_{frame_num}.npy")
        np.save(npy_file, landmark_list)

        frame_num += 1

    # Display the landmarks on the frame
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame
    cv2.imshow("Hand Landmarks Tracking", frame)
    sequence_length = 50

    if frame_num >= sequence_length:
        frame_num = 0
        sequence_num += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
