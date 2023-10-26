from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import model_from_json
from function import*

app = Flask(__name__, static_folder='static')

# Load your sign language recognition model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("new_model2.h5")

# Create a list of sign language labels
sign_language_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize detection variables
sequence = []
predicted_sign = ''
threshold = 0.8



# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)


# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

@app.route('/')
def index():
    return render_template('webcam.html')

def gen_frames():
    global sequence, predicted_sign
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            cropframe = frame[40:400, 0:300]
            frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
            image, results = mediapipe_detection(cropframe, hands)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            try:
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predicted_sign = sign_language_labels[np.argmax(res)]

                cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
                cv2.putText(frame, "Predicted Sign: " + predicted_sign, (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            except Exception as e:
                pass

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

#Link to the site: http://127.0.0.1:5000