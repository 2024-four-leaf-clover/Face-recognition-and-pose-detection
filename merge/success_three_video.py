from flask import Flask, render_template, redirect, url_for
import cv2
import mediapipe as mp
import threading

app = Flask(__name__)

gesture_detected = False

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def is_finger_extended(finger_tip, finger_dip, palm_direction):
    return finger_tip.y < finger_dip.y if palm_direction > 0 else finger_tip.y > finger_dip.y

def is_hand_back_facing(wrist, middle_finger_base):
    return wrist.z > middle_finger_base.z

def is_three_fingers_extended(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    middle_finger_base = hand_landmarks.landmark[9]
    palm_direction = 1 if is_hand_back_facing(wrist, middle_finger_base) else -1
    
    is_index_extended = is_finger_extended(hand_landmarks.landmark[8], hand_landmarks.landmark[6], palm_direction)
    is_middle_extended = is_finger_extended(hand_landmarks.landmark[12], hand_landmarks.landmark[10], palm_direction)
    is_ring_extended = is_finger_extended(hand_landmarks.landmark[16], hand_landmarks.landmark[14], palm_direction)
    is_pinky_folded = not is_finger_extended(hand_landmarks.landmark[20], hand_landmarks.landmark[18], palm_direction)
    
    return is_index_extended and is_middle_extended and is_ring_extended and is_pinky_folded

def webcam_thread():
    global gesture_detected
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while not gesture_detected:
            success, image = cap.read()
            if not success:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if is_three_fingers_extended(hand_landmarks):
                        gesture_detected = True
                        break
            if gesture_detected:
                break
    cap.release()

@app.route('/')
def home():
    return redirect(url_for('camera'))

@app.route('/camera')
def camera():
    global gesture_detected
    gesture_detected = False
    threading.Thread(target=webcam_thread, daemon=True).start()
    return render_template('yoga.html')

@app.route('/check_gesture')
def check_gesture():
    if gesture_detected:
        return redirect(url_for('main'))
    return '', 204  # No Content

@app.route('/main')
def main():
    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True)
