from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import math
import json

app = Flask(__name__)

# MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# 웹캠 설정
cap = cv2.VideoCapture(0)

# 얼굴 정보를 저장할 json 파일 경로
json_file = 'eyes.json'

def is_finger_up(landmarks, finger_tip, finger_pip):
    return landmarks[finger_tip].y < landmarks[finger_pip].y

def calculate_3d_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/video_feed')
def video_feed():
    detection_status = {"detected": False}
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark

                INDEX_FINGER_TIP = 8
                INDEX_FINGER_PIP = 6
                MIDDLE_FINGER_TIP = 12
                MIDDLE_FINGER_PIP = 10
                RING_FINGER_TIP = 16
                RING_FINGER_PIP = 14
                PINKY_FINGER_TIP = 20
                PINKY_FINGER_PIP = 18

                index_finger_up = is_finger_up(landmarks, INDEX_FINGER_TIP, INDEX_FINGER_PIP)
                middle_finger_down = not is_finger_up(landmarks, MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP)
                ring_finger_down = not is_finger_up(landmarks, RING_FINGER_TIP, RING_FINGER_PIP)
                pinky_finger_down = not is_finger_up(landmarks, PINKY_FINGER_TIP, PINKY_FINGER_PIP)

                if index_finger_up and middle_finger_down and ring_finger_down and pinky_finger_down:
                    detection_status["detected"] = True
                    cap.release()
                    cv2.destroyAllWindows()
                    try:
                        hands.close()
                    except ValueError:
                        pass
                    return jsonify(detection_status)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    try:
        hands.close()
    except ValueError:
        pass
    
    return jsonify(detection_status)

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({"error": "Invalid user ID."})

    cap.open(0)  # 카메라 다시 오픈
    initial_eye_distance = None
    frame_count = 0
    required_frames = 30  # 안정적으로 얼굴을 인식해야 하는 프레임 수 (1초 정도)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            return jsonify({"error": "Failed to capture video."})

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            frame_count += 1
            if frame_count >= required_frames:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye = face_landmarks.landmark[33]
                    right_eye = face_landmarks.landmark[263]

                    left_eye_coords = (left_eye.x, left_eye.y, left_eye.z)
                    right_eye_coords = (right_eye.x, right_eye.y, right_eye.z)

                    eye_distance = calculate_3d_distance(left_eye_coords, right_eye_coords)
                    if initial_eye_distance is None:
                        initial_eye_distance = eye_distance
                        face_data = {
                            user_id: {
                                'eye_distance': initial_eye_distance
                            }
                        }
                        with open(json_file, 'w') as f:
                            json.dump(face_data, f)
                        cap.release()
                        cv2.destroyAllWindows()
                        return jsonify({"message": f"Face data saved for user {user_id}."})
        else:
            frame_count = 0  # 얼굴이 감지되지 않으면 카운트 초기화

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"error": "Failed to save face data."})

if __name__ == "__main__":
    app.run(debug=True)