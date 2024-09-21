from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
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

# 손가락 상태 판별 함수
def is_finger_up(landmarks, finger_tip, finger_pip):
    return landmarks[finger_tip].y < landmarks[finger_pip].y

def calculate_3d_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

# 얼굴 정보 불러오는 함수
def load_face_data():
    try:
        with open(json_file, 'r') as f:
            face_data = json.load(f)
            return face_data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/video_feed')
def video_feed():
    detection_status = {"detected": False, "action": ""}  # action: register or login
    
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

                # 손가락 마디 인덱스
                INDEX_FINGER_TIP = 8
                INDEX_FINGER_PIP = 6
                MIDDLE_FINGER_TIP = 12
                MIDDLE_FINGER_PIP = 10
                RING_FINGER_TIP = 16
                RING_FINGER_PIP = 14
                PINKY_FINGER_TIP = 20
                PINKY_FINGER_PIP = 18

                index_finger_up = is_finger_up(landmarks, INDEX_FINGER_TIP, INDEX_FINGER_PIP)
                middle_finger_up = is_finger_up(landmarks, MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP)
                ring_finger_down = not is_finger_up(landmarks, RING_FINGER_TIP, RING_FINGER_PIP)
                pinky_finger_down = not is_finger_up(landmarks, PINKY_FINGER_TIP, PINKY_FINGER_PIP)

                # 검지만 펼쳤을 때: 회원가입
                if index_finger_up and not middle_finger_up and ring_finger_down and pinky_finger_down:
                    detection_status["detected"] = True
                    detection_status["action"] = "register"
                    cap.release()
                    cv2.destroyAllWindows()
                    return jsonify(detection_status)

                # 검지와 중지만 펼쳤을 때: 로그인
                if index_finger_up and middle_finger_up and ring_finger_down and pinky_finger_down:
                    detection_status["detected"] = True
                    detection_status["action"] = "login"
                    cap.release()
                    cv2.destroyAllWindows()
                    return jsonify(detection_status)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify(detection_status)

@app.route('/login', methods=['POST'])
def login():
    cap.open(0)  # 카메라 다시 오픈
    frame_count = 0
    required_frames = 30  # 안정적으로 얼굴을 인식해야 하는 프레임 수

    # eyes.json 파일에서 모든 저장된 얼굴 정보 불러오기
    face_data = load_face_data()

    if not face_data:
        return jsonify({"error": "등록된 얼굴 정보가 없습니다."})

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

                    # 등록된 얼굴 정보와 비교
                    for user_id, user_data in face_data.items():
                        stored_eye_distance = user_data.get('eye_distance')
                        if abs(stored_eye_distance - eye_distance) < 0.05:  # 허용 오차 범위 지정
                            cap.release()
                            cv2.destroyAllWindows()
                            return jsonify({"redirect_url": url_for('yoga')})  # 성공 시 yoga.html로 이동
        else:
            frame_count = 0  # 얼굴이 감지되지 않으면 카운트 초기화

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"error": "얼굴 정보가 일치하지 않습니다."})

@app.route('/yoga')
def yoga():
    return render_template('yoga.html')

if __name__ == "__main__":
    app.run(debug=True)
