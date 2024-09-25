from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import mediapipe as mp
import math
import json
import numpy as np
import time
import os

app = Flask(__name__)

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 손 인식 모델
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 얼굴 정보를 저장할 json 파일 경로
json_file = 'eyes.json'

# 요가 자세 이미지 로드
standard_pose_image_path = "C:/Capstone2/static/yoga_posture/dataset/vriksasana/1-0.png"
standard_pose_image = cv2.imread(standard_pose_image_path)
standard_pose_image = cv2.resize(standard_pose_image, (640, 480))
standard_pose_image_rgb = cv2.cvtColor(standard_pose_image, cv2.COLOR_BGR2RGB)

with mp_pose.Pose(static_image_mode=True) as pose:
    standard_results = pose.process(standard_pose_image_rgb)
standard_pose_landmarks = standard_results.pose_landmarks

# 손가락 상태 판별 함수
def is_finger_up(landmarks, finger_tip, finger_pip):
    return landmarks[finger_tip].y < landmarks[finger_pip].y

def calculate_3d_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# 얼굴 정보 불러오는 함수
def load_face_data(user_id):
    try:
        with open(json_file, 'r') as f:
            face_data = json.load(f)
            return face_data.get(user_id, {}).get('eye_distance')
    except (FileNotFoundError, json.JSONDecodeError):
        return None

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/video_feed')
def video_feed():
    cap = cv2.VideoCapture(0)
    detection_status = {"detected": False, "action": ""}
    
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
                middle_finger_up = is_finger_up(landmarks, MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP)
                ring_finger_down = not is_finger_up(landmarks, RING_FINGER_TIP, RING_FINGER_PIP)
                pinky_finger_down = not is_finger_up(landmarks, PINKY_FINGER_TIP, PINKY_FINGER_PIP)

                if index_finger_up and not middle_finger_up and ring_finger_down and pinky_finger_down:
                    detection_status["detected"] = True
                    detection_status["action"] = "register"
                    cap.release()
                    cv2.destroyAllWindows()
                    return jsonify(detection_status)

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

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({"error": "아이디를 입력해야 합니다."}), 400

    # 얼굴 정보 저장 로직 추가
    cap = cv2.VideoCapture(0)
    frame_count = 0
    required_frames = 30
    eye_distance = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            return jsonify({"error": "카메라 영상을 캡처하는 데 실패했습니다."})

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
                    break

        if frame_count >= required_frames:
            break

    cap.release()

    if eye_distance is None:
        return jsonify({"error": "얼굴 인식을 실패했습니다. 다시 시도해주세요."}), 400

    try:
        with open(json_file, 'r+') as f:
            try:
                face_data = json.load(f)
            except json.JSONDecodeError:
                face_data = {}

            if user_id in face_data:
                return jsonify({"error": "이미 등록된 아이디입니다."}), 400

            face_data[user_id] = {"eye_distance": eye_distance}

            f.seek(0)
            json.dump(face_data, f)
            f.truncate()

    except FileNotFoundError:
        with open(json_file, 'w') as f:
            face_data = {user_id: {"eye_distance": eye_distance}}
            json.dump(face_data, f)

    return jsonify({"message": "아이디가 성공적으로 등록되었습니다.", "eye_distance": eye_distance}), 200

@app.route('/login', methods=['POST'])
def login():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    required_frames = 30

    try:
        with open(json_file, 'r') as f:
            face_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
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

                    for user_id, user_data in face_data.items():
                        stored_eye_distance = user_data.get('eye_distance')
                        if abs(stored_eye_distance - eye_distance) < 0.05:
                            cap.release()
                            return redirect(url_for('yoga'))  # 로그인 성공 시 리다이렉트
                        
                    # 얼굴 정보가 일치하지 않으면 카메라 재실행하도록 에러 반환
                    return jsonify({"error": "얼굴 정보가 일치하지 않습니다."})
        else:
            frame_count = 0

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"error": "Failed to login."})

@app.route('/yoga')
def yoga():
    return render_template('yoga.html')

@app.route('/game')
def game():
    pose_name = os.path.basename(standard_pose_image_path).split('.')[0]
    return render_template('game.html', pose_name=pose_name)

@app.route('/video_feed_yoga')
def video_feed_yoga():
    return Response(gen_yoga_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_yoga_frames():
    cap = cv2.VideoCapture(0)

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    joint_list = [
        (11, 13, 15),
        (12, 14, 16),
        (23, 25, 27),
        (24, 26, 28)
    ]
    holding_time = 3
    correct_start_time = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            all_angles_correct = True

            for joints in joint_list:
                std_angles = calculate_angle(
                    [standard_pose_landmarks.landmark[joints[0]].x, standard_pose_landmarks.landmark[joints[0]].y],
                    [standard_pose_landmarks.landmark[joints[1]].x, standard_pose_landmarks.landmark[joints[1]].y],
                    [standard_pose_landmarks.landmark[joints[2]].x, standard_pose_landmarks.landmark[joints[2]].y]
                )

                user_angles = calculate_angle(
                    [results.pose_landmarks.landmark[joints[0]].x, results.pose_landmarks.landmark[joints[0]].y],
                    [results.pose_landmarks.landmark[joints[1]].x, results.pose_landmarks.landmark[joints[1]].y],
                    [results.pose_landmarks.landmark[joints[2]].x, results.pose_landmarks.landmark[joints[2]].y]
                )

                if abs(std_angles - user_angles) > 15:
                    color = (0, 0, 255)
                    all_angles_correct = False
                else:
                    color = (0, 255, 0)

                # 관절에 색상 표시
                cv2.circle(frame, (int(results.pose_landmarks.landmark[joints[0]].x * frame.shape[1]),
                                   int(results.pose_landmarks.landmark[joints[0]].y * frame.shape[0])), 10, color, -1)
                cv2.circle(frame, (int(results.pose_landmarks.landmark[joints[1]].x * frame.shape[1]),
                                   int(results.pose_landmarks.landmark[joints[1]].y * frame.shape[0])), 10, color, -1)
                cv2.circle(frame, (int(results.pose_landmarks.landmark[joints[2]].x * frame.shape[1]),
                                   int(results.pose_landmarks.landmark[joints[2]].y * frame.shape[0])), 10, color, -1)

            if all_angles_correct:
                if correct_start_time is None:
                    correct_start_time = time.time()
                elapsed_time = time.time() - correct_start_time

                # 자세가 정확할 때 경과 시간 표시
                cv2.putText(frame, f'Time: {elapsed_time:.2f} sec', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if elapsed_time >= holding_time:
                    cv2.putText(frame, 'Pose Correct', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cap.release()
                    return redirect(url_for('game'))  # 게임 페이지로 이동
            else:
                correct_start_time = None

        combined_image = np.hstack((standard_pose_image, frame))

        # 이미지를 바이너리로 변환하여 브라우저로 전송
        ret, buffer = cv2.imencode('.jpg', combined_image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

if __name__ == "__main__":
    app.run(debug=True)
