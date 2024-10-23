from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import mediapipe as mp
import math
import json
import numpy as np
import time
import os
import speech_recognition as sr
import threading

app = Flask(__name__)

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 인식 모델
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 얼굴 정보를 저장할 json 파일 경로
json_file = 'eyes.json'

# 전역 변수
game_running = False
register_running = False
login_running = False

# 3D 거리 계산 함수
def calculate_3d_distance(point1, point2):  # 두 3D 좌표 사이의 유클리드 거리 계산
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)  # 두 점 사이의 거리를 반환

# 각도 계산 함수
def calculate_angle(a, b, c):  # 세 점의 좌표로 각도를 계산
    a = np.array(a)  # 첫 번째 좌표 배열
    b = np.array(b)  # 두 번째 좌표 배열
    c = np.array(c)  # 세 번째 좌표 배열
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])  # 두 벡터의 각도 계산
    angle = np.abs(radians * 180.0 / np.pi)  # 라디안을 각도로 변환
    
    if angle > 180.0:  # 각도가 180도보다 크면 360도에서 빼줌
        angle = 360 - angle
        
    return angle  # 계산된 각도를 반환

# 요가 자세 이미지 경로와 이미지 로드 및 전처리
standard_pose_image_path = "C:/Capstone2/static/yoga_posture/dataset/agnistambhasana/10-0.png"  # 요가 자세 이미지 경로
standard_pose_image = cv2.imread(standard_pose_image_path)  # 이미지를 읽어옴
standard_pose_image = cv2.resize(standard_pose_image, (640, 480))  # 이미지를 640x480으로 리사이징
standard_pose_image_rgb = cv2.cvtColor(standard_pose_image, cv2.COLOR_BGR2RGB)  # 이미지를 BGR에서 RGB로 변환

# 자세 인식 결과 처리
with mp_pose.Pose(static_image_mode=True) as pose:  # 자세 인식 모델을 초기화하여 사용
    standard_results = pose.process(standard_pose_image_rgb)  # 이미지에서 자세 인식을 수행
standard_pose_landmarks = standard_results.pose_landmarks  # 자세 랜드마크를 저장

# 얼굴 정보를 불러오는 함수
def load_face_data(user_id):  # 특정 user_id에 해당하는 얼굴 정보를 로드
    try:
        with open(json_file, 'r') as f:  # json 파일을 읽기 모드로 엶
            face_data = json.load(f)  # JSON 데이터를 불러옴
            return face_data.get(user_id, {})  # user_id에 해당하는 데이터를 반환, 없으면 빈 딕셔너리 반환
    except (FileNotFoundError, json.JSONDecodeError):  # 파일이 없거나 형식이 잘못된 경우
        return None  # 오류 시 None 반환

# 음성 인식 함수
def recognize_speech():
    global game_running  # 전역 변수 사용
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for the command...")
        try:
            audio = r.listen(source)
            command = r.recognize_google(audio, language="ko-KR")
            print("Recognized command:", command)
            if command == "종료":
                game_running = False  # 게임 종료 상태로 설정
            return command
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""

def recognize_speech_async():
    while game_running:
        recognize_speech()


# 음성 인식 함수 (register용)
def recognize_speech_for_register():
    global register_running
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for '회원가입' command...")
        try:
            audio = r.listen(source)
            command = r.recognize_google(audio, language="ko-KR")
            print(f"Recognized command (register): {command}")
            if "회원가입" or "회원 가입"in command:
                register_running = False  # 회원가입 완료 상태로 설정
                return True
            return False
        except sr.UnknownValueError:
            print("Could not understand audio")
            return False
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return False

def recognize_speech_async_for_register():
    while register_running:
        if recognize_speech_for_register():
            print("Register command detected")
            break


# 음성 인식 함수 (login용)
def recognize_speech_for_login():
    global login_running
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for '로그인' command...")
        try:
            audio = r.listen(source)
            command = r.recognize_google(audio, language="ko-KR")
            print(f"Recognized command (login): {command}")
            if "로그인" in command:
                login_running = False  # 로그인 완료 상태로 설정
                return True
            return False
        except sr.UnknownValueError:
            print("Could not understand audio")
            return False
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return False

def recognize_speech_async_for_login():
    while login_running:
        if recognize_speech_for_login():
            print("Login command detected")
            break

@app.route('/start_voice_recognition', methods=['POST'])
def start_voice_recognition():
    command = recognize_speech()
    
    if command == "회원가입":
        # 음성 인식이 "회원가입"일 때
        return jsonify({'action': 'register', 'command': command})
    elif command == "로그인":
        # 음성 인식이 "로그인"일 때
        return jsonify({'action': 'login', 'command': command})
    else:
        # 알 수 없는 명령어일 경우
        return jsonify({'action': 'unknown', 'command': command})


    
# 메인 페이지 렌더링
@app.route('/')  # '/' 경로로 요청이 오면 index 함수 실행
def index():
    return render_template('main.html')  # main.html을 렌더링하여 반환


# 회원가입 라우트
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



# 로그인 라우트
@app.route('/login', methods=['GET', 'POST'])  # POST와 GET 모두 허용
def login():
    if request.method == 'POST':
        # POST 방식의 로그인 처리
        # 여기서 기존의 로그인 처리를 합니다.
        cap = cv2.VideoCapture(0)
        frame_count = 0
        required_frames = 30

        try:
            with open(json_file, 'r') as f:
                face_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return jsonify({"error": "등록된 얼굴 정보가 없습니다."})

        matching_user_id = None

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

                        eye_distance = calculate_3d_distance(
                            (left_eye.x, left_eye.y, left_eye.z),
                            (right_eye.x, right_eye.y, right_eye.z)
                        )   
                        
                        for user_id, user_data in face_data.items():
                            stored_eye_distance = user_data.get('eye_distance')
                            if abs(stored_eye_distance - eye_distance) < 0.05:
                                matching_user_id = user_id
                                break
                        
                    if matching_user_id:
                        cap.release()
                        return redirect(url_for('yoga', user_id=matching_user_id))  # yoga.html로 이동

                    return jsonify({"error": "얼굴 정보가 일치하지 않습니다."})
            else:
                frame_count = 0

        cap.release()
        cv2.destroyAllWindows()
        return jsonify({"error": "Failed to login."})

    return "로그인 페이지"  # GET 방식일 경우, 로그인 페이지를 반환하거나 다른 동작 처리


@app.route('/yoga')
def yoga():
    return render_template('yoga.html')

@app.route('/game')
def game():
    global game_running
    game_running = True  # 게임 실행 상태 초기화
    threading.Thread(target=recognize_speech_async, daemon=True).start()  # 음성 인식 스레드 시작
    pose_name = os.path.basename(standard_pose_image_path).split('.')[0]
    return render_template('game.html', pose_name=pose_name)


@app.route('/video_feed_yoga')
def video_feed_yoga():
    return Response(gen_yoga_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_yoga_frames():
    global game_running  # 전역 변수 사용
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

    while cap.isOpened() and game_running:
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
                    # END_GAME 메시지를 클라이언트로 보냄
                    yield (b'--frame\r\n'
                           b'Content-Type: text/plain\r\n\r\n' + b'END_GAME' + b'\r\n')
                    cap.release()
                    break  # 루프 종료하여 카메라 종료
            else:
                correct_start_time = None

        combined_image = np.hstack((standard_pose_image, frame))
        ret, buffer = cv2.imencode('.jpg', combined_image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    cv2.destroyAllWindows()

    if not game_running:  # 음성 명령으로 '종료'가 인식되면
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + b'END_GAME' + b'\r\n')  # 종료 메시지 전송


if __name__ == "__main__":
    app.run(debug=True)