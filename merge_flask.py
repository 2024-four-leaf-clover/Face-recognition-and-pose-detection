# Flask와 필요한 모듈들 가져오기
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2  # OpenCV를 사용한 영상 처리
import mediapipe as mp  # MediaPipe 라이브러리
import math  # 수학 연산을 위한 모듈
import json  # JSON 파일을 다루기 위한 모듈
import numpy as np  # 배열 처리와 수학 연산을 위한 NumPy
import time  # 시간 측정을 위한 모듈
import os  # 운영체제 관련 기능을 위한 모듈

# Flask 앱 초기화
app = Flask(__name__)

# MediaPipe 초기화
mp_hands = mp.solutions.hands  # 손 인식 모델 초기화
mp_face_mesh = mp.solutions.face_mesh  # 얼굴 인식 모델 초기화
mp_pose = mp.solutions.pose  # 자세 인식 모델 초기화
mp_drawing = mp.solutions.drawing_utils  # MediaPipe의 그리기 유틸리티

# 손 인식 모델 객체 생성
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)  # 손 인식 설정
# 얼굴 메시 모델 생성
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)  
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)  # 자세 인식 모델 생성

# 얼굴 정보를 저장할 json 파일 경로 설정
json_file = 'eyes.json'

# 요가 자세 이미지 경로와 이미지 로드 및 전처리
standard_pose_image_path = "C:/Capstone2/static/yoga_posture/dataset/vriksasana/1-0.png"  # 요가 이미지 경로
standard_pose_image = cv2.imread(standard_pose_image_path)  # 이미지 읽기
standard_pose_image = cv2.resize(standard_pose_image, (640, 480))  # 이미지 크기 조정
standard_pose_image_rgb = cv2.cvtColor(standard_pose_image, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환

# 자세 인식 결과 처리
with mp_pose.Pose(static_image_mode=True) as pose:
    standard_results = pose.process(standard_pose_image_rgb)  # 이미지에서 자세 인식
standard_pose_landmarks = standard_results.pose_landmarks  # 자세 랜드마크 추출

# 손가락 상태 판별 함수 정의
def is_finger_up(landmarks, finger_tip, finger_pip):
    return landmarks[finger_tip].y < landmarks[finger_pip].y  # 손가락 끝이 더 위에 있으면 True

# 3D 거리 계산 함수
def calculate_3d_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)  # 두 점 사이 거리 계산

# 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)  # 좌표를 배열로 변환
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])  # 각도 계산
    angle = np.abs(radians * 180.0 / np.pi)  # 라디안을 각도로 변환
    
    if angle > 180.0:  # 180도를 넘으면 360도에서 빼줌
        angle = 360 - angle
        
    return angle  # 계산된 각도 반환

# 얼굴 정보를 불러오는 함수
def load_face_data(user_id):
    try:
        with open(json_file, 'r') as f:  # json 파일을 읽기 모드로 열기
            face_data = json.load(f)  # JSON 데이터를 불러오기
            return face_data.get(user_id, {}).get('eye_distance')  # 아이디에 해당하는 눈 사이 거리 반환
    except (FileNotFoundError, json.JSONDecodeError):  # 파일을 찾을 수 없거나 JSON 형식이 잘못된 경우
        return None  # None 반환

# 메인 페이지 렌더링
@app.route('/')
def index():
    return render_template('main.html')  # main.html 파일 렌더링

# 비디오 피드를 위한 라우트
@app.route('/video_feed')
def video_feed():
    cap = cv2.VideoCapture(0)  # 웹캠을 열기
    detection_status = {"detected": False, "action": ""}  # 감지 상태 초기화
    
    while True:
        success, frame = cap.read()  # 프레임 읽기
        if not success:
            break  # 읽기에 실패하면 종료

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임을 RGB로 변환
        image.flags.writeable = False  # 이미지 쓰기 금지 설정
        results = hands.process(image)  # 손 인식 결과 처리

        if results.multi_hand_landmarks:  # 여러 손의 랜드마크가 있으면
            for hand_landmarks in results.multi_hand_landmarks:  # 각 손에 대해 반복
                landmarks = hand_landmarks.landmark  # 손 랜드마크 추출

                # 손가락 끝과 PIP 마디의 인덱스 정의
                INDEX_FINGER_TIP = 8
                INDEX_FINGER_PIP = 6
                MIDDLE_FINGER_TIP = 12
                MIDDLE_FINGER_PIP = 10
                RING_FINGER_TIP = 16
                RING_FINGER_PIP = 14
                PINKY_FINGER_TIP = 20
                PINKY_FINGER_PIP = 18

                # 각 손가락의 상태 판별
                index_finger_up = is_finger_up(landmarks, INDEX_FINGER_TIP, INDEX_FINGER_PIP)  # 검지가 위로 올라갔는지 확인
                middle_finger_up = is_finger_up(landmarks, MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP)  # 중지가 위로 올라갔는지 확인
                ring_finger_down = not is_finger_up(landmarks, RING_FINGER_TIP, RING_FINGER_PIP)  # 약지가 내려갔는지 확인
                pinky_finger_down = not is_finger_up(landmarks, PINKY_FINGER_TIP, PINKY_FINGER_PIP)  # 새끼손가락이 내려갔는지 확인

                # 검지만 위로 올라간 경우
                if index_finger_up and not middle_finger_up and ring_finger_down and pinky_finger_down:
                    detection_status["detected"] = True
                    detection_status["action"] = "register"  # 등록 동작 감지
                    cap.release()  # 웹캠 릴리즈
                    cv2.destroyAllWindows()  # 모든 윈도우 닫기
                    return jsonify(detection_status)  # JSON 형태로 반환

                # 검지와 중지가 위로 올라간 경우
                if index_finger_up and middle_finger_up and ring_finger_down and pinky_finger_down:
                    detection_status["detected"] = True
                    detection_status["action"] = "login"  # 로그인 동작 감지
                    cap.release()  # 웹캠 릴리즈
                    cv2.destroyAllWindows()  # 모든 윈도우 닫기
                    return jsonify(detection_status)  # JSON 형태로 반환

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 루프 종료
            break

    cap.release()  # 웹캠 릴리즈
    cv2.destroyAllWindows()  # 모든 윈도우 닫기
    return jsonify(detection_status)  # 감지 상태 반환

# 사용자 등록 라우트
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()  # 클라이언트로부터 받은 JSON 데이터 파싱
    user_id = data.get('user_id')  # user_id 가져오기

    if not user_id:  # user_id가 없으면
        return jsonify({"error": "아이디를 입력해야 합니다."}), 400  # 에러 메시지 반환

    # 얼굴 정보 저장 로직 추가
    cap = cv2.VideoCapture(0)  # 웹캠 열기
    frame_count = 0  # 프레임 카운터 초기화
    required_frames = 30  # 필요한 프레임 수
    eye_distance = None  # 눈 사이 거리 초기화

    while cap.isOpened():
        success, frame = cap.read()  # 프레임 읽기
        if not success:
            return jsonify({"error": "카메라 영상을 캡처하는 데 실패했습니다."})  # 실패 시 에러 반환

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임을 RGB로 변환
        results = face_mesh.process(rgb_frame)  # 얼굴 인식 처리

        if results.multi_face_landmarks:  # 얼굴 랜드마크가 있으면
            frame_count += 1  # 프레임 카운트 증가
            if frame_count >= required_frames:  # 필요한 프레임 수에 도달했을 때
                for face_landmarks in results.multi_face_landmarks:  # 각 얼굴에 대해
                    left_eye = face_landmarks.landmark[33]  # 왼쪽 눈 좌표 추출
                    right_eye = face_landmarks.landmark[263]  # 오른쪽 눈 좌표 추출

                    left_eye_coords = (left_eye.x, left_eye.y, left_eye.z)  # 왼쪽 눈 좌표를 튜플로 저장
                    right_eye_coords = (right_eye.x, right_eye.y, right_eye.z)  # 오른쪽 눈 좌표를 튜플로 저장

                    eye_distance = calculate_3d_distance(left_eye_coords, right_eye_coords)  # 두 눈 사이 거리 계산
                    break

        if frame_count >= required_frames:  # 필요한 프레임 수에 도달했을 때 루프 종료
            break

    cap.release()  # 웹캠 릴리즈

    if eye_distance is None:  # 눈 사이 거리를 계산하지 못한 경우
        return jsonify({"error": "얼굴 인식을 실패했습니다. 다시 시도해주세요."}), 400  # 에러 메시지 반환

    try:
        with open(json_file, 'r+') as f:  # json 파일을 읽고 쓸 수 있는 모드로 열기
            try:
                face_data = json.load(f)  # 기존 얼굴 데이터를 불러오기
            except json.JSONDecodeError:  # 파일이 비어 있거나 형식이 잘못된 경우
                face_data = {}  # 빈 데이터로 초기화

            if user_id in face_data:  # 이미 등록된 아이디가 있는지 확인
                return jsonify({"error": "이미 등록된 아이디입니다."}), 400  # 에러 메시지 반환

            face_data[user_id] = {"eye_distance": eye_distance}  # 새로운 얼굴 데이터 추가

            f.seek(0)  # 파일의 처음으로 이동
            json.dump(face_data, f)  # 업데이트된 데이터 덤프
            f.truncate()  # 파일 크기를 현재 위치로 자르기

    except FileNotFoundError:  # json 파일이 없을 때
        with open(json_file, 'w') as f:  # 파일을 새로 생성
            face_data = {user_id: {"eye_distance": eye_distance}}  # 새 데이터 작성
            json.dump(face_data, f)  # 데이터를 파일에 덤프

    return jsonify({"message": "아이디가 성공적으로 등록되었습니다.", "eye_distance": eye_distance}), 200  # 성공 메시지 반환

# 로그인 처리 라우트
@app.route('/login', methods=['POST'])
def login():
    cap = cv2.VideoCapture(0)  # 웹캠 열기
    frame_count = 0  # 프레임 카운터 초기화
    required_frames = 30  # 필요한 프레임 수

    try:
        with open(json_file, 'r') as f:  # json 파일 열기
            face_data = json.load(f)  # 얼굴 데이터 불러오기
    except (FileNotFoundError, json.JSONDecodeError):  # 파일이 없거나 JSON 형식이 잘못된 경우
        return jsonify({"error": "등록된 얼굴 정보가 없습니다."})  # 에러 메시지 반환

    while cap.isOpened():
        success, frame = cap.read()  # 프레임 읽기
        if not success:
            return jsonify({"error": "Failed to capture video."})  # 실패 시 에러 반환

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임을 RGB로 변환
        results = face_mesh.process(rgb_frame)  # 얼굴 인식 처리

        if results.multi_face_landmarks:  # 얼굴 랜드마크가 있으면
            frame_count += 1  # 프레임 카운트 증가
            if frame_count >= required_frames:  # 필요한 프레임 수에 도달했을 때
                for face_landmarks in results.multi_face_landmarks:  # 각 얼굴에 대해
                    left_eye = face_landmarks.landmark[33]  # 왼쪽 눈 좌표 추출
                    right_eye = face_landmarks.landmark[263]  # 오른쪽 눈 좌표 추출

                    left_eye_coords = (left_eye.x, left_eye.y, left_eye.z)  # 왼쪽 눈 좌표를 튜플로 저장
                    right_eye_coords = (right_eye.x, right_eye.y, right_eye.z)  # 오른쪽 눈 좌표를 튜플로 저장

                    eye_distance = calculate_3d_distance(left_eye_coords, right_eye_coords)  # 두 눈 사이 거리 계산

                    for user_id, user_data in face_data.items():  # 등록된 얼굴 데이터와 비교
                        stored_eye_distance = user_data.get('eye_distance')  # 저장된 눈 사이 거리
                        if abs(stored_eye_distance - eye_distance) < 0.05:  # 허용 범위 내에서 일치하는지 확인
                            cap.release()  # 웹캠 릴리즈
                            return redirect(url_for('yoga'))  # 로그인 성공 시 요가 페이지로 이동
                        
                    # 얼굴 정보가 일치하지 않으면 에러 반환
                    return jsonify({"error": "얼굴 정보가 일치하지 않습니다."})
        else:
            frame_count = 0  # 얼굴이 인식되지 않으면 카운트 초기화

        if cv2.waitKey(5) & 0xFF == ord('q'):  # 'q'를 누르면 루프 종료
            break

    cap.release()  # 웹캠 릴리즈
    cv2.destroyAllWindows()  # 모든 윈도우 닫기
    return jsonify({"error": "Failed to login."})  # 로그인 실패 시 에러 반환

# 요가 페이지 렌더링
@app.route('/yoga')
def yoga():
    return render_template('yoga.html')  # 요가 페이지 렌더링

# 게임 페이지 렌더링
@app.route('/game')
def game():
    pose_name = os.path.basename(standard_pose_image_path).split('.')[0]  # 요가 자세 이름 추출
    return render_template('game.html', pose_name=pose_name)  # 게임 페이지 렌더링

# 요가 비디오 피드 라우트
@app.route('/video_feed_yoga')
def video_feed_yoga():
    return Response(gen_yoga_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  # 비디오 피드 생성

# 요가 프레임 생성 함수
def gen_yoga_frames():
    cap = cv2.VideoCapture(0)  # 웹캠 열기

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)  # 자세 인식 모델 생성
    
    joint_list = [
        (11, 13, 15),  # 왼쪽 어깨, 팔꿈치, 손목
        (12, 14, 16),  # 오른쪽 어깨, 팔꿈치, 손목
        (23, 25, 27),  # 왼쪽 골반, 무릎, 발목
        (24, 26, 28)   # 오른쪽 골반, 무릎, 발목
    ]
    holding_time = 3  # 자세를 유지할 시간 (초)
    correct_start_time = None  # 정확한 자세를 시작한 시간

    while cap.isOpened():
        success, frame = cap.read()  # 프레임 읽기
        if not success:
            break  # 실패 시 종료

        frame = cv2.flip(frame, 1)  # 프레임을 좌우 반전
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임을 RGB로 변환
        results = pose.process(frame_rgb)  # 자세 인식 처리

        if results.pose_landmarks:  # 자세 랜드마크가 있으면
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)  # 랜드마크 그리기
            
            all_angles_correct = True  # 모든 각도가 정확한지 확인

            for joints in joint_list:  # 각 관절 조합에 대해
                std_angles = calculate_angle(
                    [standard_pose_landmarks.landmark[joints[0]].x, standard_pose_landmarks.landmark[joints[0]].y],
                    [standard_pose_landmarks.landmark[joints[1]].x, standard_pose_landmarks.landmark[joints[1]].y],
                    [standard_pose_landmarks.landmark[joints[2]].x, standard_pose_landmarks.landmark[joints[2]].y]
                )  # 기준 자세 각도 계산

                user_angles = calculate_angle(
                    [results.pose_landmarks.landmark[joints[0]].x, results.pose_landmarks.landmark[joints[0]].y],
                    [results.pose_landmarks.landmark[joints[1]].x, results.pose_landmarks.landmark[joints[1]].y],
                    [results.pose_landmarks.landmark[joints[2]].x, results.pose_landmarks.landmark[joints[2]].y]
                )  # 사용자 자세 각도 계산

                if abs(std_angles - user_angles) > 15:  # 기준 각도와 사용자 각도의 차이가 15도 이상이면
                    color = (0, 0, 255)  # 빨간색으로 표시
                    all_angles_correct = False  # 각도가 정확하지 않음
                else:
                    color = (0, 255, 0)  # 초록색으로 표시

                # 각 관절에 원을 그려서 표시
                cv2.circle(frame, (int(results.pose_landmarks.landmark[joints[0]].x * frame.shape[1]),
                                   int(results.pose_landmarks.landmark[joints[0]].y * frame.shape[0])), 10, color, -1)
                cv2.circle(frame, (int(results.pose_landmarks.landmark[joints[1]].x * frame.shape[1]),
                                   int(results.pose_landmarks.landmark[joints[1]].y * frame.shape[0])), 10, color, -1)
                cv2.circle(frame, (int(results.pose_landmarks.landmark[joints[2]].x * frame.shape[1]),
                                   int(results.pose_landmarks.landmark[joints[2]].y * frame.shape[0])), 10, color, -1)

            if all_angles_correct:  # 모든 각도가 정확하면
                if correct_start_time is None:  # 처음으로 자세가 정확한 순간을 기록
                    correct_start_time = time.time()
                elapsed_time = time.time() - correct_start_time  # 경과 시간 계산

                # 경과 시간을 화면에 표시
                cv2.putText(frame, f'Time: {elapsed_time:.2f} sec', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if elapsed_time >= holding_time:  # 설정된 시간 동안 자세가 유지되면
                    cv2.putText(frame, 'Pose Correct', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cap.release()  # 웹캠 릴리즈
                    return redirect(url_for('game'))  # 게임 페이지로 이동
            else:
                correct_start_time = None  # 자세가 틀리면 시간 초기화

        combined_image = np.hstack((standard_pose_image, frame))  # 기준 자세와 사용자의 프레임을 나란히 합침

        # 이미지를 바이너리로 변환하여 브라우저로 전송
        ret, buffer = cv2.imencode('.jpg', combined_image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'  # 프레임 구분자
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 프레임 데이터 전송

    cap.release()  # 웹캠 릴리즈

# 애플리케이션 실행
if __name__ == "__main__":
    app.run(debug=True)  # 디버그 모드로 실행
