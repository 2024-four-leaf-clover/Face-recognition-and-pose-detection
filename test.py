# Flask와 필요한 모듈들 가져오기
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session  # Flask 관련 모듈들 가져오기
import cv2  # OpenCV를 사용한 영상 처리
import mediapipe as mp  # MediaPipe 라이브러리로 영상 처리
import math  # 수학 연산을 위한 모듈
import json  # JSON 파일을 다루기 위한 모듈
import numpy as np  # 배열 처리와 수학 연산을 위한 NumPy
import time  # 시간 측정을 위한 모듈
import os  # 운영체제 관련 기능을 위한 모듈

# Flask 앱 초기화
app = Flask(__name__)  # Flask 애플리케이션 생성
app.secret_key = "your_secret_key"

# MediaPipe 초기화
mp_hands = mp.solutions.hands  # 손 인식 모델 초기화
mp_face_mesh = mp.solutions.face_mesh  # 얼굴 인식 모델 초기화
mp_pose = mp.solutions.pose  # 자세 인식 모델 초기화
mp_drawing = mp.solutions.drawing_utils  # MediaPipe의 그리기 유틸리티

# 손 인식 모델 객체 생성
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)  # 손 인식 설정 (신뢰도 0.7, 추적 신뢰도 0.5)

# 얼굴 메시 모델 생성
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # 정적 이미지 모드 활성화
    max_num_faces=1,         # 한 번에 최대 인식할 얼굴 수는 1
    refine_landmarks=True,   # 세밀한 랜드마크 활성화
    min_detection_confidence=0.5  # 얼굴 감지 최소 신뢰도 설정
)

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)  # 자세 인식 모델 생성 (실시간 모드, 신뢰도 0.5)

# 얼굴 정보를 저장할 json 파일 경로 설정
json_file = 'eyes.json'  # 얼굴 정보가 저장될 JSON 파일 경로 설정

# 요가 자세 이미지 경로와 이미지 로드 및 전처리
standard_pose_image_path = "C:/Capstone2/static/yoga_posture/dataset/agnistambhasana/10-0.png"  # 요가 자세 이미지 경로

standard_pose_image = cv2.imread(standard_pose_image_path)  # 이미지를 읽어옴
standard_pose_image = cv2.resize(standard_pose_image, (640, 480))  # 이미지를 640x480으로 리사이징
standard_pose_image_rgb = cv2.cvtColor(standard_pose_image, cv2.COLOR_BGR2RGB)  # 이미지를 BGR에서 RGB로 변환

# 자세 인식 결과 처리
with mp_pose.Pose(static_image_mode=True) as pose:  # 자세 인식 모델을 초기화하여 사용
    standard_results = pose.process(standard_pose_image_rgb)  # 이미지에서 자세 인식을 수행
standard_pose_landmarks = standard_results.pose_landmarks  # 자세 랜드마크를 저장

# 손가락 상태 판별 함수 정의
def is_finger_up(landmarks, finger_tip, finger_pip):  # 손가락 끝과 PIP(두 번째 마디) 좌표를 비교하여 손가락이 올라갔는지 확인
    return landmarks[finger_tip].y < landmarks[finger_pip].y  # 손가락 끝이 PIP보다 위에 있으면 True 반환

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

# 얼굴 정보를 불러오는 함수
def load_face_data(user_id):  # 특정 user_id에 해당하는 얼굴 정보를 로드
    try:
        with open(json_file, 'r') as f:  # json 파일을 읽기 모드로 엶
            face_data = json.load(f)  # JSON 데이터를 불러옴
            return face_data.get(user_id, {})  # user_id에 해당하는 데이터를 반환, 없으면 빈 딕셔너리 반환
    except (FileNotFoundError, json.JSONDecodeError):  # 파일이 없거나 형식이 잘못된 경우
        return None  # 오류 시 None 반환

# 메인 페이지 렌더링
@app.route('/')  # '/' 경로로 요청이 오면 index 함수 실행
def index():
    return render_template('main.html')  # main.html을 렌더링하여 반환

# 비디오 피드를 위한 라우트
@app.route('/video_feed')  # '/video_feed' 경로로 요청이 오면 video_feed 함수 실행
def video_feed():
    cap = cv2.VideoCapture(0)  # 웹캠을 열기
    detection_status = {"detected": False, "action": ""}  # 손 동작 감지 상태를 저장할 딕셔너리 초기화
    
    while True:  # 무한 루프 시작
        success, frame = cap.read()  # 프레임 읽기
        if not success:  # 프레임 읽기에 실패하면 루프 종료
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임을 BGR에서 RGB로 변환
        image.flags.writeable = False  # 이미지 쓰기 금지 (성능 향상)
        results = hands.process(image)  # 손 인식 처리

        if results.multi_hand_landmarks:  # 여러 손 랜드마크가 감지된 경우
            for hand_landmarks in results.multi_hand_landmarks:  # 각 손에 대해 반복
                landmarks = hand_landmarks.landmark  # 손 랜드마크 추출

                # 손가락 끝과 PIP(두 번째 마디)의 인덱스 정의
                INDEX_FINGER_TIP = 8  # 검지 끝 좌표
                INDEX_FINGER_PIP = 6  # 검지 두 번째 마디 좌표
                MIDDLE_FINGER_TIP = 12  # 중지 끝 좌표
                MIDDLE_FINGER_PIP = 10  # 중지 두 번째 마디 좌표
                RING_FINGER_TIP = 16  # 약지 끝 좌표
                RING_FINGER_PIP = 14  # 약지 두 번째 마디 좌표
                PINKY_FINGER_TIP = 20  # 새끼손가락 끝 좌표
                PINKY_FINGER_PIP = 18  # 새끼손가락 두 번째 마디 좌표

                # 각 손가락의 상태 판별
                index_finger_up = is_finger_up(landmarks, INDEX_FINGER_TIP, INDEX_FINGER_PIP)  # 검지가 올라갔는지 확인
                middle_finger_up = is_finger_up(landmarks, MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP)  # 중지가 올라갔는지 확인
                ring_finger_down = not is_finger_up(landmarks, RING_FINGER_TIP, RING_FINGER_PIP)  # 약지가 내려갔는지 확인
                pinky_finger_down = not is_finger_up(landmarks, PINKY_FINGER_TIP, PINKY_FINGER_PIP)  # 새끼손가락이 내려갔는지 확인

                # 검지만 위로 올라간 경우 (등록 동작)
                if index_finger_up and not middle_finger_up and ring_finger_down and pinky_finger_down:  # 검지만 올라간 경우
                    detection_status["detected"] = True  # 감지 상태를 True로 설정
                    detection_status["action"] = "register"  # 동작을 'register'로 설정
                    cap.release()  # 웹캠 릴리즈
                    cv2.destroyAllWindows()  # 모든 창 닫기
                    return jsonify(detection_status)  # 감지 상태를 JSON으로 반환

                # 검지와 중지가 위로 올라간 경우 (로그인 동작)
                if index_finger_up and middle_finger_up and ring_finger_down and pinky_finger_down:  # 검지와 중지가 올라간 경우
                    detection_status["detected"] = True  # 감지 상태를 True로 설정
                    detection_status["action"] = "login"  # 동작을 'login'으로 설정
                    cap.release()  # 웹캠 릴리즈
                    cv2.destroyAllWindows()  # 모든 창 닫기
                    return jsonify(detection_status)  # 감지 상태를 JSON으로 반환

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 루프 종료
            break

    cap.release()  # 웹캠 릴리즈
    cv2.destroyAllWindows()  # 모든 창 닫기
    return jsonify(detection_status)  # 감지 상태 반환

# 사용자 등록 라우트
@app.route('/register', methods=['POST'])  # '/register' 경로로 POST 요청이 오면 register 함수 실행
def register():
    data = request.get_json()  # 클라이언트로부터 받은 JSON 데이터 파싱
    user_id = data.get('user_id')  # JSON에서 user_id 가져오기

    if not user_id:  # user_id가 없으면
        return jsonify({"error": "아이디를 입력해야 합니다."}), 400  # 에러 메시지 반환

    cap = cv2.VideoCapture(0)  # 웹캠 열기
    frame_count = 0  # 프레임 카운터 초기화
    required_frames = 30  # 필요한 프레임 수 설정

    # 얼굴 데이터를 저장할 리스트 초기화
    eye_distances = []
    nose_chin_distances = []
    mouth_widths = []
    forehead_chin_distances = []
    cheek_distances = []
    nose_bridge_lengths = []

    while cap.isOpened():  # 웹캠이 열려 있을 동안 루프
        success, frame = cap.read()  # 프레임 읽기
        if not success:  # 프레임 읽기에 실패하면 루프 종료
            return jsonify({"error": "카메라 영상을 캡처하는 데 실패했습니다."})  # 실패 시 에러 반환

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임을 RGB로 변환
        results = face_mesh.process(rgb_frame)  # 얼굴 인식 처리

        if results.multi_face_landmarks:  # 얼굴 랜드마크가 감지되면
            frame_count += 1  # 프레임 카운트 증가
            if frame_count >= required_frames:  # 필요한 프레임 수에 도달했을 때
                for face_landmarks in results.multi_face_landmarks:  # 감지된 각 얼굴에 대해 처리
                    left_eye = face_landmarks.landmark[33]  # 왼쪽 눈 좌표 추출
                    right_eye = face_landmarks.landmark[263]  # 오른쪽 눈 좌표 추출
                    nose_tip = face_landmarks.landmark[1]  # 코 끝 좌표 추출
                    chin = face_landmarks.landmark[152]  # 턱 좌표 추출
                    left_mouth = face_landmarks.landmark[61]  # 왼쪽 입꼬리 좌표 추출
                    right_mouth = face_landmarks.landmark[291]  # 오른쪽 입꼬리 좌표 추출
                    forehead = face_landmarks.landmark[10]  # 이마 좌표 추출
                    left_cheek = face_landmarks.landmark[234]  # 왼쪽 광대
                    right_cheek = face_landmarks.landmark[454]  # 오른쪽 광대
                    nose_bridge = face_landmarks.landmark[6]  # 콧대 중간 지점

                    # 여러 프레임에서 수집한 데이터 리스트에 저장
                    eye_distances.append(calculate_3d_distance(
                        (left_eye.x, left_eye.y, left_eye.z),
                        (right_eye.x, right_eye.y, right_eye.z)
                    ))
                    
                    nose_chin_distances.append(calculate_3d_distance(
                        (nose_tip.x, nose_tip.y, nose_tip.z),
                        (chin.x, chin.y, chin.z)
                    ))

                    mouth_widths.append(calculate_3d_distance(
                        (left_mouth.x, left_mouth.y, left_mouth.z),
                        (right_mouth.x, right_mouth.y, right_mouth.z)
                    ))

                    forehead_chin_distances.append(calculate_3d_distance(
                        (forehead.x, forehead.y, forehead.z),
                        (chin.x, chin.y, chin.z)
                    ))

                    cheek_distances.append(calculate_3d_distance(
                        (left_cheek.x, left_cheek.y, left_cheek.z),
                        (right_cheek.x, right_cheek.y, right_cheek.z)
                    ))

                    nose_bridge_lengths.append(calculate_3d_distance(
                        (nose_tip.x, nose_tip.y, nose_tip.z),
                        (nose_bridge.x, nose_bridge.y, nose_bridge.z)
                    ))

            if frame_count >= required_frames:  # 필요한 프레임 수에 도달하면 루프 종료
                break

    cap.release()  # 웹캠 릴리즈

    if not eye_distances:  # 얼굴 특징이 없으면
        return jsonify({"error": "얼굴 인식을 실패했습니다. 다시 시도해주세요."}), 400  # 에러 메시지 반환

    # 여러 프레임에서 수집한 데이터의 평균값 계산
    face_features = {  
        "eye_distance": np.mean(eye_distances),
        "nose_chin_distance": np.mean(nose_chin_distances),
        "mouth_width": np.mean(mouth_widths),
        "forehead_chin_distance": np.mean(forehead_chin_distances),
        "cheek_distance": np.mean(cheek_distances),
        "nose_bridge_length": np.mean(nose_bridge_lengths)
    }

    try:
        with open(json_file, 'r+') as f:  # json 파일을 읽고 쓸 수 있는 모드로 엶
            try:
                face_data = json.load(f)  # 기존의 얼굴 데이터를 불러옴
            except json.JSONDecodeError:  # 파일이 비어 있거나 형식이 잘못된 경우
                face_data = {}  # 빈 딕셔너리로 초기화

            if user_id in face_data:  # 이미 등록된 아이디가 있으면
                return jsonify({"error": "이미 등록된 아이디입니다."}), 400  # 에러 메시지 반환

            face_data[user_id] = face_features  # 새로운 얼굴 데이터를 추가

            f.seek(0)  # 파일의 처음으로 이동
            json.dump(face_data, f, indent=4)  # 업데이트된 데이터를 JSON으로 저장 (indent=4로 보기 좋게 저장)
            f.truncate()  # 파일 크기를 현재 위치로 자르기

    except FileNotFoundError:  # json 파일이 없으면
        with open(json_file, 'w') as f:  # 파일을 새로 생성
            face_data = {user_id: face_features}  # 새 데이터 작성
            json.dump(face_data, f, indent=4)  # 데이터를 예쁘게 출력하도록 설정

    return jsonify({"message": "아이디가 성공적으로 등록되었습니다.", "face_features": face_features}), 200  # 성공 메시지 반환

# 로그인 처리 라우트
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
                    nose_tip = face_landmarks.landmark[1]
                    chin = face_landmarks.landmark[152]
                    left_mouth = face_landmarks.landmark[61]
                    right_mouth = face_landmarks.landmark[291]
                    forehead = face_landmarks.landmark[10]
                    left_cheek = face_landmarks.landmark[234]
                    right_cheek = face_landmarks.landmark[454]
                    nose_bridge = face_landmarks.landmark[6]

                    eye_distance = calculate_3d_distance(
                        (left_eye.x, left_eye.y, left_eye.z),
                        (right_eye.x, right_eye.y, right_eye.z)
                    )
                    nose_chin_distance = calculate_3d_distance(
                        (nose_tip.x, nose_tip.y, nose_tip.z),
                        (chin.x, chin.y, chin.z)
                    )
                    mouth_width = calculate_3d_distance(
                        (left_mouth.x, left_mouth.y, left_mouth.z),
                        (right_mouth.x, right_mouth.y, right_mouth.z)
                    )
                    forehead_chin_distance = calculate_3d_distance(
                        (forehead.x, forehead.y, forehead.z),
                        (chin.x, chin.y, chin.z)
                    )
                    cheek_distance = calculate_3d_distance(
                        (left_cheek.x, left_cheek.y, left_cheek.z),
                        (right_cheek.x, right_cheek.y, right_cheek.z)
                    )
                    nose_bridge_length = calculate_3d_distance(
                        (nose_tip.x, nose_tip.y, nose_tip.z),
                        (nose_bridge.x, nose_bridge.y, nose_bridge.z)
                    )

                    for user_id, user_data in face_data.items():
                        if (
                            abs(user_data.get('eye_distance') - eye_distance) < 0.03 and
                            abs(user_data.get('nose_chin_distance') - nose_chin_distance) < 0.03 and
                            abs(user_data.get('mouth_width') - mouth_width) < 0.03 and
                            abs(user_data.get('forehead_chin_distance') - forehead_chin_distance) < 0.03 and
                            abs(user_data.get('cheek_distance') - cheek_distance) < 0.03 and
                            abs(user_data.get('nose_bridge_length') - nose_bridge_length) < 0.03  # 쉼표 삭제
                        ):
                            matching_user_id = user_id
                            break

                if matching_user_id:
                    cap.release()
                    session['user_id'] = matching_user_id  # 로그인 성공 시 세션에 user_id 저장
                    return redirect(url_for('yoga'))  # yoga 페이지로 리다이렉트

                return jsonify({"error": "얼굴 정보가 일치하지 않습니다."})

        else:
            frame_count = 0

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"error": "Failed to login."})


# 요가 페이지 렌더링
@app.route('/yoga')
def yoga():
    user_id = session.get('user_id', 'Unknown')  # 세션에서 user_id 가져오기
    return render_template('yoga.html', user_id=user_id)

# 추가한 요소
@app.route('/get_user_id')
def get_user_id():
    return jsonify({"user_id": session.get('user_id', 'Unknown')})

# 게임 페이지 렌더링
@app.route('/game')
def game():
    user_id = session.get('user_id', 'Unknown')  # 세션에서 user_id 가져오기
    pose_name = os.path.basename(standard_pose_image_path).split('.')[0]
    return render_template('game.html', pose_name=pose_name, user_id=user_id)

# 요가 비디오 피드 라우트
@app.route('/video_feed_yoga')  # '/video_feed_yoga' 경로로 요청이 오면 video_feed_yoga 함수 실행
def video_feed_yoga():
    return Response(gen_yoga_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  # 비디오 피드 생성

# 필요한 손가락 검사 함수 추가
def check_all_fingers_straight(landmarks):
    for tip_id in [8, 12, 16, 20]:  # 각 손가락 끝과 해당 마디 위치 비교
        tip_y = landmarks[tip_id].y
        base_y = landmarks[tip_id - 2].y
        if tip_y > base_y:
            return False
    return True

def check_thumb_position(landmarks, hand_label):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    if hand_label == "Right" and thumb_tip.x < index_tip.x:
        return True
    elif hand_label == "Left" and thumb_tip.x > index_tip.x:
        return True
    return False

# 요가 프레임 생성 함수
def gen_yoga_frames():
    cap = cv2.VideoCapture(0)  # 웹캠 열기

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)  # 자세 인식 모델 생성
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)  # 손 인식 모델 생성

    joint_list = [  # 관절 좌표 리스트
        (11, 13, 15),  # 왼쪽 어깨, 팔꿈치, 손목
        (12, 14, 16),  # 오른쪽 어깨, 팔꿈치, 손목
        (23, 25, 27),  # 왼쪽 골반, 무릎, 발목
        (24, 26, 28)   # 오른쪽 골반, 무릎, 발목
    ]
    holding_time = 3  # 자세를 유지할 시간 (초 단위)
    correct_start_time = None  # 정확한 자세를 시작한 시간 기록

    while cap.isOpened():  # 웹캠이 열려 있을 동안 루프
        success, frame = cap.read()  # 프레임 읽기
        if not success:  # 프레임 읽기에 실패하면 루프 종료
            break

        frame = cv2.flip(frame, 1)  # 프레임을 좌우 반전
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임을 RGB로 변환
        results_pose = pose.process(frame_rgb)  # 자세 인식 처리
        results_hands = hands.process(frame_rgb)  # 손 인식 처리

        # 자세 인식 결과가 있을 때
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)  # 랜드마크를 그리기
            all_angles_correct = True  # 모든 각도가 정확한지 여부를 저장할 변수

            for joints in joint_list:  # 각 관절 좌표 조합에 대해 반복
                std_angles = calculate_angle(  # 기준 자세의 각도 계산
                    [standard_pose_landmarks.landmark[joints[0]].x, standard_pose_landmarks.landmark[joints[0]].y],
                    [standard_pose_landmarks.landmark[joints[1]].x, standard_pose_landmarks.landmark[joints[1]].y],
                    [standard_pose_landmarks.landmark[joints[2]].x, standard_pose_landmarks.landmark[joints[2]].y]
                )

                user_angles = calculate_angle(  # 사용자의 현재 자세 각도 계산
                    [results_pose.pose_landmarks.landmark[joints[0]].x, results_pose.pose_landmarks.landmark[joints[0]].y],
                    [results_pose.pose_landmarks.landmark[joints[1]].x, results_pose.pose_landmarks.landmark[joints[1]].y],
                    [results_pose.pose_landmarks.landmark[joints[2]].x, results_pose.pose_landmarks.landmark[joints[2]].y]
                )

                if abs(std_angles - user_angles) > 15:  # 기준 각도와 차이가 15도 이상이면
                    color = (0, 0, 255)  # 빨간색으로 표시
                    all_angles_correct = False  # 자세가 정확하지 않음
                else:
                    color = (0, 255, 0)  # 초록색으로 표시

                # 관절 좌표에 원을 그려서 표시
                cv2.circle(frame, (int(results_pose.pose_landmarks.landmark[joints[0]].x * frame.shape[1]),
                                   int(results_pose.pose_landmarks.landmark[joints[0]].y * frame.shape[0])), 10, color, -1)
                cv2.circle(frame, (int(results_pose.pose_landmarks.landmark[joints[1]].x * frame.shape[1]),
                                   int(results_pose.pose_landmarks.landmark[joints[1]].y * frame.shape[0])), 10, color, -1)
                cv2.circle(frame, (int(results_pose.pose_landmarks.landmark[joints[2]].x * frame.shape[1]),
                                   int(results_pose.pose_landmarks.landmark[joints[2]].y * frame.shape[0])), 10, color, -1)

            if all_angles_correct:  # 모든 각도가 정확하면
                if correct_start_time is None:  # 처음으로 자세가 정확한 순간을 기록
                    correct_start_time = time.time()
                elapsed_time = time.time() - correct_start_time  # 정확한 자세를 유지한 시간 계산

                # 경과 시간을 화면에 표시
                cv2.putText(frame, f'Time: {elapsed_time:.2f} sec', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if elapsed_time >= holding_time:  # 설정된 시간 동안 자세가 유지되면
                    cv2.putText(frame, 'Pose Correct', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cap.release()  # 웹캠 릴리즈
                    return redirect(url_for('game'))  # 게임 페이지로 이동
            else:
                correct_start_time = None  # 자세가 틀렸을 때 시간 초기화

        # 손 인식 결과가 있을 때
        if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                landmarks = hand_landmarks.landmark
                hand_label = handedness.classification[0].label  # "Left" 또는 "Right"

                # 손가락과 엄지 상태 검사
                if check_all_fingers_straight(landmarks) and check_thumb_position(landmarks, hand_label):
                    cap.release()  # 웹캠 릴리즈
                    cv2.destroyAllWindows()  # 모든 창 닫기
                    # 조건 충족 시 yoga.html로 이동
                    return redirect(url_for('yoga'))

                # 손 랜드마크 그리기
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        combined_image = np.hstack((standard_pose_image, frame))  # 기준 자세와 사용자의 프레임을 나란히 합침

        # 이미지를 바이너리로 변환하여 브라우저로 전송
        ret, buffer = cv2.imencode('.jpg', combined_image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'  # 프레임 구분자
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 프레임 데이터 전송

    cap.release()  # 웹캠 릴리즈


# 애플리케이션 실행
if __name__ == "__main__":  # 이 스크립트가 메인으로 실행될 때만
    app.run(debug=True)  # Flask 애플리케이션 실행 (디버그 모드)