from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
import mediapipe as mp
import numpy as np
import math
import threading
import time

app = Flask(__name__)

# 전역 변수: 웹캠 활성화
video_capture = cv2.VideoCapture(0)

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 전역 변수
hand_detected = False

# 요가 자세 기준 이미지 설정
standard_pose_image_path = "C:/Capstone2/static/yoga_posture/dataset/agnistambhasana/10-0.png"
standard_pose_image = cv2.imread(standard_pose_image_path)
standard_pose_image = cv2.resize(standard_pose_image, (640, 480))
standard_pose_image_rgb = cv2.cvtColor(standard_pose_image, cv2.COLOR_BGR2RGB)

# 자세 모델 초기화
with mp_pose.Pose(static_image_mode=True) as pose:
    standard_results = pose.process(standard_pose_image_rgb)
standard_pose_landmarks = standard_results.pose_landmarks

def detect_hand(frame):
    global hand_detected
    print("Detect hand function called")  # 디버깅용 출력
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            print("Hand landmarks detected")
            for hand_landmarks in results.multi_hand_landmarks:
                if check_all_fingers_straight(hand_landmarks.landmark):
                    hand_detected = True
                    print("Hand detected! Flag set to True")
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            print("No hand detected in this frame")
    return frame

# 모든 손가락이 펴져 있는지 확인
def check_all_fingers_straight(landmarks):
    for tip_id in [8, 12, 16, 20]:  # 검지, 중지, 약지, 소지 끝
        base_id = tip_id - 2
        if landmarks[tip_id].y > landmarks[base_id].y:  # 접혀 있다면 False
            return False
    return True

# 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

# 요가 자세 프레임 생성 (손바닥 감지 포함)
def gen_yoga_frames():
    global hand_detected, video_capture
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    joint_list = [(11, 13, 15), (12, 14, 16), (23, 25, 27), (24, 26, 28)]  # 요가 관절 목록
    holding_time = 3  # 요가 자세 유지 시간
    correct_start_time = None  # 올바른 자세 시작 시간

    while True:
        ret, frame = video_capture.read()  # 전역 웹캠에서 프레임 읽기
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
        pose_results = pose.process(frame_rgb)  # 요가 자세 인식

        print("gen_yoga_frames loop: hand_detected =", hand_detected)  # 디버깅용 출력

        # 손바닥 감지 처리
        frame = detect_hand(frame)

        # 요가 자세 인식 처리
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)  # 랜드마크 그리기
            all_angles_correct = True  # 모든 관절 각도가 정확한지 여부

            for joints in joint_list:
                std_angles = calculate_angle(
                    [standard_pose_landmarks.landmark[joints[0]].x, standard_pose_landmarks.landmark[joints[0]].y],
                    [standard_pose_landmarks.landmark[joints[1]].x, standard_pose_landmarks.landmark[joints[1]].y],
                    [standard_pose_landmarks.landmark[joints[2]].x, standard_pose_landmarks.landmark[joints[2]].y]
                )
                user_angles = calculate_angle(
                    [pose_results.pose_landmarks.landmark[joints[0]].x, pose_results.pose_landmarks.landmark[joints[0]].y],
                    [pose_results.pose_landmarks.landmark[joints[1]].x, pose_results.pose_landmarks.landmark[joints[1]].y],
                    [pose_results.pose_landmarks.landmark[joints[2]].x, pose_results.pose_landmarks.landmark[joints[2]].y]
                )
                if abs(std_angles - user_angles) > 15:  # 기준 각도와 차이가 15도 이상이면
                    color = (0, 0, 255)  # 빨간색으로 표시
                    all_angles_correct = False  # 자세가 정확하지 않음
                else:
                    color = (0, 255, 0)  # 초록색으로 표시

                # 관절 좌표에 원을 그려서 표시
                cv2.circle(frame, (int(pose_results.pose_landmarks.landmark[joints[0]].x * frame.shape[1]),
                                   int(pose_results.pose_landmarks.landmark[joints[0]].y * frame.shape[0])), 10, color, -1)
                cv2.circle(frame, (int(pose_results.pose_landmarks.landmark[joints[1]].x * frame.shape[1]),
                                   int(pose_results.pose_landmarks.landmark[joints[1]].y * frame.shape[0])), 10, color, -1)
                cv2.circle(frame, (int(pose_results.pose_landmarks.landmark[joints[2]].x * frame.shape[1]),
                                   int(pose_results.pose_landmarks.landmark[joints[2]].y * frame.shape[0])), 10, color, -1)

        # 기준 자세와 현재 프레임 합치기
        combined_image = np.hstack((standard_pose_image, frame))
        _, buffer = cv2.imencode('.jpg', combined_image)
        frame = buffer.tobytes()

        # HTTP 스트림으로 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 라우트
@app.route('/')
def index():
    return render_template('yoga.html')

@app.route('/game')
def game():
    global hand_detected
    hand_detected = False  # 손바닥 감지 상태 초기화
    return render_template('game.html')

@app.route('/video_feed_yoga')
def video_feed_yoga():
    """웹캠 피드 스트림"""
    return Response(gen_yoga_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check-hand')
def check_hand():
    global hand_detected
    print("check_hand called, hand_detected =", hand_detected)  # 디버깅 메시지 추가
    if hand_detected:
        print("detected!")
        return 'hand_detected', 200  # 손바닥 감지 상태 반환
    return '', 204  # 감지되지 않으면 상태 없음

if __name__ == '__main__':
    app.run(debug=True)
