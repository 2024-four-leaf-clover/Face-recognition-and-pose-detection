from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time
import os

app = Flask(__name__)

# MediaPipe Pose 모델과 기본 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 표준 요가 자세 이미지 (미리 로드)
standard_pose_image_path = "C:/Capstone2/static/yoga_posture/dataset/vriksasana/1-0.png"
standard_pose_image = cv2.imread(standard_pose_image_path)
standard_pose_image = cv2.resize(standard_pose_image, (640, 480))
standard_pose_image_rgb = cv2.cvtColor(standard_pose_image, cv2.COLOR_BGR2RGB)

# 파일명 추출
pose_name = os.path.basename(standard_pose_image_path).split('.')[0]  # '1-0.png'에서 '1-0'만 추출

# 표준 요가 자세 이미지에서 관절을 검출
with mp_pose.Pose(static_image_mode=True) as pose:
    standard_results = pose.process(standard_pose_image_rgb)
standard_pose_landmarks = standard_results.pose_landmarks

# 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# 웹캠 피드 생성
def gen_frames():
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
        
        # 좌우 반전 및 RGB로 변환
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 사용자 영상에서 관절 검출
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

                cv2.putText(frame, f'Time: {elapsed_time:.2f} sec', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if elapsed_time >= holding_time:
                    cv2.putText(frame, 'Pose Correct', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    break
            else:
                correct_start_time = None

        # 표준 요가 자세 이미지와 사용자 이미지를 결합
        combined_image = np.hstack((standard_pose_image, frame))

        # 이미지를 바이너리로 변환하여 브라우저로 전송
        ret, buffer = cv2.imencode('.jpg', combined_image)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# yoga.html 페이지
@app.route('/')
def yoga():
    return render_template('yoga.html')

# game.html 페이지
@app.route('/game')
def game():
    pose_name = os.path.basename(standard_pose_image_path).split('.')[0]
    return render_template('game.html', pose_name=pose_name)

# 비디오 스트리밍 경로
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
