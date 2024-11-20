from flask import Flask, render_template, Response, redirect, url_for
import cv2
import mediapipe as mp
import threading

app = Flask(__name__)

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 전역 변수
hand_detected = False
video_capture = cv2.VideoCapture(0)  # 웹캠 열기

def detect_hand(frame):
    """손바닥 감지를 수행하는 함수"""
    global hand_detected
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        # BGR -> RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # 손바닥 감지 로직
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if check_all_fingers_straight(hand_landmarks.landmark):
                    hand_detected = True
                # 손 Landmark를 그리기
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame

def check_all_fingers_straight(landmarks):
    """손가락이 모두 펴져 있는지 확인"""
    for tip_id in [8, 12, 16, 20]:  # 검지, 중지, 약지, 소지 끝
        base_id = tip_id - 2  # 두 번째 관절
        if landmarks[tip_id].y > landmarks[base_id].y:  # 접혀 있다면 False
            return False
    return True

def generate_frames():
    """웹캠 프레임 생성"""
    global video_capture
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        # 손바닥 감지 및 프레임 처리
        frame = detect_hand(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        # HTTP 스트림으로 전달
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """A 페이지"""
    return render_template('a.html')

@app.route('/go-to-b', methods=['POST'])
def go_to_b():
    """A 페이지에서 B 페이지로 이동"""
    return redirect(url_for('page_b'))

@app.route('/page-b')
def page_b():
    """B 페이지"""
    global hand_detected
    hand_detected = False  # 손바닥 감지 상태 초기화
    return render_template('b.html')

@app.route('/video_feed')
def video_feed():
    """웹캠 피드 스트림"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check-hand')
def check_hand():
    """손바닥 감지 상태 확인"""
    global hand_detected
    if hand_detected:
        return redirect(url_for('index'))  # 감지되면 A 페이지로 이동
    return '', 204  # 감지되지 않으면 상태 없음

if __name__ == '__main__':
    app.run(debug=True)


