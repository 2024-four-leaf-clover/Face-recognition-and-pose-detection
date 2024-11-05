import cv2
import mediapipe as mp

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 손가락 상태 판별 함수 정의
def is_finger_up(landmarks, finger_tip, finger_pip):
    # 손가락 끝이 두 번째 마디보다 위에 있으면 손가락이 펴진 상태로 간주
    return landmarks[finger_tip].y < landmarks[finger_pip].y

# 다섯 손가락이 모두 펼쳐져 있는지 확인하는 함수
def all_fingers_open(landmarks):
    # 손가락 랜드마크 인덱스 설정
    THUMB_TIP, THUMB_IP = 4, 3
    INDEX_FINGER_TIP, INDEX_FINGER_PIP = 8, 6
    MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP = 12, 10
    RING_FINGER_TIP, RING_FINGER_PIP = 16, 14
    PINKY_FINGER_TIP, PINKY_FINGER_PIP = 20, 18

    # 모든 손가락이 펴져 있는지 확인
    return (
        is_finger_up(landmarks, THUMB_TIP, THUMB_IP) and  # 엄지가 펼쳐진 상태
        is_finger_up(landmarks, INDEX_FINGER_TIP, INDEX_FINGER_PIP) and
        is_finger_up(landmarks, MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP) and
        is_finger_up(landmarks, RING_FINGER_TIP, RING_FINGER_PIP) and
        is_finger_up(landmarks, PINKY_FINGER_TIP, PINKY_FINGER_PIP)
    )

# 웹캠으로 실시간 손 인식
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 프레임을 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    # BGR로 다시 변환하여 표시
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 다섯 손가락이 모두 펼쳐졌는지 확인
            if all_fingers_open(hand_landmarks.landmark):
                cv2.putText(image, "All fingers are open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#지금 문제점
#엄지가 접혀있지는 않지만 손바닥을 향해 기울여진 상태인데 정상적으로 인식함