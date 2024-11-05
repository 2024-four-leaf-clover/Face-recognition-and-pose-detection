import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 손가락이 모두 펴져 있는지 확인하는 함수
def check_all_fingers_straight(landmarks):
    # 각 손가락 끝 (4, 8, 12, 16, 20)과 해당 손가락의 관절을 비교해 손가락이 펴져 있는지 확인
    for tip_id in [8, 12, 16, 20]:  # 엄지 (4번 landmark)는 별도로 검사
        tip_y = landmarks[tip_id].y
        base_y = landmarks[tip_id - 2].y
        if tip_y > base_y:  # 손가락 끝이 base에 비해 아래에 있으면 펴진 것으로 간주하지 않음
            return False
    return True

# 엄지 위치가 검지의 특정 방향에 있는지 확인하는 함수
def check_thumb_position(landmarks, hand_label):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    
    if hand_label == "Right":  # 오른손일 때
        if thumb_tip.x < index_tip.x:  # 엄지가 검지의 왼쪽에 위치해야 함
            return True
    elif hand_label == "Left":  # 왼손일 때
        if thumb_tip.x > index_tip.x:  # 엄지가 검지의 오른쪽에 위치해야 함
            return True
    return False

# 카메라 초기화
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2, # 양손 인식
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # RGB를 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmarks = hand_landmarks.landmark
                hand_label = handedness.classification[0].label  # "Left" 또는 "Right" 값

                # 손가락이 모두 펴져 있고 엄지가 검지의 특정 방향에 있는지 확인
                if check_all_fingers_straight(landmarks) and check_thumb_position(landmarks, hand_label):
                    cv2.putText(image, f"{hand_label} Hand Fully Opened with Thumb in Position", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 손 Landmark를 화면에 그리기
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 결과 화면에 표시
        cv2.imshow('Hand Detection', image)

        # ESC 키를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

#문제점 발생
#- 손가락이 다 펼쳐져 있을 때를 인식하는 거임
#    - 그래서 엄지가 손바닥쪽으로 기울어져있어도 마디가 접혀있지 않으면 정상적으로 인식함

#해결방법
#1단계: 엄지와 검지 거리 범위를 정하고 각도를 조정
#    - 오차범위 내에 평행해야 함
#    - 각도를 90도 이내로 함
#    - 실패
#2단계: 검지 기준으로 엄지 방향 설정
#    - 오른손: 엄지가 검지 왼쪽에 있어야 함
#    - 왼손: 엄지가 검지 오른쪽에 있어야 함
#    - 성공!!