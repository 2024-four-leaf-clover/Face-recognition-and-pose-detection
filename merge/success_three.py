import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 손가락 펼침 여부 확인 함수
def is_finger_extended(finger_tip, finger_dip, palm_direction):
    # 손등을 등졌을 때, 손가락 팁이 DIP보다 더 멀리 있는지 확인
    if palm_direction > 0:  # 손등이 카메라를 등짐
        return finger_tip.y < finger_dip.y
    else:  # 손바닥이 카메라를 향함
        return finger_tip.y > finger_dip.y

# 손등 방향 확인 함수
def is_hand_back_facing(wrist, middle_finger_base):
    return wrist.z > middle_finger_base.z

# 검지, 중지, 약지만 펴진 상태 확인
def is_three_fingers_extended(hand_landmarks):
    fingers = [8, 12, 16, 20]  # 검지, 중지, 약지, 새끼 손가락 팁
    dips = [6, 10, 14, 18]     # DIP 위치
    
    # 손등 방향 확인
    wrist = hand_landmarks.landmark[0]
    middle_finger_base = hand_landmarks.landmark[9]
    palm_direction = 1 if is_hand_back_facing(wrist, middle_finger_base) else -1
    
    # 검지, 중지, 약지 상태 확인
    is_index_extended = is_finger_extended(hand_landmarks.landmark[8], hand_landmarks.landmark[6], palm_direction)
    is_middle_extended = is_finger_extended(hand_landmarks.landmark[12], hand_landmarks.landmark[10], palm_direction)
    is_ring_extended = is_finger_extended(hand_landmarks.landmark[16], hand_landmarks.landmark[14], palm_direction)
    is_pinky_folded = not is_finger_extended(hand_landmarks.landmark[20], hand_landmarks.landmark[18], palm_direction)
    
    # 새끼손가락이 접혀 있고, 나머지 세 손가락이 펼쳐져 있는 경우
    return is_index_extended and is_middle_extended and is_ring_extended and is_pinky_folded

# 웹캠 열기
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라에서 이미지를 읽을 수 없습니다.")
            break

        # 이미지를 BGR에서 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # MediaPipe Hands 처리
        results = hands.process(image)

        # 이미지를 다시 BGR로 변환 (OpenCV에서 사용)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 검지, 중지, 약지만 펴졌는지 확인
                if is_three_fingers_extended(hand_landmarks):
                    cv2.putText(image, "Three Fingers Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2, cv2.LINE_AA)

        # 결과 출력
        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):  # 'q'키를 누르면 종료
            break

cap.release()
cv2.destroyAllWindows()
