import cv2
import mediapipe as mp

# 미디어파이프 손 감지 모듈 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# 웹캠 실행
cap = cv2.VideoCapture(0)

def is_finger_up(landmarks, finger_tip, finger_pip):
    # 손가락이 펴져 있는지 여부를 판단 (손가락 끝이 첫번째 관절보다 위에 있는지 여부 확인)
    return landmarks[finger_tip].y < landmarks[finger_pip].y

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    # 다시 BGR로 변환 (화면 표시용)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 각 손가락의 랜드마크 좌표 가져오기
            landmarks = hand_landmarks.landmark
            
            # 검지 손가락 랜드마크 인덱스
            INDEX_FINGER_TIP = 8  # 검지 끝 랜드마크
            INDEX_FINGER_PIP = 6  # 검지 관절 랜드마크

            # 중지, 약지, 새끼 손가락 끝 랜드마크
            MIDDLE_FINGER_TIP = 12
            RING_FINGER_TIP = 16
            PINKY_FINGER_TIP = 20

            # 중지, 약지, 새끼손가락 관절 랜드마크
            MIDDLE_FINGER_PIP = 10
            RING_FINGER_PIP = 14
            PINKY_FINGER_PIP = 18

            # 검지가 펴져 있는지 확인
            index_finger_up = is_finger_up(landmarks, INDEX_FINGER_TIP, INDEX_FINGER_PIP)

            # 나머지 손가락(중지, 약지, 새끼손가락)은 접혀 있는지 확인
            middle_finger_down = not is_finger_up(landmarks, MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP)
            ring_finger_down = not is_finger_up(landmarks, RING_FINGER_TIP, RING_FINGER_PIP)
            pinky_finger_down = not is_finger_up(landmarks, PINKY_FINGER_TIP, PINKY_FINGER_PIP)

            # 검지만 펴져 있으면 숫자 "1" 감지
            if index_finger_up and middle_finger_down and ring_finger_down and pinky_finger_down:
                cv2.putText(image, "1 Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print("숫자 1 감지됨!")
                # 'q' 키 누르지 않아도 감지되면 알아서 프로그램 종료
                cap.release()
                cv2.destroyAllWindows()
                hands.close()
                exit()  # 프로그램 완전히 종료
            else:
                print("숫자 1이 감지되지 않음.")

    # 화면에 이미지 출력
    cv2.imshow('Hand Detection', image)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 웹캠 종료 및 리소스 해제
cap.release()
cv2.destroyAllWindows()
hands.close()