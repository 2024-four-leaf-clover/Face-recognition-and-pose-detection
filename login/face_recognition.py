import cv2
import mediapipe as mp
import math
import json
import time

# MediaPipe 얼굴 메쉬 모듈 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# 3D 좌표 사이의 거리 계산 함수
def calculate_3d_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

# 얼굴 정보를 저장한 json 파일 경로
json_file = 'eyes.json'

# 얼굴 정보를 불러오는 함수
def load_face_data():
    try:
        with open(json_file, 'r') as f:
            face_data = json.load(f)
            return face_data.get('eye_distance')
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"{json_file} 파일을 불러오지 못했습니다.")
        return None

# 웹캠 설정
cap = cv2.VideoCapture(0)

# 얼굴 정보 비교
stored_eye_distance = load_face_data()

if stored_eye_distance is None:
    print("저장된 얼굴 정보가 없습니다. 먼저 얼굴 정보를 저장해주세요.")
    cap.release()
    cv2.destroyAllWindows()
else:
    frame_count = 0
    required_frames = 30  # 안정적으로 얼굴을 인식해야 하는 프레임 수

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("웹캠에서 영상을 가져오지 못했습니다.")
            break

        # BGR 이미지를 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 메쉬 추출
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            frame_count += 1
            if frame_count >= required_frames:
                for face_landmarks in results.multi_face_landmarks:
                    # 두 눈의 3D 랜드마크 좌표 (좌측 눈: 33번, 우측 눈: 263번)
                    left_eye = face_landmarks.landmark[33]
                    right_eye = face_landmarks.landmark[263]

                    # 3D 좌표로 거리 계산
                    left_eye_coords = (left_eye.x, left_eye.y, left_eye.z)
                    right_eye_coords = (right_eye.x, right_eye.y, right_eye.z)

                    # 두 눈 사이의 3D 거리 계산
                    eye_distance = calculate_3d_distance(left_eye_coords, right_eye_coords)
                    print(f"두 눈 사이 3D 거리: {eye_distance}")

                    # 저장된 거리와 현재 거리 비교
                    if abs(stored_eye_distance - eye_distance) < 0.05:  # 허용 오차 범위 지정
                        print("얼굴 인식 성공!")
                    else:
                        print("다른 사람입니다.")
                    break  # 비교 후 종료
        else:
            frame_count = 0  # 얼굴이 감지되지 않으면 카운트 초기화

        # 화면에 이미지 출력
        cv2.imshow('Face Recognition', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()