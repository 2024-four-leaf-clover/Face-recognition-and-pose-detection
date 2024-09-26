# 스요클(스마트 요가 클럽)
AI를 활용하여 자세 교정 및 재활에 도움을 주고, 최신 동향인 헬시플레저(건강과 기쁨의 합성어로, 건강을 즐겁게 관리한다라는 의미)로 지속 가능한 건강 관리를 도모하는 요가 게임 시스템

## 주요 기능
- AI 기술을 기반한 자세 교정으로 정확도 향상
- 얼굴 인식으로 인한 편한 로그인
- 카메라로 운동하는 사용자를 촬영해 모니터로 실시간 피드백 제공
- 운동 기록 저장 및 분석 시스템

## [app.py](https://github.com/2024-four-leaf-clover/Face-recognition-and-pose-detection/blob/main/app.py)
`app.py`는 `Flask` 웹 프레임워크를 이용해 웹 서버를 만들고, 웹캠을 통해 손동작 및 얼굴 인식을 수행하는 로직을 포함한다. `MediaPipe` 라이브러리를 활용해 손동작과 얼굴을 감지하며, 이를 통해 회원가입 또는 로그인 동작을 수행한다.

## [posture_flask.py](https://github.com/2024-four-leaf-clover/Face-recognition-and-pose-detection/blob/main/posture_flask.py)
`posture_flask.py`는 `Flask `웹 프레임워크를 사용하여 오가 자세 인식을 수행하는 웹 애플리케이션을 구축한다. `MediaPipe` 라이브러리를 사용해 웹캠으로 사용자의 요가 자세를 인식하고, 미리 로드한 표준 요가 자세 이미지와 비교하여 인식 결과를 제공한다.

## [merge_flask.py](https://github.com/2024-four-leaf-clover/Face-recognition-and-pose-detection/blob/main/merge_flask.py)
`app.py`와 `posture_flask.py`를 결합한 파일

**(1) 회원가입과 로그인**
|아이디 입력|아이디 저장|
|---|----|
|![id_input](static/result/id_input.png)|![id_complete](static/result/id_complete.png)|

1. 손 동작 인식<br>
- `merge_flask.py`
- `index_finger_up`이 `True`이고 나머지 손가락은 `False`이면 검지만 펼쳐진 상태로 인식 (`register`) 
- `index_finger_up`과 `middle_finger_up`이 `True`이고 나머지 손가락이 `False`이면 검지와 중지만 펼쳐진 상태로 인식 (`login`) 
- 해당 결과는 `/video_feed API`를 통해 클라이언트로 전달<br><br>

2. 화면 로딩 후 손 동작 처리<br>
- `main.js`
- 페이지 로딩 시 `/video_feed` API 호출로 손 동작 상태 확인
- `register` 동작 시 `promptUserId`로 회원가입 프롬포트 실행
- `login` 동작 시 `loginUser`로 로그인 처리<br><br>

3. 회원가입과 로그인 로직<br>
- `merge_flask.py`, `main.js`
- 로그인
    - `promptUserId`: 아이디 입력 프롬포트 실행
    - 조건에 부합하는 아이디가 입력되면 `/register`로 전달
    - Python에서 얼굴 정보를 촬영해 `eyes.json` 파일에 아이디와 함께 저장
- 회원가입
    - `loginUser`: 카메라로 얼굴을 촬영한 후 `eyes.json`에 저장된 값과 비교
    - 일치 시 로그인 성공 처리, 불일칠 시 오류 메시지 표시 후 재시도


