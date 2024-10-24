# 스요클(스마트 요가 클럽)
AI를 활용하여 자세 교정 및 재활에 도움을 주고, 최신 동향인 헬시플레저(건강과 기쁨의 합성어로, 건강을 즐겁게 관리한다라는 의미)로 지속 가능한 건강 관리를 도모하는 요가 게임 시스템

## 주요 기능
- AI 기술을 기반한 자세 교정으로 정확도 향상
- 얼굴 인식으로 인한 편한 로그인
- 카메라로 운동하는 사용자를 촬영해 모니터로 실시간 피드백 제공
- 운동 기록 저장 및 분석 시스템

## [finger_flask.py](https://github.com/2024-four-leaf-clover/Face-recognition-and-pose-detection/blob/main/finger_flask.py)
`finger_flask.py`는 `Flask`를 이용해 얼굴 및 자세 인식, 손 제스처 감지 기능을 제공하는 웹 애플리케이션을 설정한다. `OpenCV`와 `MediaPipe`를 통해 웹캠에서 얼굴 및 손 인식을 처리하며, 특정 제스처에 따라 회원가입(얼굴 등록)과 로그인(얼굴 인식)을 수행한다.

## [stt_flask.py](https://github.com/2024-four-leaf-clover/Face-recognition-and-pose-detection/blob/main/stt_flask.py)
`stt_flask.pyy`는 `Flask`와 `Mediapipe`, `OpenCV`를 사용해 음성 인식으로 회원가입와 로그인 기능을 처리한다. 또한 웹캠을 활용해 얼굴 및 자세 인식을 수행한다.

**(1) 회원가입과 로그인**
- 손동작에 따라 Python 로직 다르게 수행
    - 검지만 펼쳐졌을 때: 회원가입(`register`) 동작 수행
    - 검지와 중지만 펼쳐졌을 때: 로그인(`login`) 동작 수행
- 회원가입 시, `main.js`에서 프롬프트로 입력된 아이디와 촬영한 얼굴 정보를 Python이 `eyes.json`에 저장
- 로그인 시, 촬영된 얼굴 정보가 `eyes.json`에 저장된 값과 비교되어 인증을 진행

||사진|
|:---:|:---:|
|**아이디 입력**|![id_input](static/result/1.id_input.png)|
|**아이디 저장**|![id_complete](static/result/2.id_complete.png)|
|**얼굴 정보 저장**|![id_eyes.json](static/result/3.id_eyes.json.png)|
|**로그인 시도**|![login_attempt](static/result/4.login_attempt.png)|
|**로그인 성공**|![login_complete](static/result/5.login_complete.png)|
|**자세 인식**|![posture_detection](static/result/6.posture_detection.png)|

## 그외

### [app.py](https://github.com/2024-four-leaf-clover/Face-recognition-and-pose-detection/blob/main/merge/app.py)
`app.py`는 `Flask` 웹 프레임워크를 이용해 웹 서버를 만들고, 웹캠을 통해 손동작 및 얼굴 인식을 수행하는 로직을 포함한다. `MediaPipe` 라이브러리를 활용해 손동작과 얼굴을 감지하며, 이를 통해 회원가입 또는 로그인 동작을 수행한다. `main.html`에 `main.js`를 연결한다.

### [posture_flask.py](https://github.com/2024-four-leaf-clover/Face-recognition-and-pose-detection/blob/main/merge/posture_flask.py)
`posture_flask.py`는 `Flask `웹 프레임워크를 사용하여 오가 자세 인식을 수행하는 웹 애플리케이션을 구축한다. `MediaPipe` 라이브러리를 사용해 웹캠으로 사용자의 요가 자세를 인식하고, 미리 로드한 표준 요가 자세 이미지와 비교하여 인식 결과를 제공한다. `main.html`에 `main_stt.js`를 연결한다.

### [merge_flask.py](https://github.com/2024-four-leaf-clover/Face-recognition-and-pose-detection/blob/main/merge_flask.py)
`app.py`와 `posture_flask.py`를 결합한 파일
