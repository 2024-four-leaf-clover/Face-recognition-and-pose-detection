# SYC 웹 사이트 구축

## 1. login 폴더에 있는 파일
### (1) face_recognition.py
- eyes.json에 저장한 얼굴 정보를 바탕으로 얼굴 인식하는 파이썬 코드
- MediaPipe 코드 활용

### (2) face_store.py
- 얼굴 정보(눈 자간 3D 좌표 거리) eyes.json 파일에 저장하는 파이썬 코드
- MediaPipe 코드 활용

### (3) logIn.py
- 검지와 중지를 펼쳤을 때를 감지하는 파이썬 코드
- MediaPipe의 손 랜드마크 인지 파이썬 코드를 활용
- 로그인할 때 참고할 파이썬 코드

### (4) logOut.py
- 검지, 중지, 약지를 펼쳤을 때를 감지하는 파이썬 코드
- MediaPipe의 손 랜드마크 인지 파이썬 코드를 활용
- 로그아웃할 때 참고할 파이썬 코드

### (5) signUp.py
- 검지만 펼쳤을 때를 감지하는 파이썬 코드
- MediaPipe의 손 랜드마크 인지 파이썬 코드를 활용
- 회원가입할 때 참고할 파이썬 코드

## 2. static 폴더에 있는 폴더
### (1) css
- html을 꾸밀 css 파일을 담은 폴더

### (2) img
- html에 사용된 img 파일을 담은 폴더

### (3) js
- html에 사용된 js 파일을 담은 폴더

## 3. templates 폴더
- html 파일을 담은 폴더

## 4. app.py
- SYC 웹 사이트를 구축시킬 flask 파일

## 5. eyes.json
- face_store.py를 실행시켜 카메라로 촬영한 사용자의 얼굴 자간 정보를 저장할 json 파일