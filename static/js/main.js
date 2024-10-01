// 페이지가 로드될 때 실행되는 함수
window.onload = function() {
    fetch('/video_feed')  // '/video_feed'로 요청을 보냄
    .then(response => response.json())  // JSON 형식으로 응답을 받음
    .then(data => {
        if (data.detected) {  // 손 모양이 감지되었을 경우
            if (data.action === 'register') {  // 감지된 동작이 'register'이면
                promptUserId('/register');  // 회원가입: 아이디 입력 프롬프트 실행
            } else if (data.action === 'login') {  // 감지된 동작이 'login'이면
                loginUser();  // 로그인: 얼굴 인식을 통한 로그인 실행
            }
        } else {  // 손 모양 인식 실패 시
            alert("손 모양 인식에 실패했습니다.");  // 경고창 표시
        }
    })
    .catch(error => console.error("Error:", error));  // 에러가 발생했을 때 콘솔에 에러 출력
};

// 아이디 입력 프롬프트를 띄우는 함수
function promptUserId(endpoint) {
    let userId;
    while (true) {  // 유효한 아이디가 입력될 때까지 반복
        userId = prompt("아이디를 입력해주세요 (영어와 숫자의 조합으로 6자 이상, 첫 시작은 영어)");  // 사용자에게 아이디 입력을 요청
        if (!userId) {  // 아이디가 입력되지 않으면
            alert("아이디를 입력해야 합니다.");  // 경고창을 띄우고 다시 프롬프트 실행
            continue;
        }
        // 아이디가 조건에 맞지 않으면 다시 입력 요청
        if (!/^[a-zA-Z][a-zA-Z0-9]*$/.test(userId) || !/\d/.test(userId) || !/[a-zA-Z]/.test(userId) || userId.length < 6) {
            alert("아이디는 영어와 숫자의 조합으로 6자 이상이어야 하며 첫 시작은 영어여야 합니다.");  // 아이디 조건을 알려줌
            continue;
        }

        // 조건을 만족한 경우 아이디를 서버에 전달하여 등록 요청
        fetch(endpoint, {
            method: 'POST',  // POST 요청으로 아이디를 서버에 전달
            headers: {
                'Content-Type': 'application/json'  // 요청의 내용이 JSON 형식임을 명시
            },
            body: JSON.stringify({ user_id: userId })  // user_id를 JSON 형태로 보냄
        })
        .then(response => response.json())  // 응답을 JSON으로 받음
        .then(data => {
            if (data.error) {  // 에러가 발생한 경우
                alert(data.error);  // 에러 메시지를 경고창에 표시
                promptUserId(endpoint);  // 다시 프롬프트 실행
            } else {  // 성공 시
                alert(data.message);  // 성공 메시지를 경고창에 표시
                // 아이디 등록 후 카메라 피드를 다시 시작
                startCamera();  // 카메라 피드 다시 시작
            }
        })
        .catch(error => {
            console.error('Error:', error);  // 에러를 콘솔에 출력
            alert("오류가 발생했습니다. 다시 시도해주세요.");  // 경고창에 에러 메시지 표시
        });
        break;  // 조건을 만족하고 요청을 성공적으로 보낸 후 반복 종료
    }
}

// 카메라 피드를 다시 시작하는 함수
function startCamera() {
    // '/video_feed'로 다시 요청을 보내 카메라 피드를 시작
    fetch('/video_feed')
    .then(response => response.json())  // 응답을 JSON으로 받음
    .then(data => {
        if (data.detected) {  // 손 모양이 감지되었을 경우
            if (data.action === 'login') {  // 감지된 동작이 'login'이면
                loginUser();  // 로그인 실행
            } else if (data.action === 'register') {  // 감지된 동작이 'register'이면
                promptUserId('/register');  // 회원가입을 위한 아이디 입력 프롬프트 실행
            }
        }
    })
    .catch(error => console.error('Error:', error));  // 에러가 발생했을 경우 콘솔에 에러 출력
}

// 로그인 처리를 하는 함수
function loginUser() {
    fetch('/login', {
        method: 'POST',  // POST 요청으로 로그인 시도
        headers: {
            'Content-Type': 'application/json'  // 요청의 내용이 JSON 형식임을 명시
        }
    })
    .then(response => {
        if (response.redirected) {  // 로그인 성공 시 리다이렉션 여부 확인
            window.location.href = response.url;  // 성공 시 리다이렉트된 URL로 이동
        } else {
            return response.json();  // 리다이렉트되지 않은 경우 JSON 응답 처리
        }
    })
    .then(data => {
        if (data && data.error) {  // 에러 메시지가 있으면
            alert(data.error);  // 에러 메시지를 경고창에 표시
            
            // 얼굴 정보가 일치하지 않거나 등록된 얼굴 정보가 없으면 카메라 피드를 다시 시작
            if (data.error === "얼굴 정보가 일치하지 않습니다." || data.error === "등록된 얼굴 정보가 없습니다.") {
                startCamera();  // 카메라 피드 다시 시작
            }
        } else if (data) {  // 성공 메시지가 있으면
            alert(data.message);  // 성공 메시지를 경고창에 표시
        }
    })
    .catch(error => {
        console.error('Error:', error);  // 에러 발생 시 콘솔에 에러 출력
        alert("오류가 발생했습니다. 다시 시도해주세요.");  // 경고창에 에러 메시지 표시
    });
}