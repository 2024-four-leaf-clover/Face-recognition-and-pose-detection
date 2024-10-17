// 알림 메시지를 표시하고 3초 후에 자동으로 사라지게 하는 함수
function showNotification(message) {
    var notification = document.getElementById('notification');
    var notificationText = document.getElementById('notificationText');
    
    notificationText.textContent = message;
    notification.style.display = 'block';  // 알림 메시지 보이기

    // 3초 후에 알림 메시지 숨기기
    setTimeout(function() {
        notification.style.display = 'none';
    }, 3000);  // 3000ms = 3초
}

// 페이지가 로드될 때 실행되는 함수
window.onload = function() {
    startVoiceRecognition();
};

// 음성 인식 시작하는 함수
function startVoiceRecognition() {
    showNotification("음성 인식을 시작합니다...");
    console.log("음성 인식 시작됨...");

    // 서버에서 음성 인식 결과를 받아오기
    fetch('/start_voice_recognition', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        console.log("사용자가 말한 내용:", data.command);
        showNotification("사용자가 말한 내용: " + data.command);  // 선택사항

        if (data.action === 'register') {
            // 회원가입 POST 요청
            let userId = prompt("아이디를 입력해주세요 (영어와 숫자의 조합으로 6자 이상)");
            if (userId) {
                fetch('/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ user_id: userId })  // 사용자 입력 받은 아이디를 서버로 전송
                })
                .then(response => response.json())
                .then(data => {
                    showNotification(data.message);
                })
                .catch(error => console.error('Error:', error));
            }

        } else if (data.action === 'login') {
            // 로그인 POST 요청
            fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => {
                if (response.redirected) {
                    window.location.href = response.url;
                } else {
                    return response.json();
                }
            })
            .then(data => {
                if (data && data.error) {
                    showNotification(data.error);
                } else if (data) {
                    showNotification(data.message);
                }
            })
            .catch(error => console.error('Error:', error));
        } else {
            showNotification("음성 명령 인식에 실패했습니다. 다시 시도해주세요.");
        }
    })
    .catch(error => console.error("Error:", error));
}


// 아이디 입력 프롬프트를 띄우는 함수
function promptUserId(endpoint) {
    let userId = prompt("아이디를 입력해주세요 (영어와 숫자의 조합으로 6자 이상, 첫 시작은 영어)");

    if (userId === null) {  // 사용자가 '취소'를 누른 경우
        startVoiceRecognition();  // 음성 인식 다시 시작
        return;  // 함수 종료
    }

    while (true) {
        if (!userId) {  // 아이디가 입력되지 않으면
            showNotification("아이디를 입력해야 합니다.");
            userId = prompt("아이디를 입력해주세요 (영어와 숫자의 조합으로 6자 이상, 첫 시작은 영어)");
            if (userId === null) {  // 사용자가 '취소'를 누른 경우
                startVoiceRecognition();  // 음성 인식 다시 시작
                return;  // 함수 종료
            }
            continue;
        }

        if (!/^[a-zA-Z][a-zA-Z0-9]*$/.test(userId) || !/\d/.test(userId) || !/[a-zA-Z]/.test(userId) || userId.length < 6) {
            showNotification("아이디는 영어와 숫자의 조합으로 6자 이상이어야 하며 첫 시작은 영어여야 합니다.");
            userId = prompt("아이디를 입력해주세요 (영어와 숫자의 조합으로 6자 이상, 첫 시작은 영어)");
            if (userId === null) {  // 사용자가 '취소'를 누른 경우
                startVoiceRecognition();  // 음성 인식 다시 시작
                return;  // 함수 종료
            }
            continue;
        }

        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_id: userId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showNotification(data.error);
                promptUserId(endpoint);  // 다시 프롬프트 실행
            } else {
                showNotification(data.message);
                startVoiceRecognition();  // 음성 인식 다시 시작
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification("오류가 발생했습니다. 다시 시도해주세요.");
        });
        break;  // 조건을 만족하고 요청을 성공적으로 보낸 후 반복 종료
    }
}

// 로그인 처리를 하는 함수
function loginUser() {
    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (response.redirected) {
            window.location.href = response.url;
        } else {
            return response.json();
        }
    })
    .then(data => {
        if (data && data.error) {
            showNotification(data.error);
            if (data.error === "얼굴 정보가 일치하지 않습니다." || data.error === "등록된 얼굴 정보가 없습니다.") {
                startVoiceRecognition();  // 음성 인식 다시 시작
            }
        } else if (data) {
            showNotification(data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification("오류가 발생했습니다. 다시 시도해주세요.");
    });
}