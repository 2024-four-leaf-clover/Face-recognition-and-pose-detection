window.onload = function() {
    fetch('/video_feed')
    .then(response => response.json())
    .then(data => {
        if (data.detected) {
            if (data.action === 'register') {
                promptUserId('/register');  // 회원가입: 아이디 입력 프롬프트 실행
            } else if (data.action === 'login') {
                loginUser();  // 로그인: 검지와 중지가 펼쳐지면 얼굴 인식으로 로그인
            }
        } else {
            alert("손 모양 인식에 실패했습니다.");
        }
    })
    .catch(error => console.error("Error:", error));
};

function promptUserId(endpoint) {
    let userId;
    while (true) {
        userId = prompt("아이디를 입력해주세요 (영어와 숫자의 조합으로 6자 이상, 첫 시작은 영어)");
        if (!userId) {
            alert("아이디를 입력해야 합니다.");
            continue;
        }
        if (!/^[a-zA-Z][a-zA-Z0-9]*$/.test(userId) || !/\d/.test(userId) || !/[a-zA-Z]/.test(userId) || userId.length < 6) {
            alert("아이디는 영어와 숫자의 조합으로 6자 이상이어야 하며 첫 시작은 영어여야 합니다.");
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
                alert(data.error);  // 이미 등록된 아이디가 있으면 메시지 표시
                promptUserId(endpoint);  // 다시 프롬프트 실행
            } else {
                alert(data.message);
                // 아이디 등록 후 카메라 피드를 다시 표시
                startCamera();  // 카메라 피드를 다시 띄움
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert("오류가 발생했습니다. 다시 시도해주세요.");
        });
        break;
    }
}

function startCamera() {
    // 카메라 피드를 다시 띄우기
    fetch('/video_feed')
    .then(response => response.json())
    .then(data => {
        if (data.detected) {
            if (data.action === 'login') {
                loginUser();  // 손가락 제스처에 따라 로그인 진행
            }
        }
    })
    .catch(error => console.error('Error:', error));
}

function loginUser() {
    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (response.redirected) {
            window.location.href = response.url;  // 리다이렉션된 URL로 이동
        } else {
            return response.json();
        }
    })
    .then(data => {
        if (data && data.error) {
            alert(data.error);  // 에러 메시지 표시
        } else if (data) {
            alert(data.message);  // 성공 메시지
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert("오류가 발생했습니다. 다시 시도해주세요.");
    });
}
