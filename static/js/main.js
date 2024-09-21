window.onload = function() {
    startVideoFeed();
};

function startVideoFeed() {
    fetch('/video_feed')
    .then(response => response.json())
    .then(data => {
        if (data.detected) {
            if (data.action === 'register') {
                promptUserId('/register');  // 회원가입: 아이디 입력 프롬프트 실행
            } else if (data.action === 'login') {
                loginUser();  // 로그인: 얼굴 인식으로 직접 로그인
            }
        } else {
            alert("손 모양 인식에 실패했습니다.");
        }
    })
    .catch(error => console.error("Error:", error));
}

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
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert("오류가 발생했습니다. 다시 시도해주세요.");
        });
        break;
    }
}

function loginUser() {
    // 검지와 중지를 펼쳤을 때는 아이디를 입력하지 않고 바로 얼굴 인식을 통해 로그인 진행
    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);  // 얼굴 불일치 등의 에러 메시지 표시
            startVideoFeed();  // 얼굴 인식 실패 시 다시 손가락 감지로 돌아감
        } else if (data.redirect_url) {
            window.location.href = data.redirect_url;  // 얼굴 인식 성공 시 yoga.html로 이동
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert("오류가 발생했습니다. 다시 시도해주세요.");
    });
}
