window.onload = function() {
    fetch('/video_feed')
    .then(response => response.json())
    .then(data => {
        if (data.detected) {
            promptUserId();
        } else {
            alert("검지 인식에 실패했습니다.");
        }
    })
    .catch(error => console.error("Error:", error));
};

function promptUserId() {
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

        fetch('/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_id: userId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
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