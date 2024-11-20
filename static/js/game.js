// 현재 단계 변수 초기화 (초기 단계는 1로 설정)
var currentStep = 1;

// 단계에 따라 p 태그의 텍스트를 업데이트하는 함수
function updateStep(step) {
    var stepTextElement = document.getElementById('stepText');  // stepText라는 id를 가진 요소를 찾음
    stepTextElement.textContent = 'STEP' + step;  // 해당 요소의 텍스트를 'STEP'과 현재 단계로 설정
    updateProgressBar(step);  // 단계 업데이트 후 진행 바도 함께 업데이트
}

// 진행 바를 업데이트하는 함수 (밑에서부터 색을 채워나감)
function updateProgressBar(step) {
    var bars = document.querySelectorAll('.bxbar');  // 모든 진행 바 요소를 선택
    var totalBars = bars.length;  // 진행 바의 총 개수를 가져옴

    bars.forEach(function(bar, index) {  // 각 진행 바에 대해 반복 실행
        if (totalBars - index <= step) {  // 현재 단계에 해당하는 만큼 진행 바에 색을 채움
            bar.style.backgroundColor = 'green';  // 조건에 맞는 진행 바는 녹색으로 채움
        } else {
            bar.style.backgroundColor = '';  // 그 외는 색을 채우지 않음
        }
    });
}

// 자세 인식 결과를 처리하는 함수
function handlePoseRecognition(success) {
    if (success) {  // 자세 인식이 성공했을 경우
        currentStep++;  // 현재 단계를 하나 증가시킴
        updateStep(currentStep);  // 증가한 단계에 맞게 화면을 업데이트
        console.log('게임 성공! 다음 단계로 이동:', currentStep);  // 성공 메시지를 콘솔에 출력
    } else {  // 자세 인식이 실패했을 경우
        console.log('자세가 올바르지 않습니다. 다시 시도해주세요.');  // 실패 메시지를 콘솔에 출력
    }
}

// 초기 단계 설정 (초기 페이지 로딩 시 현재 단계 표시)
updateStep(currentStep);

// 페이지 로드 시 posture_break.py 실행을 위한 함수 호출
//window.onload = fetchOutput;  // 페이지 로드가 완료되면 fetchOutput 함수를 호출

// 손바닥 감지 상태를 주기적으로 확인
setInterval(() => {
    fetch('/check-hand').then(response => {
        if (response.status === 200) {
            window.location.href = '/yoga';  // 감지되면 A 페이지로 이동
        }
    });
}, 1000);
/* setInterval(() => {
    console.log("Sending request to /check-hand...");  // 요청 전 로그 출력
    fetch('/check-hand')
        .then(response => {
            console.log("Response status:", response.status); // 응답 상태 출력
            if (response.status === 200) {
                console.log("Hand detected, redirecting to index page");
                window.location.href = '/';  // 감지되면 index 페이지로 이동
            }
        })
        .catch(error => {
            console.error("Error in fetch request:", error); // 에러 로그 출력
        });
}, 1000);  // 1초마다 상태 확인 */

console.log("Game.js loaded and running...");
console.log('hi');
