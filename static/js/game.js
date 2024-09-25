// 현재 단계 (초기 단계는 2로 설정)
var currentStep = 1;

// 단계에 따라 p 태그의 텍스트를 업데이트하는 함수
function updateStep(step) {
    var stepTextElement = document.getElementById('stepText');
    stepTextElement.textContent = 'STEP' + step;
    updateProgressBar(step);  // 단계 업데이트 후 진행 바도 업데이트
}

// 진행 바를 업데이트하는 함수 (밑에서부터 색을 채움)
function updateProgressBar(step) {
    var bars = document.querySelectorAll('.bxbar');
    var totalBars = bars.length;

    bars.forEach(function(bar, index) {
        if (totalBars - index <= step) {
            bar.style.backgroundColor = 'green';
        } else {
            bar.style.backgroundColor = '';
        }
    });
}

// 자세 인식 결과를 처리하는 함수
function handlePoseRecognition(success) {
    if (success) {
        currentStep++; // 성공 시 다음 단계로 이동
        updateStep(currentStep); // 단계 업데이트
        console.log('게임 성공! 다음 단계로 이동:', currentStep);
    } else {
        console.log('자세가 올바르지 않습니다. 다시 시도해주세요.');
    }
}

// 초기 단계 설정
updateStep(currentStep);

// 예제 사용법:
// 자세 인식이 성공하면 handlePoseRecognition(true)를 호출
// 자세 인식이 실패하면 handlePoseRecognition(false)를 호출

// 예제를 위해 3초 후에 자세 인식 성공으로 간주
/* setTimeout(function() {
    handlePoseRecognition(true); // 자세 인식 성공 시
}, 3000); */

// 클릭한 이미지 파일명 title과 yogaName으로 설정
/* window.addEventListener('DOMContentLoaded', function() {
    const filename = decodeURIComponent(sessionStorage.getItem('imageFilename'));   // URL 디코딩
    console.log(filename);
    if (filename) {
        document.title = filename;
        document.getElementById('pageTitle').textContent = filename; // pageTitle 태그에 filename 추가
        document.getElementById('yogaName').textContent = filename; // st_text 태그에 filename 추가
    }
}); */

// 페이지 로드 시 posture_break.py 실행
window.onload = fetchOutput;

