// 각 난이도별로 표시할 이미지 배열 설정
var easyImages = [
    '../static/yoga_posture/dataset/virabhadrasana i/4-0.png',  // 쉬운 모드 이미지 1
    '../static/yoga_posture/dataset/hanumanasana/6-0.png',  // 쉬운 모드 이미지 2
    '../static/yoga_posture/dataset/virabhadrasana i/4-0.png',  // 쉬운 모드 이미지 3
    '../static/yoga_posture/dataset/hanumanasana/6-0.png',  // 쉬운 모드 이미지 4
    '../static/yoga_posture/dataset/hanumanasana/6-0.png'  // 쉬운 모드 이미지 5
];

var normalImages = [
    '../static/img/기초요가.jpeg',  // 일반 모드 이미지 1
    '../static/img/기초요가.jpeg',  // 일반 모드 이미지 2
    '../static/img/기초요가.jpeg'  // 일반 모드 이미지 3
];

var hardImages = [
    '../static/img/다리찢기.jpeg',  // 어려운 모드 이미지 1
    '../static/img/다리찢기.jpeg',  // 어려운 모드 이미지 2
    '../static/img/다리찢기.jpeg'  // 어려운 모드 이미지 3
];

// 갤러리 컨테이너 요소를 선택
var gallery = document.getElementById('gallery');

// 선택한 모드에 따라 이미지를 갤러리에 표시하는 함수
function showImages(mode) {
    gallery.innerHTML = '';  // 기존에 표시된 이미지를 제거

    var images = [];  // 표시할 이미지를 담을 배열
    if (mode === 'easy') {  // 쉬운 모드를 선택한 경우
        images = easyImages;  // 쉬운 모드 이미지 배열 설정
    } else if (mode === 'normal') {  // 일반 모드를 선택한 경우
        images = normalImages;  // 일반 모드 이미지 배열 설정
    } else if (mode === 'hard') {  // 어려운 모드를 선택한 경우
        images = hardImages;  // 어려운 모드 이미지 배열 설정
    }

    // 선택한 이미지 배열을 순회하며 갤러리에 이미지 요소 추가
    images.forEach(function(src) {
        var img = document.createElement('img');  // 이미지 요소 생성
        img.src = src;  // 이미지 경로 설정
        img.style.cursor = 'pointer';  // 클릭 가능하도록 커서 스타일 설정
        img.onclick = function() {  // 이미지 클릭 시 실행되는 함수
            handleClick(img);  // 클릭된 이미지에 대해 처리
        };
        gallery.appendChild(img);  // 갤러리 컨테이너에 이미지 추가
    });

    // 버튼 배경색을 변경하는 함수 호출 (현재는 주석 처리됨)
    /* highlightButton(mode + 'Button'); */
}

// 이미지 클릭 시 실행되는 함수
function handleClick() {
    window.location.href = '/game';  // 클릭된 이미지에 따라 게임 페이지로 이동
}

// 페이지가 로드될 때 초기화 작업 실행
window.onload = function() {
    // 초기 로드 시 'easyButton'의 배경색 설정
    document.getElementById('easyButton').style.backgroundColor = 'rgb(218, 218, 155)';
    // 쉬운 모드 이미지를 갤러리에 표시
    showImages('easy');
};