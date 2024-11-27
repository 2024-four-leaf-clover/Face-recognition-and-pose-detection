// 각 난이도별로 표시할 이미지 배열 설정
var easyImages = [
    '/static/yoga_posture/dataset/virabhadrasana i/4-0.png',
    '/static/yoga_posture/dataset/hanumanasana/6-0.png',
    '/static/yoga_posture/dataset/virabhadrasana i/4-0.png',
    '/static/yoga_posture/dataset/hanumanasana/6-0.png',
    '/static/yoga_posture/dataset/hanumanasana/6-0.png'
];

var normalImages = [
    '/static/img/기초요가.jpeg',
    '/static/img/기초요가.jpeg',
    '/static/img/기초요가.jpeg'
];

var hardImages = [
    '/static/img/다리찢기.jpeg',
    '/static/img/다리찢기.jpeg',
    '/static/img/다리찢기.jpeg'
];

// 갤러리 컨테이너 요소를 선택
var gallery = document.getElementById('gallery');

// 선택한 모드에 따라 이미지를 갤러리에 표시하는 함수
function showImages(mode) {
    gallery.innerHTML = '';  // 기존에 표시된 이미지를 제거

    var images = [];  // 표시할 이미지를 담을 배열
    if (mode === 'easy') {
        images = easyImages;
    } else if (mode === 'normal') {
        images = normalImages;
    } else if (mode === 'hard') {
        images = hardImages;
    }

    // 선택한 이미지 배열을 순회하며 갤러리에 이미지 요소 추가
    images.forEach(function(src) {
        var img = document.createElement('img');
        img.src = src;
        img.alt = 'Yoga Pose';
        img.style.cursor = 'pointer';
        img.style.maxWidth = '100%';  // 이미지 크기 조정
        img.onclick = function() {
            handleClick(img);
        };
        gallery.appendChild(img);
    });
}

// 이미지 클릭 시 실행되는 함수
/* function handleClick() {
    // 쿼리 매개변수에서 user_id를 가져오기
    var urlParams = new URLSearchParams(window.location.search);
    var userId = urlParams.get('user_id');  // user_id 가져오기

    if (userId) {
        // user_id를 포함하여 game.html로 이동
        window.location.href = `/game?user_id=${userId}`;
    } else {
        // user_id가 없는 경우에도 이동하지만 기본값 처리 가능
        window.location.href = '/game';
    }
} */
function handleClick() {
    window.location.href = '/game';  // 바로 game.html로 이동
}

// 초기 설정: 페이지 로드 시
document.addEventListener('DOMContentLoaded', () => {
    // 'easy' mode 로드
    showImages('easy');

    const buttons = document.querySelectorAll('.button');

    buttons.forEach(button => {
    button.addEventListener('click', () => {
        // 모든 버튼의 색상을 베이지로 변경
        buttons.forEach(btn => btn.classList.remove('selected'));

        // 클릭된 버튼에 selected 클래스 추가
        button.classList.add('selected');
        });
    });
});
