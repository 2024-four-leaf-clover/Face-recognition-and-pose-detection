// 이미지 배열
var easyImages = [
    '../static/yoga_posture/dataset/virabhadrasana i/4-0.png',
    '../static/yoga_posture/dataset/hanumanasana/6-0.png',
    '../static/yoga_posture/dataset/virabhadrasana i/4-0.png',
    '../static/yoga_posture/dataset/hanumanasana/6-0.png',
    '../static/yoga_posture/dataset/hanumanasana/6-0.png',
];

var normalImages = [
    '../static/img/기초요가.jpeg',
    '../static/img/기초요가.jpeg',
    '../static/img/기초요가.jpeg'
];

var hardImages = [
    '../static/img/다리찢기.jpeg',
    '../static/img/다리찢기.jpeg',
    '../static/img/다리찢기.jpeg'
];

// 갤러리 컨테이너 선택
var gallery = document.getElementById('gallery');

// 이미지 표시 함수
function showImages(mode) {
    gallery.innerHTML = ''; // 기존 이미지 제거

    var images = [];
    if (mode === 'easy') {
        images = easyImages;
    } else if (mode === 'normal') {
        images = normalImages;
    } else if (mode === 'hard') {
        images = hardImages;
    }

    // 이미지 갤러리에 추가
    images.forEach(function(src) {
        var img = document.createElement('img');
        img.src = src;
        img.style.cursor = 'pointer';
        img.onclick = function() {
            handleClick(img);
        };
        gallery.appendChild(img);
    });

    // 버튼 배경색 변경
    /* highlightButton(mode + 'Button'); */
}

// 이미지 클릭 시 호출되는 함수
function handleClick() {
    window.location.href = '/game';
}

// 페이지가 로드될 때 초기화 함수 호출
window.onload = function() {
    document.getElementById('easyButton').style.backgroundColor = 'rgb(218, 218, 155)';
    showImages('easy');
};

