# Counting-Julnumgi

<2024년 이로운이들 교육연구회>의 연구 결과물입니다.
한국어 사용자를 위한 <줄넘기 세기> 프로그램입니다.
베이스 코드와 원리는 아래 레포지토리에서 기반합니다.
Thanks to "https://github.com/AyushTCD/VisionBasedJumpRopeCounter"


<Project Structure>
![image](https://github.com/user-attachments/assets/d7f2ef95-c1b9-4602-aa5a-173b644256db)





<System Requirements>
  
-Python 3.8 or higher

-CUDA-capable GPU recommended (for faster inference)

-Webcam (for live video input)

-At least 4GB RAM

-At least 2GB free disk space


<수정 및 개선할 사항>

인체 인식 정확도를 높이기 위해 객체탐지모델을 Yolov8로 변경했습니다.

4가지 형태(기본 뛰기, 번갈아 뛰기, 옆 흔들어 뛰기, 엇걸어 풀어 뛰기)의 인식 알고리즘을 구분하지 않고 통합했습니다.

기존의 로컬머신 베이스 코드를 streamlit 기반 웹앱으로 개선했습니다.

웹캠을 활용한 실시간 분석이 가능하도록 개선했습니다.

영상 분석 결과를 result 디렉토리에 저장하도록 개선했습니다.

*시작시 몇개가 한번에 올라가는 현상을 개선하겠습니다.

*스마트폰에서 실행할 수 있도록 개선하겠습니다.

*웹캠 분석결과 파일이 일부 컴퓨터에서 재생문제가 있습니다.
