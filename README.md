# 2022_CapstoneDesign
실시간 객체 트래킹을 활용한 동적 보행 신호 시스템
YOLO v5 및 Strong-SORT사용
Strong-SORT알고리즘은
https://github.com/dyhBUPT/StrongSORT
해당 알고리즘 활용하여 my_track.py로 커스텀함

데이터 셋은
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=189
AI 데이터 인도 보행영상 활용함

라벨링 후 데이터 셋은 Roboflow에 업로드함
https://app.roboflow.com/cwnu/ysjun-capstone-crosswalk/4

모델 학습 및 객체 인식은 Google Colab을 이용함


<img width="1440" alt="스크린샷 2022-11-13 21 12 28" src="https://user-images.githubusercontent.com/82963112/201521230-ab6248fd-f7a3-4442-8a05-97aafcdf5e26.png">
<img width="1440" alt="스크린샷 2022-11-13 21 12 38" src="https://user-images.githubusercontent.com/82963112/201521234-7309c5dc-45fb-4a19-944b-a5b32ceae2ad.png">
<img width="1440" alt="스크린샷 2022-11-13 21 12 44" src="https://user-images.githubusercontent.com/82963112/201521237-187cdc01-1814-4b37-bc58-fb09305a6ed6.png">
<img width="1440" alt="스크린샷 2022-11-13 21 12 51" src="https://user-images.githubusercontent.com/82963112/201521238-41e38429-91f9-470d-8094-fc23dccc7137.png">
<img width="1440" alt="스크린샷 2022-11-13 21 12 58" src="https://user-images.githubusercontent.com/82963112/201521239-0a8bec00-055e-4375-9808-bca5943a082b.png">
<img width="1440" alt="스크린샷 2022-11-13 21 13 04" src="https://user-images.githubusercontent.com/82963112/201521242-9dfffd14-2c58-4f7c-b611-b4cf4f74ba26.png">
<img width="1440" alt="스크린샷 2022-11-13 21 13 11" src="https://user-images.githubusercontent.com/82963112/201521243-7c3447d3-e74e-4eb6-b158-243135d05f13.png">
