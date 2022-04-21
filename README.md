# SMARCLE_MAKERS_DAY
+ 2021 여름방학 때 진행한 8주 팀 프로젝트   

[3팀 최종발표.pdf](https://github.com/kim-minsol/SMARCLE_MAKERS_DAY/files/8530192/3.pdf)


### 프로젝트 소개
![chrome_mEFzYW2Ctr](https://user-images.githubusercontent.com/81224613/164442397-9e20a7ea-626c-4d1a-9f36-3db109b5c952.jpg)

+ **의류 분류 및 재고 관리 시스템**

![image](https://user-images.githubusercontent.com/81224613/164442720-f794826b-5d14-4ee6-a80d-4581137228f7.png)
(의류 추천은 구현 x 분류까지만)
+ mobilenetv3_small 모델 사용
+ overfitting 크게 발생 -> Activation map 활용하여 모델이 어디를 보고 학습하는 지 파악
+ 우리가 상의를 분류할 때 목 부분을 보는 것처럼 의류의 목 부분을 잘라서 학습 진행 -> overfitting 해결


### 역할 분담
+ 아두이노 제작 및 UI
+ 이미지 셋 처리
+ 모델 학습 -> 담당

| 주차 | 활동 |
|---|---|
| 1주차 | 이미지 처리 |
| 2주차 | 모델 학습 |
| 3주차 | overfitting 발생 |
| 4주차 | CAM으로 모델 학습 파악 |
| 5주차 | 관심 영역 추출(haar cascade) |
| 6주차 | hyper parameter 조정 |
| 7주차 | 다른 모델들 적용 |
| 8주차(최종) | 최종 발표 |
