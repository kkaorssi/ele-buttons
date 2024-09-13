# Elevator button recognition and placement estimation system for unmanned delivery robots
## Overview
이 프로젝트는 무인 배달 로봇에 적용할 엘리베이터 버튼 인식 모델과 버튼 배치 추정 방법, 그리고 인식 결과를 Auto Labeling에 재활용하는 코드로 구성되어 있습니다.
**Detectron2**를 사용하여 버튼 인식 성능을 극대화하며, 버튼 인식 후 부족한 부분을 배치 추정으로 보완하여 매번 학습 비용을 절감하는 데 초점을 맞추고 있습니다.

### Goal
모든 버튼을 사전에 학습시키고 버튼 배치에 대한 데이터베이스를 구축하는 대신, 일반적인 버튼을 인식하고 부족한 부분을 배치 추정으로 채워 나가면서 학습 비용을 절감합니다.

## Key Features
- 일반적인 버튼 인식: 자주 사용되는 버튼 종류를 우선적으로 학습하여 인식 성능을 높입니다.
- Detectron2 활용: Detectron2를 사용하여 버튼 인식을 수행하며, 정확도 높은 인식 결과를 제공합니다.
- 배치 추정을 통한 오류 보정: False Positive, Classification Error, False Negative를 배치 추정 방법으로 보완하여 정확도를 개선합니다.
  - 이상치 제거: IQR method와 DNSCAN을 통해 버튼을 군집화하고 너무 떨어진 버튼은 이상치로 분류 후 제거합니다.
  - 주축 분석: PCA를 통해 버튼의 메인 축을 추정합니다.
  - 인접 층 버튼 간 방향과 거리 계산: 앞서 얻은 메인 축과 서브 축을 기반으로 인접 버튼 간의 순서와 위치 관계를 파악하고 거리를 계산합니다.
  - 공백 메우기 및 클래스 재부여: 거리거 너무 짧거나 먼 경우를 False Positive 혹은 False negative로 추정하고 옳게 버튼 배치를 재생성한 후 최대한 모순이 적어지는 방향으로 버튼 클래스 재부여합니다.
  - 자동 라벨링(Auto Labeling) 재활용: 최종 분류된 결과를 JSON 파일로 저장하여 Auto Labeling에 재활용합니다.
  - 성능 분석: Auto Labeling과 버튼 배치 추정 방법의 성능을 분석하여 학습 및 인식 정확도를 평가합니다.
 
## Performance
Auto Label을 통해 데이터셋을 50자에서 850장으로 증강시키고, 모델의 F1 Score를 72.37%에서 약 97.82%로 향상시켰습니다.

## License
이 프로젝트는 Apache License 2.0에 따라 배포됩니다.
