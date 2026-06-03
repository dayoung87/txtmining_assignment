# 2026-1 텍스트마이닝 과제

BERT, RoBERTa, DistilBERT 파인튜닝 후 성능 비교 위한 코드 작성

## 생성 결과값 설명

* models/{모델명}-{에포크수}epochs-{훈련 데이터셋 크기}
학습시킨 모델 저장

* outputs/epoch_metrics_{날짜시간}

에포크 별 평가 지수 및 최종 평가 지수 txt 파일로 저장

## 파일 설명
* requirement.txt

Python 패키지 리스트. 

터미널에서`pip install -r requirement.txt` 명령어 이용하여 설치 가능

* config.py

파인튜닝에 필요한 변수 지정. 

모델 종류, 데이터셋 샘플링 크기, 시드, 배치 크기, 에포크 수 등 지정 가능

* test.py

코드 수행 환경 확인 위한 코드.

CPU 환경인지 GPU 환경인지 확인 가능

* train.py

모델 학습 위한 코드.

* inference.py:

실제 텍스트에 적용 테스트 위한 코드.

## 사용 데이터셋 
* 허깅페이스 kptimes 데이터셋 이용

https://huggingface.co/datasets/midas/kptimes
