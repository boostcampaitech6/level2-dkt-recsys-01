# Baseline3: LGBM

## Setup
```bash
cd /data/ephemeral/lgbm
conda init
(base) . ~/.bashrc
(base) conda create -n lgbm python=3.10 -y
(base) conda activate lgbm
(lgbm) pip install -r requirements.txt
(lgbm) python train.py
(lgbm) python inference.py
```

## Files
`lgbm`
* `train.py`: 학습코드입니다. 학습과 동시에 해당 결과로 추론을 진행하고 `submissions.csv` 파일까지 만드는 소스코드입니다. 
* `requirements.txt`: 모델 학습에 필요한 라이브러리들이 정리되어 있습니다.

`lgbm/lgbm`
* `args.yaml`: 학습에 활용되는 여러 argument들을 작성해 놓은 파일입니다.
* `datasets.py`: lgbm 학습에 필요한 데이터를 불러오고, 처리하는 코드가 있는 파일입니다. 
* `trainer.py`: train, valid, inference에 사용되는 함수들을 포함합니다.
* `utils.py`: 학습에 필요한 부수적인 함수들을 포함합니다.
