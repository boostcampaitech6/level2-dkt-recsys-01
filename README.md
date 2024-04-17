# 프로젝트 개요
초등학교, 중학교, 고등학교, 대학교와 같은 교육기관에서 우리는 시험을 늘 봐왔습니다. 시험 성적이 높은 과목은 우리가 잘 아는 것을 나타내고 시험 성적이 낮은 과목은 반대로 공부가 더욱 필요함을 나타냅니다. 시험은 우리가 얼마만큼 아는지 평가하는 한 방법입니다.

하지만 시험에는 한계가 있습니다. 우리가 수학 시험에서 점수를 80점 받았다면 우리는 80점을 받은 학생일 뿐입니다. 우리가 돈을 들여 과외를 받지 않는 이상 우리는 우리 개개인에 맞춤화된 피드백을 받기가 어렵고 따라서 무엇을 해야 성적을 올릴 수 있을지 판단하기 어렵습니다. 이럴 때 사용할 수 있는 것이 DKT입니다!

DKT는 Deep Knowledge Tracing의 약자로 우리의 "지식 상태"를 추적하는 딥러닝 방법론입니다.

Level2 Deep Knowledge Tracing 대회에 참여하여 다양한 엔지니어링 및 머신러닝 방법론으로 실험한 결과와 배운 점을 기록했습니다.

## 1. 대회 주제

Deep Knowledge Tracing(DKT)는 시간에 따른 유저의 지식 상태를 모델링하는 교육 AI의 한 분야. 이번 대회는 유저의 지식 상태를 예측하는 것에서 나아가, **마지막에 등장하는 문제를 맞출지 예측하는 태스크.**

## 2. 대회 성능 평가 지표

본 대회는 이진분류 문제이기 때문에 Accuracy 하나만으로 평가를 하기에는 잘못된 평가가 되기 때문에 AUROC(Area Under the ROC curve)와 Accuracy 함께 사용함.

## 3. 데이터

데이터 명세

| 컬럼명 | 설명 |
| --- | --- |
| userID | 사용자의 고유번호 |
| assessmentItemID | 문항의 고유번호 |
| testId | 시험지의 고유번호 |
| answerCode | 사용자가 해당 문항을 맞았는지 여부 |
| Timestamp | 해당문항을 풀기 시작한 시점 |
| KnowledgeTag | 문항 당 하나씩 배정되는 태그 |

# 기술 스택
![](https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white)
![](https://img.shields.io/badge/jupyter-F37626?style=flat-square&logo=Jupyter&logoColor=white)
![](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=black)
![](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=Pandas&logoColor=white)
![](https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=Numpy&logoColor=white)

# 개발 환경
- OS: Linux-5.4.0-99-generic-x86_64-with-glibc2.31
- GPU: Tesla V100-SXM2-32GB * 6
- CPU cores: 8

# 프로젝트 구조
```
.
├── README.md
├── __init__.py
├── dkt
│   ├── README.md
│   ├── dkt
│   │   ├── args.yaml
│   │   ├── attnlstm
│   │   │   └── attnlstm.py
│   │   ├── criterion.py
│   │   ├── dataloader.py
│   │   ├── lastquery
│   │   │   ├── lastquery.py
│   │   │   ├── lastquery_base_model.py
│   │   │   └── lastquery_exp.py
│   │   ├── metric.py
│   │   ├── model.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   ├── sweep.yaml
│   │   ├── trainer.py
│   │   └── utils.py
│   ├── inference.py
│   ├── requirements.txt
│   └── train.py
├── ensemble.py
├── ensembles.py
├── lgbm
│   ├── lgbm
│   │   ├── README.md
│   │   ├── args.yaml
│   │   ├── datasets.py
│   │   ├── datasets_custom.py
│   │   ├── feature.py
│   │   ├── lgbm.py
│   │   ├── trainer.py
│   │   └── utils.py
│   ├── requirements.txt
│   └── train.py
└── lightgcn
    ├── README.md
    ├── inference.py
    ├── lightgcn
    │   ├── args.yaml
    │   ├── datasets.py
    │   ├── optimizer.py
    │   ├── scheduler.py
    │   ├── trainer.py
    │   └── utils.py
    ├── requirements.txt
    ├── sweep.yaml
    └── train.py
```

# 팀원 소개
| 팀원   | 역할 및 담당                      |
|--------|----------------------------------|
| [서동은](https://github.com/) | ML modeling, Hyper parameter tuning, Saint 기반 모델링 |
| [신상우](https://github.com/sangwoonoel) | LastQuery, LightGBM 모델 베이스라인 구축, HPO, LastQuery 기반 모델링 |
| [이주연](https://github.com/twndus) | ML modeling, Ensemble, OOF, CV, LSTM, LastQuery, LGBM 기반 모델링 |
| [이현규](https://github.com/) | Feature Engineering, EDA, Sasrec 기반 모델링 |
| [이현주](https://github.com/uhhyunjoo) | ML modeling, Hyper parameter tuning, LightGCN 기반 모델링 |
| [조성홍](https://github.com/GangBean) | Feature Engineering, LGBM HPO |

