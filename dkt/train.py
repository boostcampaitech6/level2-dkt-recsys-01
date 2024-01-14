import os

import numpy as np
import torch
import wandb

from dkt import trainer
from dkt.args import parse_args
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, set_seeds, logging_conf


logger = get_logger(logging_conf)


def main(args):
    wandb.login() # wandb 로그인
    set_seeds(args.seed) # random seed set
    args.device = "cuda" if torch.cuda.is_available() else "cpu" # gpu 사용가능하면 gpu 사용하도록 설정
    
    logger.info("Preparing data ...") 
    preprocess = Preprocess(args) # 데이터 전처리 담당 인스턴스 생성
    preprocess.load_train_data(file_name=args.file_name) # 학습 데이터 로드
    train_data: np.ndarray = preprocess.get_train_data() # 학습 데이터 인스턴스 가져옴
    train_data, valid_data = preprocess.split_data(data=train_data) # 훈련, 검증 셋으로 나눔
    wandb.init(project="dkt", config=vars(args)) # wandb 인스턴스 시작
    
    logger.info("Building Model ...")
    model: torch.nn.Module = trainer.get_model(args=args).to(args.device) # 입력 인자에 맞는 모델 로드
    
    logger.info("Start Training ...")
    trainer.run(args=args, train_data=train_data, valid_data=valid_data, model=model) # 모델 학습


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
