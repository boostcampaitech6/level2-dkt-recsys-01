from lgbm import trainer, datasets
import os, yaml
import numpy as np
import torch
import wandb
import time
from lgbm.utils import get_logger, set_seeds, logging_conf
from easydict import EasyDict


logger = get_logger(logging_conf)

def main(args):
    wandb.login()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Preparing data ...")
    preprocess = datasets.Preprocess(args)
    data = preprocess.load_data(args.data_dir, args.file_name)
    logger.info("Building Model ...")
    logger.info("Start Training ...")
    model = trainer.train_valid(data)
    # 시간 설정
    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    
    # wandb 이름 설정
    model_filename=f"{save_time}_{type(model).__name__}"
    wandb.init(project="dkt", config=vars(args), name=model_filename)

    test_data = preprocess.load_test_data(args.data_dir, args.test_file_name)
    column_list = test_data.columns.tolist()
    # 학습에 사용된 컬럼 정보 출력
    print(column_list)
    trainer.inference(model_filename, test_data, model)


if __name__ == "__main__":
    with open('lgbm/args.yaml') as file:
        args = EasyDict(yaml.safe_load(file))
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)