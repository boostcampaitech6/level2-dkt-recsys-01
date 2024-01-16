import os, yaml
from easydict import EasyDict

import numpy as np
import torch
import wandb
import datetime
from dkt import trainer
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, set_seeds, logging_conf


logger = get_logger(logging_conf)


def main(args):
    wandb.login()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)
    train_data: np.ndarray = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(data=train_data)

    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S') # 현재시간
    
    logger.info("Building Model ...")
    model: torch.nn.Module = trainer.get_model(args=args).to(args.device)

    run_name = f'{type(model).__name__.lower()}-{now}' # run_name 만들기
    wandb.init(project="dkt", config=vars(args),name=run_name)

    logger.info("Start Training ...")
    trainer.run(args=args, train_data=train_data, valid_data=valid_data, model=model,run_name=run_name) #run_name 추가


if __name__ == "__main__":
    with open('dkt/args.yaml') as file:
        args = EasyDict(yaml.safe_load(file))
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
