import os, yaml
from easydict import EasyDict

import numpy as np
import torch

from dkt import trainer
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, logging_conf


logger = get_logger(logging_conf)


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Preparing data ...")
    preprocess = Preprocess(args=args)
    preprocess.load_test_data(file_name=args.test_file_name)
    test_data: np.ndarray = preprocess.get_test_data()
    
    logger.info("Loading Model ...")
    model, run_name = trainer.load_model(args=args)  #.to(args.device)
    model = model.to(args.device)
    
    logger.info("Make Predictions & Save Submission ...")
    trainer.inference(args=args, test_data=test_data, model=model, run_name=run_name)


if __name__ == "__main__":
    with open('dkt/args.yaml') as file:
        args = EasyDict(yaml.safe_load(file))
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
