import os
import yaml
import argparse
from easydict import EasyDict
from datetime import datetime
import torch
import wandb

from lightgcn.datasets import prepare_dataset, prepare_dataset2
from lightgcn import trainer
from lightgcn.utils import get_logger, set_seeds, logging_conf


logger = get_logger(logging_conf)


def main(args: EasyDict):
    
    global wandb_id
    wandb_id = wandb.util.generate_id()
    args.timestamp = datetime.today().strftime("%Y%m%d%H%M")
    args.run_name = f'{args.model}_{args.timestamp}'
    wandb.login()
    wandb.init(id = wandb_id, resume = "allow", project="lightgcn", name= args.run_name, config=vars(args))
    set_seeds(args.seed)
    
    use_cuda: bool = torch.cuda.is_available() and args.use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info("Preparing data ...")
    train_data, valid_data, test_data, n_node = prepare_dataset(device=device, data_dir=args.data_dir)
    train_edges, train_edge_labels, valid_edges, valid_edge_labels = prepare_dataset2(device=device, data_dir=args.data_dir)

    logger.info("Building Model ...")
    model = trainer.build(
        n_node=n_node,
        embedding_dim=args.hidden_dim,
        num_layers=args.n_layers,
        alpha=args.alpha,
    )
    model = model.to(device)
    
    logger.info("Start Training ...")
    # trainer.run(
    #     model=model,
    #     train_data=train_data,
    #     valid_data=valid_data,
    #     args = args
    # )
    
    trainer.run2(
        model=model,
        train_edges=train_edges,
        train_edge_labels=train_edge_labels,
        valid_edges=valid_edges,
        valid_edge_labels =valid_edge_labels,
        args = args
    )
    
    trainer.inference(
        model=model,
        data=test_data,
        args= args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument
    arg('--lr', type=float, default = 0.001)
    arg('--optimizer', type=str, default = "adam")
    arg('--scheduler', type=str, default = "plateau")
    arg('--seed', type = int, default = 42)
    arg('--model', type=str, default = 'lightgcn')
    arg('--use_cuda_if_available', type=bool, default = True)
    arg('--data_dir', type=str, default = '/opt/ml/input/data/')
    arg('--model_dir', type=str, default='models/')
    arg('--model_name', type=str, default='best_model.pt')
    arg('--output_dir', type=str, default='outputs/')
    arg('--hidden_dim', type=int, default = 64)
    arg('--n_layers', type=int, default = 1)
    arg('--alpha', type=float, default = None)
    arg('--n_epochs', type=int, default = 100)
    arg('--run_name', type=str, default = None)
    
    args = parser.parse_args()
    
    if isinstance(args.alpha, str):
        args.alpha = eval(args.alpha)
    if isinstance(args.use_cuda_if_available, str):
        args.use_cuda_if_available = eval(args.use_cuda_if_available)
        
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)
