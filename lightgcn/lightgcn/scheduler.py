import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR, CosineAnnealingLR

def get_scheduler(optimizer: torch.optim.Optimizer, args):
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, mode="min", verbose=True
        )
    elif args.scheduler == "lambda":
        scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=200, gamma=0.5)
    elif args.scheduler == "cosine_annealing":
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
    return scheduler
