import torch
from torch.optim import Adam, AdamW, SGD, Adagrad


def get_optimizer(model: torch.nn.Module, args):
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
        
    elif args.optimizer == "adamW":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr, momentum = 0.9)
    elif args.optimizer == "adagrad":
        optimizer = Adagrad(model.parameters(), lr=args.lr, weight_decay=0.01)
    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()

    return optimizer
