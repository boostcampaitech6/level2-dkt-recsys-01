import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn.functional import sigmoid
import wandb

from .criterion import get_criterion
from .dataloader import get_loaders
from .metric import get_metric
from .model import LSTM, LSTMATTN, BERT, Saint
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .utils import get_logger, logging_conf
from .SaintPlus import SaintPlus

logger = get_logger(logger_conf=logging_conf)

def run(args,
        train_data: np.ndarray,
        valid_data: np.ndarray,
        model: nn.Module,
        run_name):
    train_loader, valid_loader = get_loaders(args=args, train=train_data, valid=valid_data)

    # For warmup scheduler which uses step interval
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    optimizer = get_optimizer(model=model, args=args)
    scheduler = get_scheduler(optimizer=optimizer, args=args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):
        logger.info("Start Training: Epoch %s", epoch + 1)

        # TRAIN
        train_auc, train_acc, train_loss = train(train_loader=train_loader,
                                                 model=model, optimizer=optimizer,
                                                 scheduler=scheduler, args=args)

        # VALID
        auc, acc, wandb_cf = validate(valid_loader=valid_loader, model=model, args=args)

        wandb.log(dict(epoch=epoch,
                       train_loss_epoch=train_loss,
                       train_auc_epoch=train_auc,
                       train_acc_epoch=train_acc,
                       valid_auc_epoch=auc,
                       valid_acc_epoch=acc,
                       confusion_matrix=wandb_cf))
        
        if auc > best_auc:
            best_auc = auc
            # nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(state={"epoch": epoch + 1,
                                   "state_dict": model_to_save.state_dict()},
                            model_dir=args.model_dir,
                            model_filename=f"{run_name}_best_model.pt")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                logger.info(
                    "EarlyStopping counter: %s out of %s",
                    early_stopping_counter, args.patience
                )
                break

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)


def train(train_loader: torch.utils.data.DataLoader,
          model: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        preds = model(**batch)
        targets = batch["correct"]
        
        loss = compute_loss(preds=preds, targets=targets)
        update_params(loss=loss, model=model, optimizer=optimizer,
                      scheduler=scheduler, args=args)

        if step % args.log_steps == 0:
            logger.info("Training steps: %s Loss: %.4f", step, loss.item())

        # predictions
        preds = sigmoid(preds[:, -1])
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(targets=total_targets, preds=total_preds)
    loss_avg = sum(losses) / len(losses)
    logger.info("TRAIN AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc, loss_avg


def validate(valid_loader: nn.Module, model: nn.Module, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        preds = model(**batch)
        targets = batch["correct"]

        # predictions
        preds = sigmoid(preds[:, -1])
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(targets=total_targets, preds=total_preds)
    logger.info("VALID AUC : %.4f ACC : %.4f", auc, acc)
    wandb_cf = wandb.plot.confusion_matrix(
            probs=None, y_true=total_targets, preds=np.where(total_preds >= 0.5, 1, 0),
            class_names=['0', '1'])

    return auc, acc, wandb_cf


def inference(args, test_data: np.ndarray, model: nn.Module) -> None:
    model.eval()
    _, test_loader = get_loaders(args=args, train=None, valid=test_data)

    total_preds = []
    for step, batch in enumerate(test_loader):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        preds = model(**batch)

        # predictions
        preds = sigmoid(preds[:, -1])
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)

    write_path = os.path.join(args.output_dir, "submission.csv")
    os.makedirs(name=args.output_dir, exist_ok=True)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))
    logger.info("Successfully saved submission as %s", write_path)


def get_model(args) -> nn.Module:
    model_args = dict(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_tests=args.n_tests,
        n_questions=args.n_questions,
        n_tags=args.n_tags,
        n_heads=args.n_heads,
        drop_out=args.drop_out,
        max_seq_len=args.max_seq_len,
    )
    try:
        model_name = args.model.lower()
        if model_name == 'saint':
            model=Saint(args)
            return model 
        if model_name == "saintplus":
            model=SaintPlus(args)
            return model
        model = {
        "lstm": LSTM,
        "lstmattn": LSTMATTN,
        "bert": BERT,
        }.get(model_name)(**model_args)
    except KeyError:
        logger.warn("No model name %s found", model_name)
    except Exception as e:
        logger.warn("Error while loading %s with args: %s", model_name, model_args)
        raise e
    return model


def compute_loss(preds: torch.Tensor, targets: torch.Tensor):
    """
    loss계산하고 parameter update
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(pred=preds, target=targets.float())

    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss: torch.Tensor,
                  model: nn.Module,
                  optimizer: torch.optim.Optimizer,
                  scheduler: torch.optim.lr_scheduler._LRScheduler,
                  args):
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    if args.scheduler == "linear_warmup":
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state: dict, model_dir: str, model_filename: str) -> None:
    """ Saves checkpoint to a given directory. """
    save_path = os.path.join(model_dir, model_filename)
    logger.info("saving model as %s...", save_path)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(state, save_path)

def find_latest_file(directory_path, prefix):
    # 디렉토리 내의 파일 목록을 가져옵니다.
    files = [f for f in os.listdir(directory_path) if f.startswith(prefix) and os.path.isfile(os.path.join(directory_path, f))]

    # 파일들의 수정 시간을 기준으로 정렬합니다.
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory_path, f)), reverse=True)

    if files:
        # 가장 최신 파일의 이름을 반환합니다.
        return files[0]
    else:
        return None

def load_model(args):
    latest_file_name=find_latest_file(args.model_dir, args.model.lower())
    model_path = os.path.join(args.model_dir, latest_file_name)
    logger.info("Loading Model from: %s", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)
    logger.info("Successfully loaded model state from: %s", model_path)
    return model
