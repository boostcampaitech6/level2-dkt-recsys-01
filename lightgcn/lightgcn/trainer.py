import os
import math
import wandb
import numpy as np
import pandas as pd
import torch
from torch import nn
from copy import deepcopy

from easydict import EasyDict
from torch_geometric.nn.models import LightGCN
from sklearn.metrics import accuracy_score, roc_auc_score
from .utils import get_logger, logging_conf
from .scheduler import get_scheduler
from .optimizer import get_optimizer

logger = get_logger(logger_conf=logging_conf)

def build(n_node: int, weight: str = None, **kwargs):
    model = LightGCN(num_nodes=n_node, **kwargs)
    if weight:
        if not os.path.isfile(path=weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(f=weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        return model


def run(
    model: nn.Module,
    train_data: dict,
    valid_data: dict = None,
    args: EasyDict = None
):
    model.train()
    optimizer = get_optimizer(model=model, args=args)
    scheduler = get_scheduler(optimizer=optimizer, args=args)
    os.makedirs(name=f'{args.model_dir}{args.run_name}/', exist_ok=True)

    if valid_data is None:
        print("VALID DATA IS NONE")
        breakpoint()
        eids = np.arange(len(train_data["label"]))
        eids = np.random.permutation(eids)[:1000]
        edge, label = train_data["edge"], train_data["label"]
        label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge[:, eids], label=label[eids])

    logger.info(f"Training Started")
    logger.info(f" * n_epochs    : {args.n_epochs}")
    logger.info(f" * lr          : {args.lr}")
    logger.info(f" * optimizer   : {args.optimizer}")
    logger.info(f" * scheduler   : {args.scheduler}")
    best_auc, best_epoch = 0, -1
    
    for e in range(args.n_epochs):
        logger.info("Epoch: %s", e)
        # TRAIN
        train_acc, train_auc, train_loss, wandb_train_cf = train(train_data=train_data, model=model, optimizer=optimizer, scheduler = scheduler, args = args)
    
        # VALID
        valid_acc, valid_auc, valid_loss, wandb_valid_cf = validate(valid_data=valid_data, model=model)
        
        wandb.log(dict(train_loss = train_loss,
                       valid_loss = valid_loss,
                       train_acc = train_acc,
                       train_auc = train_auc,
                       valid_acc = valid_acc,
                       valid_auc =valid_auc,
                       train_confusion_matrix = wandb_train_cf,
                       valid_confusion_matrix = wandb_valid_cf))

        if valid_auc > best_auc:
            logger.info("Best model updated AUC from %.4f to %.4f", best_auc, valid_auc)
            best_auc, best_epoch = valid_auc, e
            torch.save(obj= {"model": model.state_dict(), "epoch": e + 1}, f=os.path.join(f'{args.model_dir}{args.run_name}/', f"{args.run_name}_best_model.pt"))
        # 1번
        if args.scheduler == "plateau" and not isinstance(valid_data, type(None)):
            scheduler.step(best_auc)
        else:
            scheduler.step()
            
    torch.save(obj={"model": model.state_dict(), "epoch": e + 1}, f=os.path.join(f'{args.model_dir}{args.run_name}/', f"{args.run_name}_last_model.pt"))
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")
    
def run2(
    model: nn.Module,
    train_edges,
    train_edge_labels,
    valid_edges,
    valid_edge_labels,
    args: EasyDict
):
    model.train()
    optimizer = get_optimizer(model=model, args=args)
    scheduler = get_scheduler(optimizer=optimizer, args=args)
    
    patience = 5
    least_loss, num = 1e+9, 0
    os.makedirs(name=f'{args.model_dir}{args.run_name}/', exist_ok=True)

    for e in range(args.n_epochs):
        # train
        train_loss, train_auroc, train_accuracy = train2(optimizer, model, train_edges, train_edge_labels)
        # valid
        valid_loss, valid_auroc, valid_accuracy = validate2(optimizer, model, valid_edges, valid_edge_labels)
        #print(f"epochs {e}: train loss: {train_loss}, train auroc: {train_auroc:.4f}, train accuracy: {train_accuracy:.4f}")
        #print(f"            valid loss: {valid_loss}, valid auroc: {valid_auroc:.4f}, valid accuracy: {valid_accuracy:.4f}")
        
        if valid_loss < least_loss:
            print(f'minimum valid loss is {valid_loss:.4f} at {e} epoch')
            least_loss, num = valid_loss, 0
            best_auc, best_epochs = valid_auroc, e
            best_model = deepcopy(model.state_dict())
        else:
            num += 1
            if num >= patience:
                print(f'early stopped at {e} epoch')
                break
        wandb.log(dict(train_loss = train_loss,
                       valid_loss = valid_loss,
                       train_acc = train_accuracy,
                       train_auc = train_auroc,
                       valid_acc = valid_accuracy,
                       valid_auc =valid_auroc))

        if valid_auroc > best_auc:
            logger.info("Best model updated AUC from %.4f to %.4f", best_auc, valid_auroc)
            best_auc, best_epoch = valid_auroc, e
            torch.save(obj= {"model": model.state_dict(), "epoch": e + 1}, f=os.path.join(f'{args.model_dir}{args.run_name}/', f"{args.run_name}_best_model.pt"))
        # 1번
        if args.scheduler == "plateau":
            scheduler.step(best_auc)
        else:
            scheduler.step()

def train(model: nn.Module, train_data: dict, optimizer: torch.optim.Optimizer,  scheduler:torch.optim.lr_scheduler._LRScheduler, args: EasyDict):
    train_pred = model(train_data["edge"])
    train_loss = model.link_pred_loss(pred=train_pred, edge_label=train_data["label"])
    
    train_prob = model.predict_link(edge_index=train_data["edge"], prob=True)
    train_prob = train_prob.detach().cpu().numpy()

    train_label = train_data["label"].cpu().numpy()
    train_acc = accuracy_score(y_true=train_label, y_pred=train_prob > 0.5)
    train_auc = roc_auc_score(y_true=train_label, y_score=train_prob)

    # backward
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    if args.scheduler == "plateau":
        scheduler.step(train_auc)
    else:
        scheduler.step()
        
    logger.info("TRAIN ACC : %.4f AUC : %.4f LOSS : %.4f", train_acc, train_auc, train_loss.item())
    wandb_train_cf = wandb.plot.confusion_matrix(
            probs=None, y_true=train_label, preds=train_prob > 0.5,
            class_names=['0', '1'])
    return train_acc, train_auc, train_loss, wandb_train_cf


def validate(valid_data: dict, model: nn.Module):
    #breakpoint()
    with torch.no_grad():
        valid_pred = model(valid_data["edge"])
        valid_loss = model.link_pred_loss(pred=valid_pred, edge_label=valid_data["label"])

        valid_prob = model.predict_link(edge_index=valid_data["edge"], prob=True)
        valid_prob = valid_prob.detach().cpu().numpy()
        
        valid_label = valid_data["label"].detach().cpu().numpy()
        valid_acc = accuracy_score(y_true=valid_label, y_pred=valid_prob > 0.5)
        valid_auc = roc_auc_score(y_true=valid_label, y_score=valid_prob)
    
    logger.info("VALID ACC : %.4f AUC : %.4f LOSS : %.4f", valid_acc, valid_auc, valid_loss.item())
    wandb_valid_cf = wandb.plot.confusion_matrix(
            probs=None, y_true=valid_label, preds=valid_prob > 0.5,
            class_names=['0', '1'])

    return valid_acc, valid_auc, valid_loss, wandb_valid_cf

def train2(optimizer, model, edge_index, edge_labels):
    model.train()

    preds = model.predict_link(edge_index, prob=True)
    
    optimizer.zero_grad()
    loss = model.link_pred_loss(preds, edge_labels)
    loss.backward()
    optimizer.step()

    preds_ = preds.detach().cpu().numpy()
    labels = edge_labels.detach().cpu().numpy()
    
    auroc = roc_auc_score(labels, preds_)
    accuracy = accuracy_score(labels, preds_ > 0.5)
    logger.info("TRAIN ACC : %.4f AUC : %.4f LOSS : %.4f", accuracy, auroc, loss.item())
    return loss, auroc, accuracy

def validate2(optimizer, model, edge_index, edge_labels):
    model.eval()

    preds = model.predict_link(edge_index, prob=True)
    loss = model.link_pred_loss(preds, edge_labels)

    preds_ = preds.detach().cpu().numpy()
    labels = edge_labels.detach().cpu().numpy()

    auroc = roc_auc_score(labels, preds_)
    accuracy = accuracy_score(labels, preds_>0.5)
    logger.info("VALID ACC : %.4f AUC : %.4f LOSS : %.4f", accuracy, auroc, loss.item())
    return loss, auroc, accuracy

def inference(model: nn.Module, data: dict, args : EasyDict):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(edge_index=data["edge"], prob=True)
        
    logger.info("Saving Result ...")
    pred = pred.detach().cpu().numpy()
    os.makedirs(name=args.output_dir, exist_ok=True)
    last_write_path = os.path.join(args.output_dir, f"{args.run_name}_last_submission.csv")
    pd.DataFrame({"prediction": pred}).to_csv(path_or_buf=last_write_path, index_label="id")
    logger.info("Successfully saved submission as %s", last_write_path)
  
    best_model = torch.load(os.path.join(f'{args.model_dir}{args.run_name}/', f"{args.run_name}_best_model.pt"))
    best_write_path = os.path.join(args.output_dir, f"{args.run_name}_best_submission.csv")
    pd.DataFrame({"prediction": pred}).to_csv(path_or_buf=best_write_path, index_label="id")
    
    logger.info("Successfully saved submission as %s", last_write_path)
    logger.info("Successfully saved submission as %s", best_write_path)
