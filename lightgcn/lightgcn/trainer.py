import os
import wandb
import numpy as np
import pandas as pd
import torch
from torch import nn
from easydict import EasyDict
from torch_geometric.nn.models import LightGCN
from sklearn.metrics import accuracy_score, roc_auc_score
from .utils import get_logger, logging_conf

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
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    model_dir: str = None,
    run_name: str = None,
):
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    os.makedirs(name=f'{model_dir}{run_name}/', exist_ok=True)

    if valid_data is None:
        eids = np.arange(len(train_data["label"]))
        eids = np.random.permutation(eids)[:1000]
        edge, label = train_data["edge"], train_data["label"]
        label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge[:, eids], label=label[eids])

    
    logger.info(f"Training Started : n_epochs={n_epochs}")
    best_auc, best_epoch = 0, -1
    for e in range(n_epochs):
        logger.info("Epoch: %s", e)
        # TRAIN
        train_acc, train_auc, train_loss, wandb_train_cf = train(train_data=train_data, model=model, optimizer=optimizer)
    
        # VALID
        valid_acc, valid_auc, valid_loss, wandb_valid_cf = validate(valid_data=valid_data, model=model)
        
        wandb.log(dict(train_loss=train_loss,
                       valid_loss = valid_loss,
                       train_acc=train_acc,
                       train_auc=train_auc,
                       valid_acc=valid_acc,
                       valid_auc=valid_auc,
                       train_confusion_matrix = wandb_train_cf,
                       valid_confusion_matrix = wandb_valid_cf))

        if valid_auc > best_auc:
            logger.info("Best model updated AUC from %.4f to %.4f", best_auc, valid_auc)
            best_auc, best_epoch = valid_auc, e
            torch.save(obj= {"model": model.state_dict(), "epoch": e + 1}, f=os.path.join(f'{model_dir}{run_name}/', f"{run_name}_best_model.pt"))
    torch.save(obj={"model": model.state_dict(), "epoch": e + 1}, f=os.path.join(f'{model_dir}{run_name}/', f"{run_name}_last_model.pt"))
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")
    

def train(model: nn.Module, train_data: dict, optimizer: torch.optim.Optimizer):
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
    
    logger.info("TRAIN ACC : %.4f AUC : %.4f LOSS : %.4f", train_acc, train_auc, train_loss.item())
    wandb_train_cf = wandb.plot.confusion_matrix(
            probs=None, y_true=train_label, preds=train_prob > 0.5,
            class_names=['0', '1'])
    return train_acc, train_auc, train_loss, wandb_train_cf


def validate(valid_data: dict, model: nn.Module):
    with torch.no_grad():
        valid_pred = model(valid_data["edge"])
        valid_loss = model.link_pred_loss(pred=valid_pred, edge_label=torch.from_numpy(valid_data["label"]).view(-1).to("cuda"))
        
        valid_prob = model.predict_link(edge_index=valid_data["edge"], prob=True)
        valid_prob = valid_prob.detach().cpu().numpy()
        
        valid_label = valid_data["label"]
        valid_acc = accuracy_score(y_true=valid_label, y_pred=valid_prob > 0.5)
        valid_auc = roc_auc_score(y_true=valid_label, y_score=valid_prob)
    
    logger.info("VALID ACC : %.4f AUC : %.4f LOSS : %.4f", valid_acc, valid_auc, valid_loss.item())
    wandb_valid_cf = wandb.plot.confusion_matrix(
            probs=None, y_true=valid_label, preds=valid_prob > 0.5,
            class_names=['0', '1'])

    return valid_acc, valid_auc, valid_loss, wandb_valid_cf


def inference(model: nn.Module, data: dict, output_dir: str, model_dir: str, run_name: str):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(edge_index=data["edge"], prob=True)
        
    logger.info("Saving Result ...")
    pred = pred.detach().cpu().numpy()
    os.makedirs(name=output_dir, exist_ok=True)
    last_write_path = os.path.join(output_dir, f"{run_name}_last_submission.csv")
    pd.DataFrame({"prediction": pred}).to_csv(path_or_buf=last_write_path, index_label="id")
    
    best_model = model.load(os.path.join(f'{model_dir}{run_name}/', f"{run_name}_best_model.pt"))
    best_write_path = os.path.join(output_dir, f"{run_name}_best_submission.csv")
    pd.DataFrame({"prediction": pred}).to_csv(path_or_buf=best_write_path, index_label="id")
    
    logger.info("Successfully saved submission as %s", last_write_path)
    logger.info("Successfully saved submission as %s", best_write_path)
