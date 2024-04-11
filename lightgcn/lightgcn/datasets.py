import os
from typing import Tuple
import numpy as np
import pandas as pd
import torch

from lightgcn.utils import get_logger, logging_conf
from sklearn.model_selection import train_test_split


logger = get_logger(logging_conf)


def prepare_dataset(device: str, data_dir: str) -> Tuple[dict, dict, int]:
    data = load_data(data_dir=data_dir)
    train_data, valid_data, test_data = separate_data(data=data)
    id2index: dict = indexing_data(data=data)
    train_data_proc = process_data(data=train_data, id2index=id2index, device=device)
    valid_data_proc = process_data(data=valid_data, id2index=id2index, device=device)
    test_data_proc = process_data(data=test_data, id2index=id2index, device=device)

    print_data_stat(train_data, "Train")
    print_data_stat(valid_data, "Valid")
    print_data_stat(test_data, "Test")

    return train_data_proc, valid_data_proc, test_data_proc, len(id2index)

def prepare_dataset2(device: str, data_dir: str) -> Tuple[dict, dict, int]:
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    submission_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    
    train_last = train_df.drop_duplicates(['userID'], keep='last')
    test_last = test_df.drop_duplicates(['userID'], keep='last')

    # train_df2 = train_df.drop(index=train_last.index).drop_duplicates(['userID', 'assessmentItemID'], keep='last')
    train_df2 = train_df.drop_duplicates(['userID', 'assessmentItemID'], keep='last')
    test_df2 = test_df.drop(index=test_last.index).drop_duplicates(['userID', 'assessmentItemID'], keep='last')

    base_df = pd.concat([train_df2, test_df2], axis=0)
    #base_df.shape
    user_ids = np.concatenate([train_df.userID.unique(), test_df.userID.unique()])
    item_ids = train_df.assessmentItemID.unique()
    user_id2index = {uid:i for i, uid in enumerate(user_ids)}
    item_id2index = {iid:i+len(user_ids) for i, iid in enumerate(item_ids)}
    
    base_df['userID'] = base_df['userID'].map(lambda x: user_id2index.get(x,x))
    base_df['assessmentItemID'] = base_df['assessmentItemID'].map(lambda x: item_id2index.get(x,x))
    train_last['userID'] = train_last['userID'].map(lambda x: user_id2index.get(x,x))
    train_last['assessmentItemID'] = train_last['assessmentItemID'].map(lambda x: item_id2index.get(x,x))

    test_last['userID'] = test_last['userID'].map(lambda x: user_id2index.get(x,x))
    test_last['assessmentItemID'] = test_last['assessmentItemID'].map(lambda x: item_id2index.get(x,x))
    total_edges = base_df[['userID','assessmentItemID']].values
    total_edge_labels = base_df[['answerCode']].values
    train_last_edges = train_last[['userID','assessmentItemID']].values
    train_last_edge_labels = train_last[['answerCode']].values

    test_last_edges = test_last[['userID','assessmentItemID']].values
    test_last_edge_labels = test_last[['answerCode']].values
    shuffle_index = list(range(len(total_edges))) 
    np.random.shuffle(shuffle_index)

    total_edges = total_edges[shuffle_index]
    total_edge_labels = total_edge_labels[shuffle_index]
    train_edges, valid_edges, train_edge_labels, valid_edge_labels = \
    train_test_split(total_edges, total_edge_labels, 
                     test_size=.15, shuffle=True, stratify=total_edge_labels)
    
    train_edges = torch.tensor(train_edges).T.to(device)
    valid_edges = torch.tensor(valid_edges).T.to(device)
    
    train_edge_labels = torch.tensor(train_edge_labels).squeeze(-1).to(device)
    valid_edge_labels = torch.tensor(valid_edge_labels).squeeze(-1).to(device)

    return train_edges, train_edge_labels, valid_edges, valid_edge_labels

def load_data(data_dir: str) -> pd.DataFrame: 
    path1 = os.path.join(data_dir, "train_data.csv")
    path2 = os.path.join(data_dir, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)

    data = pd.concat([data1, data2])
    data.drop_duplicates(subset=["userID", "assessmentItemID"], keep="last", inplace=True)
    return data


def separate_data(data: pd.DataFrame) -> Tuple[pd.DataFrame]:
  
    train_num = int(len(data)*0.8)
    data = data.sample(frac=1).reset_index(drop=True)
    train_data = data[data.answerCode >= 0].iloc[:train_num, :]
    valid_data = data[data.answerCode >= 0].iloc[train_num:, :]
    test_data = data[data.answerCode < 0]
    return train_data, valid_data, test_data


def indexing_data(data: pd.DataFrame) -> dict:
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid2index = {v: i for i, v in enumerate(userid)}
    itemid2index = {v: i + n_user for i, v in enumerate(itemid)}
    id2index = dict(userid2index, **itemid2index)
    return id2index


def process_data(data: pd.DataFrame, id2index: dict, device: str) -> dict:
    edge, label = [], []
    for user, item, acode in zip(data.userID, data.assessmentItemID, data.answerCode):
        uid, iid = id2index[user], id2index[item]
        edge.append([uid, iid])
        label.append(acode)

    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)
    return dict(edge=edge.to(device),
                label=label.to(device))


def print_data_stat(data: pd.DataFrame, name: str) -> None:
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
