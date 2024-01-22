import random, logging, pickle
from collections import defaultdict
from easydict import EasyDict
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import torch

from dataset import SimpleSequenceDKTDataset

def init_logger(runname):
    # create logger
    logger = logging.getLogger(runname)
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger

def seed_everything(seed=42, logger=None):
    if logger: logger.info("set up random seed")
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)

def get_sequence_by_user(df, features, max_length=512):
    user_ids, inputs, masks, targets = [], [], [], []

    for user_id in tqdm(df['userID'].unique()):
        
        # get user data with user_id
        user_data = df[df['userID'] == user_id]
        # get additional info (previous label)
        user_data = user_data.assign(previous_label=(user_data.answerCode.shift(1)+1).fillna(1).values)
        # get rolling mean by window
        user_data['acc_window_3'] = user_data['answerCode'].rolling(window=3).mean().shift(1).fillna(.5)
        user_data['acc_window_5'] = user_data['answerCode'].rolling(window=5).mean().shift(1).fillna(.5)
        user_data['acc_window_10'] = user_data['answerCode'].rolling(window=10).mean().shift(1).fillna(.5)

        # get sequence to numpy
        sequence = user_data[features].to_numpy()
        # get target data: last answerCode
        target = user_data['answerCode'].values[-1]

        # cut or pad sequences with max_length
        if len(sequence) < max_length:
            padding = np.zeros((max_length - len(sequence), sequence.shape[1]))
            mask = np.concatenate((np.zeros(max_length-len(sequence),), np.ones(len(sequence),)))
            sequence = np.vstack((padding, sequence))
        else:
            sequence = sequence[-max_length:]
            mask = np.ones((max_length,))
        
        user_ids.append(user_id)
        inputs.append(sequence)
        masks.append(mask)
        targets.append(target)

    return [np.array(user_ids), np.array(inputs), np.array(masks), np.array(targets)]

def get_data(data_path, configs, features, max_length):
    # read data
    df = pd.read_csv(data_path)
    ########### 4. testID를 자른 값을 추가
    # testID값을 분리한 값을 추가
    df['test_group_one'] = df['testId'].apply(lambda x: int(x[1:4])).astype(int)
    df['test_group_two'] = df['testId'].apply(lambda x: int(x[-3:])).astype(int)
    ########### 5. itemID에서 순번을 자른 값을 추가
    # 과제의 순번이 영향이 있지 않을까
    df['serial'] = df['assessmentItemID'].apply(lambda x: int(x[-3:])).astype(int)

    # additional info
    # to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # get consuming time to deal with the problem
    df['timediff'] = (df['Timestamp']-df['Timestamp'].shift(1)
        ).dt.seconds.shift(-1)
    
    # get the last item of the users
    df['next_userID'] = df.userID.shift(-1)
    last_appearances = df.apply(lambda x: True if x['userID']!=x['next_userID'] else False, axis=1)
    df.loc[last_appearances, 'timediff'] = np.nan

    # get to know last item of the tests
    df['next_testId'] = df.testId.shift(-1)
    last_appearances = df.apply(lambda x: True if x['testId']!=x['next_testId'] else False, axis=1)
    df.loc[last_appearances, 'timediff'] = np.nan

    # 1시간이 넘게 걸리면 문제가 있다고 본다.
    df.loc[df.timediff > 3600, 'timediff'] = np.nan

    threshold_value = 3.
    df['guess_yn'] = df['timediff'].apply(lambda x: 1 if x<=threshold_value else 0).astype('float')
    df['guess_yn'][df['timediff'].isna()] = 0.5

    # log transform
    df.timediff = np.log1p(df.timediff)
    
    ### 시간 정보 가공
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # 요일
    df['dayofweek'] = df['Timestamp'].dt.dayofweek
    # 시간대
    df['time_hour'] = df['Timestamp'].dt.hour//4

    additional_target = [configs.base_user_agg, configs.base_tag_agg, configs.base_item_agg, 
                         configs.base_test_agg, configs.base_g1_agg, configs.base_g2_agg]
    additional_keys = ['userID', 'KnowledgeTag', 'assessmentItemID', 
                       'testId', 'test_group_one', 'test_group_two']
    
    for target, key in zip(additional_target, additional_keys):
        if isinstance(target, type(None)): continue
        df = pd.merge(df, target, left_on=key, right_index=True, how='left')
        
    # ordinal encoding
    df[configs.category_cols] = configs.oe.transform(df[configs.category_cols])

    # time diff
    df['time_diff'] = df.groupby('userID')['Timestamp'].transform(lambda x: x.iloc[-1]-x).dt.days
    
    # scaling
    df[configs.scaling_cols] = configs.scaler.transform(df[configs.scaling_cols])

    # impute
    df[configs.scaling_cols] = configs.imputer.transform(df[configs.scaling_cols])

    # sequence
    return get_sequence_by_user(df, features, max_length)

def train_val_split(user_ids, X, masks, target, train_size=.8):
    # split index and target
    train_index, valid_index, train_y, valid_y = train_test_split(
        range(target.shape[0]), target, train_size=.8, stratify=target)
    # split X
    train_X, valid_X = X[train_index], X[valid_index]
    # split masks
    train_masks, valid_masks = masks[train_index], masks[valid_index]
    # split users
    train_users, valid_users = user_ids[train_index], user_ids[valid_index]

    return (train_users, train_X, train_masks, train_y), (valid_users, valid_X, valid_masks, valid_y)

def load_base_df(args):
    # read data
    train_df = pd.read_csv(args.train_path)#'../../data/train_data.csv')
    test_df = pd.read_csv(args.test_path)#'../../data/test_data.csv')

    # split label
    train_last = train_df.drop_duplicates(subset=['userID'], keep='last')
    test_label = test_df.drop_duplicates(subset=['userID'], keep='last')

    # drop labels
    train_df = train_df.drop(index=train_last.index)
    test_df = test_df.drop(index=test_label.index)
    base_df = pd.concat([train_df, test_df], axis=0)

    return base_df

def feature_engineering(args):
    base_df = load_base_df(args)

    ## category columns
    category_cols = ['assessmentItemID', 'testId', 'KnowledgeTag',
                     'test_group_one', 'test_group_two', 'serial',
                     'dayofweek', 'time_hour'
                     ]
    scaling_cols = ['time_diff', 'timediff']

    base_df['test_group_one'] = base_df['testId'].apply(lambda x: int(x[1:4])).astype(int)
    base_df['test_group_two'] = base_df['testId'].apply(lambda x: int(x[-3:])).astype(int)

    ########### 5. itemID에서 순번을 자른 값을 추가
    base_df['serial'] = base_df['assessmentItemID'].apply(lambda x: int(x[-3:])).astype(int)
    
    ### 시간 정보 가공
    base_df['Timestamp'] = pd.to_datetime(base_df['Timestamp'])
    # 요일
    base_df['dayofweek'] = base_df['Timestamp'].dt.dayofweek
    # 시간대
    base_df['time_hour'] = base_df['Timestamp'].dt.hour//4

    # ordinal encoder
    oe = OrdinalEncoder()
    oe = oe.fit(base_df[category_cols])

    # 문제푼지 얼마나 지났는지
    base_df['time_diff'] = base_df.groupby('userID')['Timestamp'].transform(
        lambda x: x.iloc[-1]-x).dt.days

    # to datetime
    base_df['Timestamp'] = pd.to_datetime(base_df['Timestamp'])
    # get consuming time to deal with the problem
    base_df['timediff'] = (base_df['Timestamp']-base_df['Timestamp'].shift(1)
        ).dt.seconds.shift(-1)
    
    # get the last item of the users
    base_df['next_userID'] = base_df.userID.shift(-1)
    last_appearances = base_df.apply(lambda x: True if x['userID']!=x['next_userID'] else False, axis=1)
    base_df.loc[last_appearances, 'timediff'] = np.nan

    # get to know last item of the tests
    base_df['next_testId'] = base_df.testId.shift(-1)
    last_appearances = base_df.apply(lambda x: True if x['testId']!=x['next_testId'] else False, axis=1)
    base_df.loc[last_appearances, 'timediff'] = np.nan

    # 1시간이 넘게 걸리면 문제가 있다고 본다.
    base_df.loc[base_df.timediff > 3600, 'timediff'] = np.nan

    # # 찍었는지
    # threshold_value = 3.
    # base_df['guess_yn'] = base_df['guess_yn'].apply(lambda x: 1 if x<=threshold_value else 0).astype('float')
    # base_df['guess_yn'][base_df['timediff'].isna()] = 0.5

    # log transform
    base_df.timediff = np.log1p(base_df.timediff)
    
    # standard scaling
    scaler = StandardScaler()
    scaler = scaler.fit(base_df[scaling_cols])
    base_df[scaling_cols] = scaler.transform(base_df[scaling_cols])

    # imputer
    imputer = SimpleImputer(strategy='median')
    imputer = imputer.fit(base_df[scaling_cols])
    base_df[scaling_cols] = imputer.transform(base_df[scaling_cols])

    ## groupby continuous columns
    base_user_agg = base_df.groupby('userID').agg(
        user_acc_mean=('answerCode','mean'), 
        user_time_median=('timediff','median'), 
        user_count=('answerCode','count'),
    )
    base_user_agg['user_count'] = (base_user_agg['user_count']-base_user_agg['user_count'].mean())/base_user_agg['user_count'].max()
    base_user_agg['user_ability'] = base_user_agg.user_acc_mean/(base_user_agg.user_time_median+1)
    
    base_tag_agg = base_df.groupby('KnowledgeTag').agg(
        tag_acc_mean=('answerCode','mean'),
        tag_time_median=('timediff','median'),
    )
    base_tag_agg['tag_difficulty'] = base_tag_agg.tag_acc_mean/(base_tag_agg.tag_time_median+1)

    base_item_agg = base_df.groupby('assessmentItemID').agg(
        item_acc_mean=('answerCode','mean'),
        item_time_median=('timediff','median'),
    )
    base_item_agg['item_difficulty'] = base_item_agg.item_acc_mean/(base_item_agg.item_time_median+1)

    base_test_agg = base_df.groupby('testId').agg(
        test_acc_mean=('answerCode','mean'),
        test_time_median=('timediff','median'),
    )
    base_test_agg['test_difficulty'] = base_test_agg.test_acc_mean/(base_test_agg.test_time_median+1)

    base_g1_agg = base_df.groupby('test_group_one').agg(
        g1_acc_mean=('answerCode','mean'),
        g1_time_median=('timediff','median'),
    )
    base_g1_agg['g1_difficulty'] = base_g1_agg.g1_acc_mean/(base_g1_agg.g1_time_median+1)

    base_g2_agg = base_df.groupby('test_group_two').agg(
        g2_acc_mean=('answerCode','mean'),
        g2_time_median=('timediff','median'),
    )
    base_g2_agg['g2_difficulty'] = base_g2_agg.g2_acc_mean/(base_g2_agg.g2_time_median+1)

    configs = {
        'oe': oe,
        'scaler': scaler,
        'imputer': imputer,
        'category_cols': category_cols,
        'scaling_cols': scaling_cols,
        'base_user_agg': base_user_agg,
        'base_item_agg': base_item_agg,
        'base_tag_agg': base_tag_agg,
        'base_test_agg': base_test_agg,
        'base_g1_agg': base_g1_agg,
        'base_g2_agg': base_g2_agg,

    }
    return EasyDict(configs)

def get_user_item_embeddings(oe): #user_ids, sequence_item_ids, 

    filename = 'svd-32_240117-181412_0.7931_embeddings.pickle'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    user_embedding_df = pd.DataFrame(
        data['user_embeddings'], index=data['user_ids'], 
        columns=[f'user_embed_{i}' for i in range(data['user_embeddings'].shape[1])])

    item_embedding_df = pd.DataFrame(
        data['item_embeddings'], index=data['item_ids'], 
        columns=[f'item_embed_{i}' for i in range(data['item_embeddings'].shape[1])])
    
    user_embeddings = user_embedding_df.loc[sorted(user_embedding_df.index)]
    item_embeddings = item_embedding_df.loc[oe.categories_[0]]

    embeddings = {
        'user_embeddings': user_embeddings.values,
        'item_embeddings': item_embeddings.values,
    }

    return embeddings

def main(args):
    # logger create
    logger = init_logger('test')
    seed_everything(logger=logger)

    configs = feature_engineering(args)
    print([len(feature) for feature in configs.oe.categories_])
    features = [*configs.category_cols, *configs.scaling_cols,
                'previous_label',
                # 'acc_window_3', 'acc_window_5', 'acc_window_10',
                'user_acc_mean', 'tag_acc_mean', 'item_acc_mean', 'test_acc_mean',
                'g1_acc_mean', 'g1_acc_mean',
                'user_time_median', 'tag_time_median', 'item_time_median', 
                'g1_time_median', 'g2_time_median', 
                # 'user_count', 'user_ability', 'tag_difficulty', 
                # 'item_difficulty', 'test_difficulty', 'g1_difficulty', 'g2_difficulty',
                # 'guess_yn',
    ]
    
    # save embeddings
    if args.embed:
        logger.info("save embeddings...")
        embeddings = get_user_item_embeddings(configs.oe)
        with open(f'{args.max_length}-{args.embed}-embeddings.pickle', 'wb') as f:
            pickle.dump(embeddings, f)
    
    # get data
    logger.info("load and preprocess data...")
    train_data = get_data(args.train_path, configs, features, args.max_length)
    test_data = get_data(args.test_path, configs, features, args.max_length)

    # split dataset
    logger.info("split train and validation data...")
    train_data, valid_data = train_val_split(*train_data)

    # create dataset
    logger.info("create datasets...")
    train_dataset = SimpleSequenceDKTDataset(*train_data, args.max_length)
    valid_dataset = SimpleSequenceDKTDataset(*valid_data, args.max_length)
    test_dataset = SimpleSequenceDKTDataset(*test_data, max_length=args.max_length, train=False)
    logger.info(f"train data: {len(train_dataset)}, valid data: {len(valid_dataset)}, test data: {len(test_dataset)}")
    logger.info(f"train data[0]: {train_dataset[0]['user_id']}")
    logger.info(f"train data[0]: {valid_dataset[0]['user_id']}")
    logger.info(f"test data[0]: {test_dataset[0]['user_id']}")

    # save dataset
    logger.info("save datasets...")
    torch.save(train_dataset, f'dataset/train_dataset_v4-{args.max_length}-{args.embed}-{len(features)}.pt')
    torch.save(valid_dataset, f'dataset/valid_dataset_v4-{args.max_length}-{args.embed}-{len(features)}.pt')
    torch.save(test_dataset, f'dataset/test_dataset_v4-{args.max_length}-{args.embed}-{len(features)}.pt')

if __name__ == '__main__':
    import yaml
    with open('dataset-args.yaml') as file:
        args = EasyDict(yaml.safe_load(file))
    main(args)
