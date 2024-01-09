import os
import random
import time
from datetime import datetime
from typing import Tuple
import joblib

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self,
                   data: np.ndarray,
                   ratio: float = 0.7,
                   shuffle: bool = True,
                   seed: int = 0) -> Tuple[np.ndarray]:
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]
        return data_1, data_2

    def __save_labels(self, encoder: LabelEncoder, name: str) -> None:
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]
        # one_hot_cols = ["KnowledgeTag"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        # for col in one_hot_cols:
        #     encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        #     if is_train:
        #         # For UNKNOWN class
        #         a = df[col].unique().reshape(-1, 1)
        #         encoder.fit(a)
        #         # self.__save_labels(encoder, col)
        #     else:
        #         label_path = os.path.join(self.args.asset_dir, col + "_encoder.joblib")
        #         encoder = joblib.load(label_path)

        #         df[col] = df[col].apply(
        #             lambda x: x if str(x) in encoder.categories_[0] else "unknown"
        #         )

        #     # 모든 컬럼이 범주형이라고 가정
        #     df[col] = df[col].astype(str)
        #     encoded_values = encoder.transform(df[col].values.reshape(-1, 1))
        #     df = pd.concat([df, pd.DataFrame(encoded_values, columns=[f"{col}_{int(i)}" for i in range(encoded_values.shape[1])])], axis=1)
        #     df = df.drop(col, axis=1)

        def convert_time(s: str):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)
        return df

    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Fill in if needed

        ############ 1. testId 별로 중간값을 테스트 시간으로 추가해줌
        # 테스트에 제한시간이 있진 않을까 싶어 테스트별 유저의 풀이시간을 기준으로 중간값 채택함
        user_test_timestamp = df[['userID', 'testId', 'Timestamp']].copy()
        user_test_timestamp['Timestamp'] = pd.to_datetime(user_test_timestamp['Timestamp'])  # Timestamp 열을 datetime 형식으로 변환
        
        user_test_duration = user_test_timestamp.groupby(['testId', 'userID'])['Timestamp']\
            .agg(lambda x: (x.max() - x.min()).total_seconds()).reset_index()
        user_test_duration.columns = ['testId', 'userID', 'duration']
        duration_per_test = user_test_duration.groupby('testId').agg({'duration': lambda x: x.median()})
        df = df.merge(duration_per_test, how='left', on='testId')

        ########### 2. testId 별로 순번에 따라 시험시작시간과 경과시간을 추가
        # 제한시간이 있다면, 현재까지 사용한 시간도 중요함

        return df

    def load_data_from_file(self, file_name: str, is_train: bool = True) -> np.ndarray:
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_tests = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tags = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["userID", "assessmentItemID", "testId", "answerCode"]
        one_hot_cats = ["KnowledgeTag"]
        additional_cols = ["duration"]
        # one_hot_columns = [col for col in df.columns if any(cat in col for cat in one_hot_cats)]

        duration_per_test = dict(zip(df['testId'], df['duration']))
        group = (
            df[columns + one_hot_cats + additional_cols]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["testId"].map(duration_per_test).values,
                    r["answerCode"].values,
                )
            )
        )

        return group.values

    def load_train_data(self, file_name: str) -> None:
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name: str) -> None:
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, args):
        self.data = data
        self.max_seq_len = args.max_seq_len

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]
        
        # Load from data
        test, question, tag, duration, correct = row[0], row[1], row[2], row[3], row[4]
        # print(type(duration), duration)
        data = {
            "test": torch.tensor(test + 1, dtype=torch.int), # unknown 때문에 +1 하는 듯?
            "question": torch.tensor(question + 1, dtype=torch.int),
            "tag": torch.tensor(tag + 1, dtype=torch.int),
            "correct": torch.tensor(correct, dtype=torch.int),
            "duration": torch.tensor(duration, dtype=torch.float),
        }

        # Generate mask: max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        seq_len = len(row[0])
        if seq_len > self.max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len:]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for k, seq in data.items():
                # Pre-padding non-valid sequences
                tmp = torch.zeros(self.max_seq_len) # 최대 길이 0으로 초기화
                tmp[self.max_seq_len-seq_len:] = data[k] # 
                data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1 # pre-padding 이므로 뒷 부분이 1임
        data["mask"] = mask
        
        # Generate interaction: 이전 문제를 맞췄었는지 여부를 나타냄
        interaction = data["correct"] + 1  # 패딩을 위해 correct값에 1을 더해준다.
        interaction = interaction.roll(shifts=1) # 한칸씩 옆으로 옮김
        interaction_mask = data["mask"].roll(shifts=1) # 한칸씩 옆으로 옮김
        interaction_mask[0] = 0 # 없음을 나타냄
        interaction = (interaction * interaction_mask).to(torch.int64) # interaction의 길이 보정
        data["interaction"] = interaction
        data = {k: v.int() for k, v in data.items()}
        return data

    def __len__(self) -> int:
        return len(self.data)


def get_loaders(args, train: np.ndarray, valid: np.ndarray) -> Tuple[torch.utils.data.DataLoader]:
    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader
