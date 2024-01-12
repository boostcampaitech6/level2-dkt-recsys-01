import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


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
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag", "test_group_one", "test_group_two"]

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
        user_test_timestamp = df[['userID', 'testId', 'Timestamp', 'assessmentItemID']].copy()
        user_test_timestamp['Timestamp'] = pd.to_datetime(user_test_timestamp['Timestamp'])  # Timestamp 열을 datetime 형식으로 변환
        
        user_test_duration = user_test_timestamp.groupby(['testId', 'userID'])['Timestamp']\
            .agg(lambda x: (x.max() - x.min()).total_seconds()).reset_index()
        user_test_duration.columns = ['testId', 'userID', 'duration']
        duration_per_test = user_test_duration.groupby('testId').agg({'duration': lambda x: x.median()})
        df = df.merge(duration_per_test, how='left', on='testId')
        del duration_per_test

        ########### 2. testId 별로 순번에 따라 시험시작시간과 경과시간을 추가
        # 제한시간이 있다면, 현재까지 사용한 시간이 중요하지 않을까
        # user_test_timestamp['startTime'] = user_test_timestamp[['userID', 'testId', 'Timestamp']]\
        #     .groupby(['userID', 'testId'])['Timestamp'].transform(lambda r: r.min())
        
        # user_test_timestamp['elapsedTime'] = (user_test_timestamp['Timestamp'] - user_test_timestamp['startTime']).dt.total_seconds()

        # df = df.merge(user_test_timestamp[['userID', 'assessmentItemID', 'elapsedTime']], how='left', on=['userID', 'assessmentItemID'])

        ########### 3. testId, 일자별로 user를 그룹화한 값을 추가
        # 단체 응시 같은 유형이 있으면, 같은 시험을 비슷한 시간대에 응시하지 않았을까
        # 일단은 일 단위로 자름
        timestamp = pd.to_datetime(df['Timestamp'])
        df['day'] = timestamp.dt.date
        df['user_category'] = df[['userID', 'testId', 'day']].groupby(['day', 'testId']).ngroup()

        ########### 4. testID를 자른 값을 추가
        # testID값을 분리한 값을 추가
        df['test_group_one'] = df['testId'].apply(lambda x: int(x[1:4]))
        df['test_group_two'] = df['testId'].apply(lambda x: int(x[-3:]))

        return df

    def load_data_from_file(self, file_name: str, is_train: bool = True) -> np.ndarray:
        csv_file_path = os.path.join(self.args.data_dir, file_name) # 파일 path 설정
        df = pd.read_csv(csv_file_path)  # , nrows=100000) # 파일 읽어옴
        df = self.__feature_engineering(df) # FE 처리
        df = self.__preprocessing(df, is_train) # 데이터 전처리

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

        df = df.sort_values(by=["userID", "Timestamp"], axis=0) # 유저별로 문제 풀기 시작한 시간순으로 정렬
        columns = ["userID", "assessmentItemID", "testId", "answerCode"]
        one_hot_cats = ["KnowledgeTag"]
        additional_cols = ["duration", "user_category", "test_group_one", "test_group_two"]

        ####### 1. 테스트별 제한 시간 feature 추가
        duration_per_test = dict(zip(df['testId'], df['duration'])) # testID별 duration dict

        group = (
            df[columns + one_hot_cats + additional_cols] # 사용할 columns 의 series들만 dataFrame으로 가져옴
            .groupby("userID") # 현재 풀어야 하는 문제가 순차적인 문제풀이를 했을때, 다음 문제를 맞출 수 있는지를 판단하는 문제이므로, 한 학생이 순차적으로 푼 문제를 묶어서 학습을 시켜야 함. 이를 위해 userID 로 묶어서 값을 사용.
            .apply(
                lambda r: ( # 각 row에 대해서 아래 열들을 묶어 데이터 프레임으로 반환
                    r["testId"].values, # 테스트 ID 열:
                    r["assessmentItemID"].values, #  문항 ID 열
                    r["KnowledgeTag"].values, # 태그 열
                    r["testId"].map(duration_per_test).values, # 테스트 제한시간 열
                    # r["elapsedTime"].values, # 경과시간 열
                    r["user_category"].values, # 유저 카테고리 열
                    r["test_group_one"].values,
                    r["test_group_two"].values,
                    r["answerCode"].values, # target 열
                )
            )
        )

        return group.to_numpy()

    def load_train_data(self, file_name: str) -> None: # 훈련 데이터 로드
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name: str) -> None: # 테스트 데이터 로드
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset): # Sequence 형태로 처리하는 DKT모델에서 사용할 데이터 셋
    def __init__(self, data: np.ndarray, args):
        self.data = data # 대상 데이터
        self.max_seq_len = args.max_seq_len # 최대 시퀀스 길이

    '''
    데이터 로더에서 로드해오는 단위 Data
    여기서 index는 user단위를 나타냄
    사용할 feature별로 (seq_len, feature_dim) shape의 dict로 반환함
    '''
    def __getitem__(self, index: int) -> dict: # 데이터 조회
        row = self.data[index]
        
        # Load from data
        test, question, tag, duration, userCategory, testGroupOne, testGroupTwo, correct = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
        # print(type(duration), duration)
        data = {
            "test": torch.tensor(test + 1, dtype=torch.int), # unknown 때문에 +1 하는 듯?
            "question": torch.tensor(question + 1, dtype=torch.int),
            "tag": torch.tensor(tag + 1, dtype=torch.int),
            "correct": torch.tensor(correct, dtype=torch.int),
            "duration": torch.tensor(duration, dtype=torch.float),
            # "elapsedTime": torch.tensor(elapsedTime, dtype=torch.float),
            # "user_category": torch.tensor(userCategory, dtype=torch.int),
            "test_group_one": torch.tensor(testGroupOne + 1, dtype=torch.int),
            "test_group_two": torch.tensor(testGroupTwo + 1, dtype=torch.int),
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
        interaction_mask = data["mask"].roll(shifts=1) # 마스크도 한칸씩 옆으로 옮김
        interaction_mask[0] = 0 # 없음을 나타냄
        interaction = (interaction * interaction_mask).to(torch.int64) # interaction의 길이 보정
        data["interaction"] = interaction
        data = {k: v.int() for k, v in data.items()}
        return data

    def __len__(self) -> int:
        return len(self.data)

''' 
DKT 데이터 셋을 로드하는 훈련용로더와 검증용로더를 튜플형태로 반환
'''
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
