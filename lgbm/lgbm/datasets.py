import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import random
from lgbm import feature
from sklearn.preprocessing import OrdinalEncoder

    
class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.vaild_data = None
        self.test_data = None
        self.ordinal_encoder = OrdinalEncoder()
        
        
    def feature_engineering(self, df, train=True):

        #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        df.sort_values(by=['userID','Timestamp'], inplace=True)
        
        #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
        # 유저의 정답 수
        df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
        # 문제 풀이 수
        df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
        # 정답률
        df['user_acc'] = df['user_correct_answer']/df['user_total_answer']

        # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
        # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
        correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
        correct_t.columns = ["test_mean", 'test_sum']
        correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
        correct_k.columns = ["tag_mean", 'tag_sum']

        df = pd.merge(df, correct_t, on=['testId'], how="left")
        df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
        
        # ordinal_encoding
        if train:
            df[['testId', 'assessmentItemID']] = self.ordinal_encoder.fit_transform(df[['testId', 'assessmentItemID']])
        else:
            df[['testId', 'assessmentItemID']] = self.ordinal_encoder.transform(df[['testId', 'assessmentItemID']])
        
        # 범주형 데이터로 변환, 변환하면 학습이 안됨
        # df[['testId', 'assessmentItemID']] = df[['testId', 'assessmentItemID']].astype('category')
        
        # 여기 아래 Feature Engineering 코드 적용
        # ex) df = feature.avg_past_correct(df)
        df = df.drop('Timestamp', axis=1)
        
        return df

    def custom_train_test_split(self, df, ratio=0.7, split=True):

        users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
        random.shuffle(users)

        max_train_data_len = ratio*len(df)
        sum_of_train_data = 0
        user_ids =[]

        for user_id, count in users:
            sum_of_train_data += count
            if max_train_data_len < sum_of_train_data:
                break
            user_ids.append(user_id)

        train = df[df['userID'].isin(user_ids)]
        test = df[df['userID'].isin(user_ids) == False]

        #test데이터셋은 각 유저의 마지막 interaction만 추출
        test = test[test['userID'] != test['userID'].shift(-1)]
        return train, test


    # 파일에서 데이터 불러오기
    def load_data_from_file(self, path, file_name):
        csv_file_path = os.path.join(path, file_name)
        df = pd.read_csv(csv_file_path)
        return df


    # train에 사용할 수 있도록 데이터를 불러오는 함수
    def load_data(self, path, file_name):
        df = self.load_data_from_file(path, file_name)
        df = self.feature_engineering(df)
        data = self.prepare_lgbm_data(df)
        self.train_data = data['train']
        self.vaild_data = data['test']
        return data


    # test에 사용할 수 있도록 데이터를 불러오는 함수
    def load_test_data(self, path, file_name):
        # LOAD TESTDATA
        test_csv_file_path = os.path.join(path, file_name)
        test_df = pd.read_csv(test_csv_file_path)

        # FEATURE ENGINEERING
        test_df = self.feature_engineering(test_df, False)

        # LEAVE LAST INTERACTION ONLY
        test_df = test_df[test_df['userID'] != test_df['userID'].shift(-1)]
        
        # DROP ANSWERCODE
        test = test_df.drop(['answerCode'], axis=1)
        self.test_data = test
        
        return test


    # lgbm 학습에 맞게 데이터를 준비해주는 함수
    def prepare_lgbm_data(self, df):
        train, test = self.custom_train_test_split(df)

        # X, y 값 분리
        y_train = train['answerCode']
        train = train.drop(['answerCode'], axis=1)
        
        y_test = test['answerCode']
        test = test.drop(['answerCode'], axis=1)

        lgb_train = lgb.Dataset(train, y_train)
        lgb_test = lgb.Dataset(test, y_test)
        data = {"lgb_train": lgb_train, 
                "lgb_test": lgb_test, 
                "test": test,
                "y_test": y_test,
                "train": train,
                "y_train": y_train
                }
        return data
