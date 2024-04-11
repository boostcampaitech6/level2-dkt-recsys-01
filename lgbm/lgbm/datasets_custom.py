import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import random
from lgbm import feature
from sklearn.preprocessing import OrdinalEncoder

    
class Preprocess:
    '''
    사용법:
        preprocess = datasets_custom.Preprocess(args)
        data = preprocess.load_data(args.data_dir, args.file_name)
        test_data = preprocess.load_test_data(args.data_dir, args.test_file_name)
    '''
    def __init__(self, args):
        self.args = args
        self.base_aggregation()

        self.train_data = None
        self.vaild_data = None
        self.test_data = None

    def create_base_df(self):
        # train, test df 불러오기
        train_df = self.load_data_from_file(self.args.data_dir, self.args.file_name)
        test_df = self.load_data_from_file(self.args.data_dir, self.args.test_file_name)
        # last data만 남기기
        # remove label
        train_label = train_df.drop_duplicates(subset=['userID'], keep='last')
        test_label = test_df.drop_duplicates(subset=['userID'], keep='last')
        # drop labels
        train_df = train_df.drop(index=train_label.index)
        test_df = test_df.drop(index=test_label.index)
        # concat train, test
        base_df = pd.concat([train_df, test_df], axis=0)
        return base_df

    def base_aggregation(self):
        # df: base_df
        base_df = self.create_base_df()

        # Time
        # to datetime
        base_df['Timestamp'] = pd.to_datetime(base_df['Timestamp'])
        # get consuming time to deal with the problem
        base_df['timediff'] = (base_df['Timestamp']-base_df['Timestamp'].shift(1)).dt.seconds.shift(-1)
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

        # ID 분해
        base_df['test_class'] = base_df['testId'].apply(lambda x: x[2]).astype('category')
        base_df['item_number'] = base_df['assessmentItemID'].apply(lambda x: x[-3:]).astype(int)

        # user agg
        self.base_user_agg = base_df.groupby('userID').agg(
            user_acc=('answerCode','mean'), 
            user_count=('answerCode','count'),
            user_acc_recent3=('answerCode', lambda x: x.tail(3).mean()), # head? tail?
            user_acc_recent5=('answerCode', lambda x: x.tail(5).mean()),
            user_acc_recent10=('answerCode', lambda x: x.tail(10).mean()),
            # last data
            user_previous=('answerCode', lambda x: x.tail(1)),
            # time
            user_time_median=('timediff','median'),
            user_time_1st=('timediff', lambda x: x.quantile(0.25)),
            user_time_3rd=('timediff', lambda x: x.quantile(0.75)),
        )

        # user, item agg
        base_user_item_agg = base_df.groupby(['userID', 'assessmentItemID']).agg(
            ########### 6. 유저별로 이전에 동일한 문제를 풀었던 횟수를 추가
            # 동일한 과제를 수행했으면 다음번엔 맞출 확률이 높을 것
            user_solved_count=('answerCode', 'cumcount'),
        )
        self.base_user_item_agg = pd.concat(
            [base_user_item_agg, base_df[['userID', 'assessmentItemID']]], axis=1).drop_duplicates(
                ['userID', 'assessmentItemID'], keep='last')
        
        # tag agg
        self.base_tag_agg = base_df.groupby('KnowledgeTag').agg(
            tag_acc_mean=('answerCode','mean'),
            tag_time_median=('timediff','median'),
            tag_time_1st=('timediff', lambda x: x.quantile(0.25)),
            tag_time_3rd=('timediff', lambda x: x.quantile(0.75)),
        )
        
        # item agg
        self.base_item_agg = base_df.groupby('assessmentItemID').agg(
            item_acc_mean=('answerCode','mean'),
            item_time_median=('timediff','median'),
            item_time_1st=('timediff', lambda x: x.quantile(0.25)),
            item_time_3rd=('timediff', lambda x: x.quantile(0.75)),
            test_class=('test_class',lambda x: x.head(1)),
            item_number=('item_number',lambda x: x.head(1)),
        )
        
        # test agg
        self.base_test_agg = base_df.groupby('testId').agg(
            test_time_median=('timediff','median'),
            test_time_1st=('timediff', lambda x: x.quantile(0.25)),
            test_time_3rd=('timediff', lambda x: x.quantile(0.75)),
        )

    def feature_engineering(self, df):
        # merge
        df = pd.merge(df, self.base_user_agg, left_on='userID', right_index=True, how='left')
        df = pd.merge(df, self.base_tag_agg, left_on='KnowledgeTag', right_index=True, how='left')
        df = pd.merge(df, self.base_item_agg, left_on='assessmentItemID', right_index=True, how='left')
        df = pd.merge(df, self.base_test_agg, left_on='testId', right_index=True, how='left')
        df = pd.merge(df, self.base_user_item_agg, right_on=['userID', 'assessmentItemID'], 
            left_on=['userID', 'assessmentItemID'], how='left')

        ########### 4. testID를 자른 값을 추가
        # testID값을 분리한 값을 추가
        df['test_group_one'] = df['testId'].apply(lambda x: int(x[1:4]))
        df['test_group_two'] = df['testId'].apply(lambda x: int(x[-3:]))
        ########### 5. itemID에서 순번을 자른 값을 추가
        # 과제의 순번이 영향이 있지 않을까
        df['serial'] = df['assessmentItemID'].apply(lambda x: int(x[-3:]))

        # drop column
        category_columns = ['userID', 'assessmentItemID', 'testId', 'KnowledgeTag', 'test_group_one', 'test_group_two', 
            'serial', 'test_class', 'item_number']
        df[category_columns] = df[category_columns].astype("category")
        df = df.drop(columns=['Timestamp'])

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
        df = df.drop_duplicates(subset=['userID'], keep='last')
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
        test_df = self.feature_engineering(test_df)
        # test_df = self.feature_engineering(test_df, False)

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
