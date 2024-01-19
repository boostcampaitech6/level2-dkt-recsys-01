import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
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
        cate_cols = [
            "assessmentItemID",
            "testId",
            "KnowledgeTag",
            "test_group_one",
            "test_group_two",
            "tag_group_one",
            "tag_group_two",
            "guess_yn",
            ]

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
        df["startTime"] = df["startTime"].apply(convert_time)
        return df

    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Fill in if needed        

        ############ 0. sort by userID, Timestamp
        df = df.sort_values(by=["userID", "Timestamp"], axis=0) # 유저별로 문제 풀기 시작한 시간순으로 정렬
        scaling_cols = []
        
        ############ 1. testId 별로 중간값을 테스트 시간으로 추가해줌
        # 테스트에 제한시간이 있진 않을까 싶어 테스트별 유저의 풀이시간을 기준으로 중간값 채택함
        user_test_timestamp = df[['userID', 'testId', 'Timestamp', 'assessmentItemID']].copy()
        user_test_timestamp['Timestamp'] = pd.to_datetime(user_test_timestamp['Timestamp'])  # Timestamp 열을 datetime 형식으로 변환
        
        user_test_duration = user_test_timestamp.groupby(['testId', 'userID'])['Timestamp']\
            .agg(lambda x: (x.max() - x.min()).total_seconds()).reset_index()
        user_test_duration.columns = ['testId', 'userID', 'duration']
        duration_per_test = user_test_duration.groupby('testId').agg({'duration': lambda x: x.median()})['duration'].to_dict()
        df['duration'] = df['testId'].map(duration_per_test)
        print(">> feature 1 complete")

        ########### 2. testId 별로 순번에 따라 시험시작시간과 경과시간을 추가
        # 제한시간이 있다면, 현재까지 사용한 시간이 중요하지 않을까
        start_time = df[['userID', 'testId', 'Timestamp']].groupby(['userID', 'testId']).agg({'Timestamp':'min'})
        start_time = start_time.to_dict()['Timestamp']
        df['userID_testId'] = list(zip(df['userID'], df['testId']))
        df['startTime'] = df['userID_testId'].map(start_time)
        df['elapsedTime'] = (pd.to_datetime(df['Timestamp']) - pd.to_datetime(df['startTime'])).dt.total_seconds()
        print(">> feature 2 complete")

        ########### 3. testId, 일자별로 user를 그룹화한 값을 추가
        # 단체 응시 같은 유형이 있으면, 같은 시험을 비슷한 시간대에 응시하지 않았을까
        # 일단은 일 단위로 자름
        timestamp = pd.to_datetime(df['Timestamp'])
        df['day'] = timestamp.dt.date
        df['user_category'] = df[['userID', 'testId', 'day']].groupby(['day', 'testId']).ngroup()
        print(">> feature 3 complete")

        ########### 4. testID를 자른 값을 추가
        # testID값을 분리한 값을 추가
        df['test_group_one'] = df['testId'].apply(lambda x: int(x[1:4]))
        df['test_group_two'] = df['testId'].apply(lambda x: int(x[-3:]))
        print(">> feature 4 complete")

        ########### 5. itemID에서 순번을 자른 값을 추가
        # 과제의 순번이 영향이 있지 않을까
        df['serial'] = df['assessmentItemID'].apply(lambda x: int(x[-3:]))
        print(">> feature 5 complete")

        ########### 6. 유저별로 이전에 동일한 문제를 풀었던 횟수를 추가
        # 동일한 과제를 수행했으면 다음번엔 맞출 확률이 높을 것
        df['solved_count'] = df.groupby(['userID', 'assessmentItemID']).cumcount()
        scaling_cols.append('solved_count')
        print(">> feature 6 complete")

        ########### 7. 유저별로 이전에 동일한 문제를 맞췄던 횟수를 추가
        # 동일한 과제를 맞췄었으면 다음번엔 맞출 확률이 높을 것
        df['correct_before'] = df[['userID', 'assessmentItemID', 'answerCode']].groupby(['userID', 'assessmentItemID'])['answerCode'].cumsum()
        df['correct_before'] = df['correct_before'] - df['answerCode']
        scaling_cols.append('correct_before')
        print(">> feature 7 complete")

        ########### 8. 유저별로 이전에 동일한 문제를 틀렸던 횟수를 추가
        # 동일한 과제를 틀렸었으면 다음번엔 맞출 확률이 높을 것
        df['wrong_before'] = df['solved_count'] - df['correct_before']
        scaling_cols.append('wrong_before')
        print(">> feature 8 complete")

        ########### 9. 유저별로 이전에 동일한 태그의 문제를 풀었던 횟수를 추가
        # 동일한 과제를 수행했으면 다음번엔 맞출 확률이 높을 것
        df['same_tag_solved_count'] = df.groupby(['userID', 'assessmentItemID', 'KnowledgeTag']).cumcount()
        scaling_cols.append('same_tag_solved_count')
        print(">> feature 9 complete")

        ########### 10. 유저별로 이전에 동일한 태그의 문제를 맞췄던 횟수를 추가
        # 동일한 과제를 맞췄었으면 다음번엔 맞출 확률이 높을 것
        df['same_tag_correct_before'] = df[['userID', 'assessmentItemID', 'answerCode', 'KnowledgeTag']].groupby(['userID', 'assessmentItemID', 'KnowledgeTag'])['answerCode'].cumsum()
        df['same_tag_correct_before'] = df['same_tag_correct_before'] - df['answerCode']
        scaling_cols.append('same_tag_correct_before')
        print(">> feature 10 complete")

        ########### 11. 유저별로 이전에 동일한 태그의 문제를 틀렸던 횟수를 추가
        #동일한 과제를 틀렸었으면 다음번엔 맞출 확률이 높을 것
        df['same_tag_wrong_before'] = df['same_tag_solved_count'] - df['same_tag_correct_before']
        scaling_cols.append('same_tag_wrong_before')
        print(">> feature 11 complete")

        ########### 12. 과제별 정답률을 추가
        #과제의 정답률을 추가하면 과제의 수준을 알 수 있어 좋을 것이다.
        item_info = df[['assessmentItemID', 'answerCode']].groupby(['assessmentItemID']).agg({'answerCode':'mean'})['answerCode'].to_dict()
        df['item_correct_percent'] = df['assessmentItemID'].map(item_info)
        print(">> feature 12 complete")

        ########### 13. 유저별 정답률을 추가
        #유저의 정답률을 추가하면 유저의 수준을 알 수 있어 좋을 것이다.
        user_info = df[['userID', 'answerCode']].groupby(['userID']).agg({'answerCode':'mean'})['answerCode'].to_dict()
        df['user_correct_percent'] = df['userID'].map(user_info)
        print(">> feature 13 complete")

        ########### 14. 현재까지 맞춘 과제 수
        #현재까지 맞춘 수를 알면 해당 문제를 푸는 시점까지의 유저 상태를 알 수 있을 것이다.
        user_info = df[['userID', 'testId', 'answerCode']].groupby(['userID', 'testId']).agg({'answerCode':'cumsum'})
        # user_info = user_info - df['answerCode']
        df['current_correct_count'] = user_info
        df['current_correct_count'] = df['current_correct_count'] - df['answerCode']
        scaling_cols.append('current_correct_count')
        print(">> feature 14 complete")

        ########### 15. 태그와 테스트 그룹간에 관계가 있지 않을까
        #테스트 그룹을 포함시켰을때 성능이 올라갔는데, 테스트 그룹이 난이도를 나타낸다면,
        #문제 유형으로 예상되는 태그와 연결시켰을때 얻을 수 있는 정보가 생기기 않을까
        df['tag_group_one'] = df['KnowledgeTag'].astype(str) + df['test_group_one'].astype(str)
        df['tag_group_two'] = df['KnowledgeTag'].astype(str) + df['test_group_two'].astype(str)
        print(">> feature 15 complete")

        ########### 16. 문제 푸는 데 걸린 시간
        # to datetime
        data = pd.to_datetime(df['Timestamp'])

        # get consuming time to deal with the problem
        df['time_for_solve'] = (data.shift(-1) - data).dt.seconds

        # get the last item of the users
        df['next_userID'] = df.userID.shift(-1)
        last_appearances = df.apply(lambda x: True if x['userID']!=x['next_userID'] else False, axis=1)
        df.loc[last_appearances, 'time_for_solve'] = np.nan

        # get to know last item of the tests
        df['next_testId'] = df.testId.shift(-1)
        last_appearances = df.apply(lambda x: True if x['testId']!=x['next_testId'] else False, axis=1)
        df.loc[last_appearances, 'time_for_solve'] = np.nan

        # 1시간이 넘게 걸리면 문제가 있다고 본다.
        df.loc[df.time_for_solve > 3600, 'time_for_solve'] = np.nan

        df['time_for_solve'].fillna(df['time_for_solve'].mode()[0], inplace=True)
        scaling_cols.append('time_for_solve')
        print(">> feature 16 complete")

        ########### 17. 찍기 의심 대상
        #time_for_solve 기준 하위 5% 이하인 경우 찍었을 확률이 높다고 판단
        threshold_value = np.percentile(df['time_for_solve'], 5)
        df['guess_yn'] = df['time_for_solve'].apply(lambda x: 'y' if x < threshold_value else 'n')
        print(">> feature 17 complete")

        ########### 18. 유저별 찍는 비율
        guess_yn_per = df[['userID', 'guess_yn']].groupby('userID').agg(guess_yn_user=('guess_yn', lambda x: (x == 'y').mean()))
        df = df.merge(guess_yn_per, how='left', on='userID')
        print(">> feature 18 complete")

        ########### 19. 테스트별 찍는 비율
        guess_yn_per = df[['testId', 'guess_yn']].groupby('testId').agg(guess_yn_test=('guess_yn', lambda x: (x == 'y').mean()))
        df = df.merge(guess_yn_per, how='left', on='testId')
        print(">> feature 19 complete")

        ########### 20. 순번별 찍는 비율
        guess_yn_per = df[['serial', 'guess_yn']].groupby('serial').agg(guess_yn_serial=('guess_yn', lambda x: (x == 'y').mean()))
        df = df.merge(guess_yn_per, how='left', on='serial')
        print(">> feature 20 complete")

        ########### 21. 과제별 찍는 비율
        guess_yn_per = df[['assessmentItemID', 'guess_yn']].groupby('assessmentItemID').agg(guess_yn_assessment=('guess_yn', lambda x: (x == 'y').mean()))
        df = df.merge(guess_yn_per, how='left', on='assessmentItemID')
        print(">> feature 21 complete")

        ########### 22. 태그별 찍는 비율
        guess_yn_per = df[['KnowledgeTag', 'guess_yn']].groupby('KnowledgeTag').agg(guess_yn_tag=('guess_yn', lambda x: (x == 'y').mean()))
        df = df.merge(guess_yn_per, how='left', on='KnowledgeTag')
        print(">> feature 22 complete")

        ########### 23. 요일별 찍는 비율
        df['day_of_week'] = pd.to_datetime(df['Timestamp']).dt.dayofweek
        guess_yn_per = df[['day_of_week', 'guess_yn']].groupby('day_of_week').agg(guess_yn_day=('guess_yn', lambda x: (x == 'y').mean()))
        df = df.merge(guess_yn_per, how='left', on='day_of_week')
        print(">> feature 23 complete")

        ########### 24. 테스트 그룹1 별 찍는 비율
        guess_yn_per = df[['test_group_one', 'guess_yn']].groupby('test_group_one').agg(guess_yn_group_one=('guess_yn', lambda x: (x == 'y').mean()))
        df = df.merge(guess_yn_per, how='left', on='test_group_one')
        print(">> feature 24 complete")

        ########### 25. 테스트 그룹2 별 찍는 비율
        guess_yn_per = df[['test_group_two', 'guess_yn']].groupby('test_group_two').agg(guess_yn_group_two=('guess_yn', lambda x: (x == 'y').mean()))
        df = df.merge(guess_yn_per, how='left', on='test_group_two')
        print(">> feature 25 complete")

        ########### 26. 요일별 정답률을 추가
        info = df[['day_of_week', 'answerCode']].groupby(['day_of_week']).agg({'answerCode':['sum', 'count']})
        info['correct_percent'] = info[('answerCode', 'sum')] / info[('answerCode', 'count')]
        percent = info['correct_percent'].to_dict()
        df['day_correct_percent'] = df['day_of_week'].map(percent)
        print(">> feature 26 complete")

        ########### 27. 테스트 그룹1 별 정답률을 추가
        info = df[['test_group_one', 'answerCode']].groupby(['test_group_one']).agg({'answerCode':['sum', 'count']})
        info['correct_percent'] = info[('answerCode', 'sum')] / info[('answerCode', 'count')]
        percent = info['correct_percent'].to_dict()
        df['correct_percent_group_one'] = df['test_group_one'].map(percent)
        print(">> feature 27 complete")

        ########### 28. 테스트 그룹2 별 정답률을 추가
        info = df[['test_group_two', 'answerCode']].groupby(['test_group_two']).agg({'answerCode':['sum', 'count']})
        info['correct_percent'] = info[('answerCode', 'sum')] / info[('answerCode', 'count')]
        percent = info['correct_percent'].to_dict()
        df['correct_percent_group_two'] = df['test_group_two'].map(percent)
        print(">> feature 28 complete")

        ########### 29. 순번별 정답률을 추가
        info = df[['serial', 'answerCode']].groupby(['serial']).agg({'answerCode':['sum', 'count']})
        info['correct_percent'] = info[('answerCode', 'sum')] / info[('answerCode', 'count')]
        percent = info['correct_percent'].to_dict()
        df['correct_percent_serial'] = df['serial'].map(percent)
        print(">> feature 29 complete")

        ########### 30. 순번별 정답률을 추가
        threshold_value = np.percentile(df['time_for_solve'], 5)
        df['guess_yn'] = df['time_for_solve'].apply(lambda x: 'y' if x < threshold_value else 'n')
        print(">> feature 30 complete")

        ########### 31. 유저별 문제 풀이 시간
        duration_per_user = user_test_duration.groupby('userID').agg({'duration': lambda x: x.median()})['duration'].to_dict()
        df['duration_user'] = df['userID'].map(duration_per_user)
        print(">> feature 31 complete")

        ########### 32. 유저별 문제 풀이 시간대
        # 가장 많이 푼 시간대에 푸는 경우 더 정답률이 높을 것이다.
        df['hour'] = pd.to_datetime(df['Timestamp']).dt.hour
        hour_per_user = df[['userID', 'hour']].groupby('userID').agg({'hour': lambda x: x.mode()}).to_dict()
        df['user_mode_hour'] = df['userID'].map(hour_per_user)
        print(">> feature 32 complete")

        ########### 33. 유저별 최다 문제 풀이 년도
        # 가장 많이 푼 년도에 가장 실력이 높았다가 점점 낮아질 것이다.
        df['year'] = pd.to_datetime(df['Timestamp']).dt.year
        year_per_user = df[['userID', 'year']].groupby('userID').agg({'year': lambda x: x.mode()}).to_dict()
        df['user_mode_year'] = df['userID'].map(year_per_user)
        print(">> feature 33 complete")

        ########### 34. 테스트별 처음 발생 년도
        # 오래된 문제는 풀이법이 많이 풀려서 정답을 맞출 확률이 높을 것이다.
        year_per_test = df[['testId', 'year']].groupby('testId').agg({'year': lambda x: x.min()}).to_dict()
        df['test_min_year'] = df['testId'].map(year_per_test)
        print(">> feature 34 complete")

        ########### 35. 테스트별 문항 최다 발생 년도
        # 가장 빈번하게 풀린 문제는 해당 년도애 풀면 더 잘 맞출 것이다.
        year_per_test = df[['testId', 'year']].groupby('testId').agg({'year': lambda x: x.mode()}).to_dict()
        df['test_mode_year'] = df['testId'].map(year_per_test)
        print(">> feature 35 complete")

        ########### 36. 테스트별 마지막 발생 년도
        # 오래전에 풀린 문제일 수록 정보가 적어져 잘 못 풀수 있을 것이다.
        year_per_test = df[['testId', 'year']].groupby('testId').agg({'year': lambda x: x.max()}).to_dict()
        df['test_max_year'] = df['testId'].map(year_per_test)
        print(">> feature 36 complete")

        ########### 37. 아이템별 처음 발생 년도
        # 오래된 문제는 풀이법이 많이 풀려서 정답을 맞출 확률이 높을 것이다.
        year_per_item = df[['assessmentItemID', 'year']].groupby('assessmentItemID').agg({'year': lambda x: x.min()}).to_dict()
        df['item_min_year'] = df['assessmentItemID'].map(year_per_item)
        print(">> feature 37 complete")

        ########### 38. 아이템별 문항 최다 발생 년도
        # 가장 빈번하게 풀린 문제는 해당 년도에 풀면 더 잘 맞출 것이다.
        year_per_item = df[['assessmentItemID', 'year']].groupby('assessmentItemID').agg({'year': lambda x: x.mode()}).to_dict()
        df['item_mode_year'] = df['assessmentItemID'].map(year_per_item)
        print(">> feature 38 complete")

        ########### 39. 아이템별 마지막 발생 년도
        # 오래전에 풀린 문제일 수록 정보가 적어져 못 풀수 있을 것이다.
        year_per_item = df[['assessmentItemID', 'year']].groupby('assessmentItemID').agg({'year': lambda x: x.max()}).to_dict()
        df['item_max_year'] = df['assessmentItemID'].map(year_per_item)
        print(">> feature 39 complete")

        ########### 40. 유저별 마지막 풀이 년도
        # 테스트를 오래전에 풀었을 수록 실력이 떨어질 것이다.
        year_info = df[['userID', 'year']].groupby('userID').agg({'year': lambda x: x.max()}).to_dict()
        df['user_max_year'] = df['userID'].map(year_info)
        print(">> feature 40 complete")

        ########### 41. 유저별 최초 풀이 년도
        # 테스트를 오래전부터 풀기 시작했을수록 실력이 높을 것이다.
        year_info = df[['userID', 'year']].groupby('userID').agg({'year': lambda x: x.min()}).to_dict()
        df['user_min_year'] = df['userID'].map(year_info)
        print(">> feature 41 complete")

        ########### 42. 유저별 문제 풀이 기간
        # 테스트를 오랜 기간 풀었을 수록 실력이 높을 것이다.
        df['user_period_year'] = df['user_max_year'] - df['user_min_year']
        print(">> feature 42 complete")

        ########### 43. 테스트별 문제 풀이 횟수
        # 테스트가 많이 풀렸을 수록 정답 정보가 많이 알려져 난이도가 낮아질 것이다.
        test_info = df[['testId', 'answerCode']].groupby('testId').agg({'answerCode': lambda x: x.count()}).to_dict()
        df['test_count'] = df['testId'].map(test_info)
        scaling_cols.append('test_count')
        print(">> feature 43 complete")

        ########### 44. 과제별 문제 풀이 횟수
        # 과제가 많이 풀렸을 수록 정답 정보가 많이 알려져 난이도가 낮아질 것이다.
        test_info = df[['assessmentItemID', 'answerCode']].groupby('assessmentItemID').agg({'answerCode': lambda x: x.count()}).to_dict()
        df['item_count'] = df['assessmentItemID'].map(test_info)
        scaling_cols.append('item_count')
        print(">> feature 44 complete")

        ########### 45. 유저별 문제 풀이 횟수
        # 과제가 많이 풀렸을 수록 정답 정보가 많이 알려져 난이도가 낮아질 것이다.
        # test_info = df[['assessmentItemID', 'answerCode']].groupby('assessmentItemID').agg({'answerCode': lambda x: x.count()}).to_dict()
        # df['item_count'] = df['assessmentItemID'].map(test_info)
        # scaling_cols.append('item_count')
        # print(">> feature 45 complete")
        ########### 45. 과제 난이도
        df['item_difficulty'] = df['time_for_solve'] / (df['item_correct_percent'] + 1)
        print(">> feature 45 complete")

        ########### 46. 마지막 학습일 까지의 기간
        # 마지막 학습일과 각 레코드의 날짜 차이 (일수)
        df['time_diff'] = df.groupby('userID')['Timestamp'].transform(lambda x: pd.to_datetime(x.iloc[-1])-pd.to_datetime(x)).dt.seconds
        scaling_cols.append('time_diff')
        print(">> feature 46 complete")

        ########### 47. 그냥 0 넣었을 때 성능 올라갔던것 같아서 추가
        df['zero'] = 0
        print(">> feature 47 complete")

        scaler = StandardScaler()
        scaler = scaler.fit(df[scaling_cols])
        print(">> Standardization complete")

        print(df.columns)
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
        additional_cols = ["duration",
                           "startTime",
                           "elapsedTime",
                           "user_category",
                           "test_group_one",
                           "test_group_two",
                           "serial",
                           "solved_count",
                           "correct_before",
                           "wrong_before",
                           "same_tag_solved_count",
                           "same_tag_correct_before",
                           "same_tag_wrong_before",
                           "item_correct_percent",
                           "user_correct_percent",
                           "current_correct_count",
                           "tag_group_one",
                           "tag_group_two",
                           'time_for_solve',
                           'guess_yn',
                           'guess_yn_user',
                           'guess_yn_test',
                           'guess_yn_serial',
                           'guess_yn_assessment',
                           'guess_yn_tag',
                           'guess_yn_day',
                           'guess_yn_group_one',
                           'guess_yn_group_two',
                           'correct_percent_group_one',
                           'correct_percent_group_two',
                           'correct_percent_serial',
                           'day_of_week',
                           'duration_user',
                           'item_difficulty',
                           'zero',
                           ]

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
                    r["startTime"].values,
                    r["elapsedTime"].values, # 경과시간 열
                    r["test_group_one"].values,
                    r["test_group_two"].values,
                    r["serial"].values,
                    r["solved_count"].values,
                    r["correct_before"].values,
                    r["wrong_before"].values,
                    r["same_tag_solved_count"].values,
                    r["same_tag_correct_before"].values,
                    r["same_tag_wrong_before"].values,
                    r["item_correct_percent"].values,
                    r["user_correct_percent"].values,
                    r["current_correct_count"].values,
                    r["tag_group_one"].values,
                    r["tag_group_two"].values,
                    r["answerCode"].values, # target 열
                    r["time_for_solve"].values,
                    r["guess_yn"].values,
                    r["guess_yn_user"].values,
                    r["guess_yn_test"].values,
                    r["guess_yn_serial"].values,
                    r["guess_yn_assessment"].values,
                    r["guess_yn_tag"].values,
                    r["guess_yn_day"].values,
                    r["guess_yn_group_one"].values,
                    r["guess_yn_group_two"].values,
                    r["correct_percent_group_one"].values,
                    r["correct_percent_group_two"].values,
                    r["correct_percent_serial"].values,
                    r["day_of_week"].values,
                    r["duration_user"].values,
                    r['item_difficulty'].values,
                    r['zero'].values,
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
        (test, # 
         question, # 
         tag, # 
         duration, # 
         startTime, # 
         elapsedTime, # 
         testGroupOne, # 
         testGroupTwo, # 
         serial, # 
         solved_count, # 
         correct_before, # 
         wrong_before, # 
         same_tag_solved_count, # 
         same_tag_correct_before, # 
         same_tag_wrong_before, # 
         item_correct_percent, # 
         user_correct_percent, # 
         current_correct_count, # 
         tag_group_one, # 
         tag_group_two, # 
         correct, # 
         time_for_solve, # 
         guess_yn, # 
         guess_yn_user, # 
         guess_yn_test, # 
         guess_yn_serial, # 
         guess_yn_assessment, # 
         guess_yn_tag, # 
         guess_yn_day, # 
         guess_yn_group_one, # 
         guess_yn_group_two, # 
         correct_percent_group_one, # 
         correct_percent_group_two, # 
         correct_percent_serial, # 
         day_of_week,
         duration_user,
         item_difficulty,
         zero,
         ) = (
            *row,
            )
        # print(type(duration), duration)
        data = {
            "test": torch.tensor(test + 1, dtype=torch.int),
            # "assessmentItemID": torch.tensor(assessmentItemID + 1, dtype=torch.int),
            "question": torch.tensor(question + 1, dtype=torch.int),
            "tag": torch.tensor(tag + 1, dtype=torch.int),
            "correct": torch.tensor(correct, dtype=torch.int),
            "duration": torch.tensor(duration, dtype=torch.float),
            "startTime": torch.tensor(startTime, dtype=torch.float),
            "elapsedTime": torch.tensor(elapsedTime, dtype=torch.float),
            "test_group_one": torch.tensor(testGroupOne + 1, dtype=torch.int),
            "test_group_two": torch.tensor(testGroupTwo + 1, dtype=torch.int),
            "serial": torch.tensor(serial, dtype=torch.int),
            "solved_count": torch.tensor(solved_count, dtype=torch.float),
            "correct_before": torch.tensor(correct_before, dtype=torch.float),
            "wrong_before": torch.tensor(wrong_before, dtype=torch.float),
            "same_tag_solved_count": torch.tensor(same_tag_solved_count, dtype=torch.float),
            "same_tag_correct_before": torch.tensor(same_tag_correct_before, dtype=torch.float),
            "same_tag_wrong_before": torch.tensor(same_tag_wrong_before, dtype=torch.float),
            "item_correct_percent": torch.tensor(item_correct_percent, dtype=torch.float),
            "user_correct_percent": torch.tensor(user_correct_percent, dtype=torch.float),
            "current_correct_count": torch.tensor(current_correct_count, dtype=torch.float),
            "tag_group_one": torch.tensor(tag_group_one + 1, dtype=torch.int),
            "tag_group_two": torch.tensor(tag_group_two + 1, dtype=torch.int),
            "time_for_solve": torch.tensor(time_for_solve, dtype=torch.float),
            "guess_yn": torch.tensor(guess_yn, dtype=torch.int),
            "guess_yn_user": torch.tensor(guess_yn_user, dtype=torch.float),
            "guess_yn_test": torch.tensor(guess_yn_test, dtype=torch.float),
            "guess_yn_serial": torch.tensor(guess_yn_serial, dtype=torch.float),
            "guess_yn_assessment": torch.tensor(guess_yn_assessment, dtype=torch.float),
            "guess_yn_tag": torch.tensor(guess_yn_tag, dtype=torch.float),
            "guess_yn_day": torch.tensor(guess_yn_day, dtype=torch.float),
            "guess_yn_group_one": torch.tensor(guess_yn_group_one, dtype=torch.float),
            "guess_yn_group_two": torch.tensor(guess_yn_group_two, dtype=torch.float),
            "correct_percent_group_one": torch.tensor(correct_percent_group_one, dtype=torch.float),
            "correct_percent_group_two": torch.tensor(correct_percent_group_two, dtype=torch.float),
            "correct_percent_serial": torch.tensor(correct_percent_serial, dtype=torch.float),
            "day_of_week": torch.tensor(day_of_week, dtype=torch.int),
            "duration_user": torch.tensor(duration_user, dtype=torch.float),
            "item_difficulty": torch.tensor(item_difficulty, dtype=torch.float),
            'zero': torch.tensor(zero, dtype=torch.float),
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
