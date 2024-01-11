from datetime import datetime
import time
import pandas as pd

def FE(train) -> pd.DataFrame:
    train['level'] = train['testId'].apply(lambda x: x[2])
    train['Timestamp'] = train['Timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    train['cal_time'] = train['Timestamp'].apply(lambda x: int(time.mktime(x.timetuple())))
    train['shift'] = train['cal_time'].shift(-1).fillna(0).astype(int)
    train['elapsed'] =  train['shift'] - train['cal_time']
    train['check'] = train['testId'].shift(-1)
    train.loc[train['testId'] != train['check'], 'elapsed'] = 0
    train = train.drop(['cal_time','shift', 'check'], axis=1)
    # takes 2m 

    # 정답과 오답의 평균,중간 소요시간
    collect_elp_mean = train[train['answerCode'] == 1].groupby('assessmentItemID')['elapsed'].mean()
    train = train.join(collect_elp_mean, on='assessmentItemID',rsuffix='_1_avg')
    wrong_elp_mean = train[train['answerCode']== 0].groupby('assessmentItemID')['elapsed'].mean()
    train = train.join(wrong_elp_mean, on='assessmentItemID',rsuffix='_0_avg')

    collect_elp_median = train[train['answerCode'] == 1].groupby('assessmentItemID')['elapsed'].median()
    train = train.join(collect_elp_median, on='assessmentItemID',rsuffix='_1_mdn')
    wrong_elp_median = train[train['answerCode']== 0].groupby('assessmentItemID')['elapsed'].median()
    train = train.join(wrong_elp_median, on='assessmentItemID',rsuffix='_0_mdn')

    # 전체 표준편차
    elapsed_std = train.groupby('assessmentItemID')['elapsed'].std()
    train = train.join(elapsed_std, on='assessmentItemID', rsuffix='_std')
    # 맞춘 인원의 표준편차
    elapsed_1_std = train[train['answerCode'] == 1].groupby('assessmentItemID')['elapsed'].std()
    train = train.join(elapsed_1_std, on='assessmentItemID', rsuffix='_1_std')
    # 정답률
    answer_rate = train.groupby('assessmentItemID')['answerCode'].mean()
    train = train.join(answer_rate, on='assessmentItemID', rsuffix='_rate')
    #문제를 푼 인원 
    usr_cnt = train.groupby('assessmentItemID')['userID'].count()
    train = train.join(usr_cnt, on='assessmentItemID', rsuffix='_cnt')

    return train
