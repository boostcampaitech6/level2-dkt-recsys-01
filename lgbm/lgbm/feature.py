import numpy as np
import pandas as pd


def shift_correct(df):
    # 미래 정보
    df['correct_shift_-2'] = df.groupby('userID')['answerCode'].shift(-2)
    df['correct_shift_-1'] = df.groupby('userID')['answerCode'].shift(-1)

    # 과거 정보
    df['correct_shift_1'] = df.groupby('userID')['answerCode'].shift(1)
    df['correct_shift_2'] = df.groupby('userID')['answerCode'].shift(2)
    return df
    

# 과거에 맞춘 문제 수
def num_correct_past(df):
    # 과거에 맞춘 문제 수
    df['shift'] = df.groupby('userID')['answerCode'].shift().fillna(0)
    df['past_correct'] = df.groupby('userID')['shift'].cumsum()
    return df


# 과거에 해당 문제를 맞춘 횟수
def past_assessmentItem_correct(df):
    # 과거에 해당 문제를 맞춘 횟수
    df['shift'] = df.groupby(['userID', 'assessmentItemID'])['answerCode'].shift().fillna(0)
    df['past_content_correct'] = df.groupby(['userID', 'assessmentItemID'])['shift'].cumsum()
    return df


# 과거에 푼 문제 수
def past_problem_nums(df):
    # 과거에 푼 문제 수
    df['past_count'] = df.groupby('userID').cumcount()
    return df


# 과거 평균 정답률
def user_average_correct(df):
    # 과거에 푼 문제 수
    df['past_count'] = df.groupby('user').cumcount()

    # 과거에 맞춘 문제 수
    df['shift'] = df.groupby('user')['correct'].shift().fillna(0)
    df['past_correct'] = df.groupby('user')['shift'].cumsum()

    # 과거 평균 정답률
    df['average_correct'] = (df['past_correct'] / df['past_count']).fillna(0)
    return df


# 과거 해당 문제 평균 정답률
def problem_average_correct(df):
    # 과거에 해당 문제를 푼 수
    df['past_content_count'] = df.groupby(['user', 'content']).cumcount()

    # 과거에 해당 문제를 맞춘 수
    df['shift'] = df.groupby(['user', 'content'])['correct'].shift().fillna(0)
    df['past_content_correct'] = df.groupby(['user', 'content'])['shift'].cumsum()
    df['past_content_correct'] = df.groupby(['userID', 'assessmentItemID'])['shift'].cumsum()
    
    # 과거 해당 문제 평균 정답률
    df['average_content_correct'] = (df['past_content_correct'] / df['past_content_count']).fillna(0)
    return df