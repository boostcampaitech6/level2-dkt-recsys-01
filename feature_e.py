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



    return train
