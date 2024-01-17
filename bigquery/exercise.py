from auth.client import client
from data import read_dataframe_with_features, add_new_feature_data
import pandas as pd

table_id = 'dkt-recsys01.dataset.dkt_train'
data = pd.read_csv('/opt/ml/input/data/train_data.csv')

# print("##### [TEST] Read features")
# data = read_dataframe_with_features('userID','assessmentItemID','testId', 'answerCode', 'Timestamp', 'KnowledgeTag')
# print(data.head())

data['test_group_two'] = data['testId'].apply(lambda x: int(x[1:4]))

# print("##### [TEST] Add new feature")
add_new_feature_data(data, feature_name = 'test_group_two', feature_type = 'INT64')