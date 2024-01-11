import argparse
from feature_e import FE
import pandas as pd
parser = argparse.ArgumentParser(description='argparse 테스트(FE)')


parser.add_argument('--path', required=False, default='/opt/ml/input/data/',help='default= /opt/ml/input/data/')
parser.add_argument('--data', required=False, default='train_data.csv',help='default= train_data.csv')
args = parser.parse_args()

train = pd.read_csv(args.path + args.data)

data = FE(train)
data.to_csv('FE_data.csv', index=False)