import numpy as np
import pandas as pd
import datetime

class Ensemble:
    '''
    [description]
    앙상블을 진행하는 클래스입니다.

    [parameter]
    filenames: 앙상블을 진행할 모델 경로를 리스트 형태로 입력합니다.
    filepath: 앙상블을 진행할 모델의 csv 파일이 저장된 경로를 입력합니다.
    '''
    def __init__(self, filenames:str, filepath:str):
        self.filenames = filenames
        self.output_list = []
        self.output_frame = pd.read_csv(filenames[0][0]).drop('prediction',axis=1)
        self.output_df = self.output_frame.copy()

        for path in filenames:
            self.output_list.append(pd.read_csv(path[0])['prediction'].to_list())
        
        # 이것은 필요가 없다고 생각됨 , 근데 이유를 모르겠음 없애니까 작동이 됨.
        #for filename,output in zip(filenames,self.output_list):
        #    self.output_df[filename] = output


    def simple_weighted(self,weight:list):
        '''
        [description]
        직접 weight를 지정하여, 앙상블합니다.
        
        [parameter]
        weight: 각 모델의 weight를 리스트 형태로 입력합니다.
        이 때, weight의 합은 1이 되도록 입력해 주세요.
        
        [return]
        result: 앙상블 결과를 리스트 형태로 반환합니다.
        '''
        if not len(self.output_list)==len(weight):
            raise ValueError("model과 weight의 길이가 일치하지 않습니다.")
        if np.sum(weight)!=1:
            raise ValueError("weight의 합이 1이 되도록 입력해 주세요.")

        pred_arr = np.append([self.output_list[0]], [self.output_list[1]], axis=0)
        for i in range(2, len(self.output_list)):
            pred_arr = np.append(pred_arr, [self.output_list[i]], axis=0)
        result = np.dot(pred_arr.T, np.array(weight))
        return result.tolist()


    def average_weighted(self):
        '''
        [description]
        (1/n)의 가중치로 앙상블을 진행합니다.
        
        [return]
        result: 앙상블 결과를 리스트 형태로 반환합니다.
        '''
        weight = [1/len(self.output_list) for _ in range(len(self.output_list))]
        pred_weight_list = [pred*np.array(w) for pred, w in zip(self.output_list,weight)]
        result = np.sum(pred_weight_list, axis=0)
        return result.tolist()


    def mixed(self):
        '''
        [description]
        Negative case 발생 시, 다음 순서에서 예측한 rating으로 넘어가서 앙상블합니다.
        
        [return]
        result: 앙상블 결과를 리스트 형태로 반환합니다.
        '''
        result = self.output_df[self.filenames[0]].copy()
        for idx in range(len(self.filenames)-1):
            pre_idx = self.filenames[idx]
            post_idx = self.filenames[idx+1]
            result[self.output_df[pre_idx]<1] = self.output_df.loc[self.output_df[pre_idx]<1,post_idx]
        return result.tolist()
