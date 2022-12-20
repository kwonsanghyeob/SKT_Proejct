import numpy as np
import datetime
import pandas as pd

def Date_extraction(Data, Colums):
    #TODO Zone나누어지면 그떄 코드 수정이 필요함
    EV = Data[Colums]
    Target = pd.DataFrame(np.sum(EV,axis=1), columns=['target'])
    Target['Unnamed: 0'] = Data['Unnamed: 0']
    return Target


def Onehot_encoding(Data):
    #웟-핫 인코딩(휴일, 요일, 시간)
    Data['time_index'] = None
    for i in range(len(Data)):
        Data['time_index'].values[i] = datetime.datetime.strptime(Data['Unnamed: 0'][i], '%Y-%m-%d %H:%M:%S').time().strftime('%H:%M')

    Data['Weekday'] = None
    # Data['check_weekend'] = None
    for i in range(len(Data)):
        Week_value=datetime.datetime.strptime(Data['Unnamed: 0'][i], '%Y-%m-%d %H:%M:%S').weekday()
        Data['Weekday'].values[i] = Week_value
        #TODO 휴일특성을 넣고싶으면 주석제거

        # if int(Week_value) <= 4:
        #     Data['check_weekend'].values[i] = 0
        # elif int(Week_value) > 4:
        #     Data['check_weekend'].values[i] = 1

    Data = pd.get_dummies(Data, columns=['time_index','Weekday'])
    filtered_df = Data.loc[:,Data.columns != 'Unnamed: 0']
    return filtered_df
