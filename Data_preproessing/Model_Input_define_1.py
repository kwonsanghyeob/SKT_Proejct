import pandas as pd
import datetime
import numpy as np

def Date_extraction(Data, Colums):
    EV = Data[Colums]
    Target = pd.DataFrame(np.array(np.sum(EV,axis=1)), columns=['target'])
    Target['Unnamed: 0'] = np.array(Data['Unnamed: 0'])
    return Target


def Onehot_encoding(Data, mode = 2):
    '''
    :param Data: 전기차 입력데이터
    :param mode: 1은 휴일X(입력데이터에서 삭제), 2는 휴일 입력데이터로 사용(원-핫인코딩), 3은 평일/휴일 따로 예측 모델 구성하기위한 모델
    :return:
    '''

    #웟-핫 인코딩(휴일, 요일, 시간)
    Data['time_index'] = None
    for i in range(len(Data)):
        Data['time_index'].values[i] = datetime.datetime.strptime(Data['Unnamed: 0'][i], '%Y-%m-%d %H:%M:%S').time().strftime('%H:%M')

    c = ['2020-01-01', '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27', '2020-03-01', '2020-04-15', '2020-04-30',
         '2020-05-05', '2020-06-06', '2020-08-17', '2020-09-30', '2020-10-01', '2020-10-02', '2020-10-03', '2020-10-09',
         '2020-12-25', '2021-01-01', '2021-02-11', '2021-02-12', '2021-02-13', '2021-03-01', '2021-05-05', '2021-05-19',
         '2021-06-06', '2021-08-15', '2021-08-16', '2021-09-20', '2021-09-21', '2021-09-22', '2021-10-04', '2021-10-09',
         '2021-12-25', '2022-01-01', '2022-01-31', '2022-02-01', '2022-02-02', '2022-03-01', '2022-03-09', '2022-05-05',
         '2022-05-08', '2022-06-01', '2022-06-06', '2022-08-15', '2022-09-09', '2022-09-10', '2022-09-11', '2022-09-12',
         '2022-10-03', '2022-10-09', '2022-10-10', '2022-12-25']


    Holiday_index = []
    Data['Weekday'] = None
    Data['check_weekend'] = None
    for i in range(len(Data)):
        Week_value=datetime.datetime.strptime(Data['Unnamed: 0'][i], '%Y-%m-%d %H:%M:%S').weekday()
        Data['Weekday'].values[i] = Week_value
        #TODO 휴일특성을 넣고싶으면 주석제거
        for j in c:
            if datetime.datetime.strptime(Data['Unnamed: 0'][i], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d') == j:
                Holiday_index.append(i)
                break
        if int(Week_value) <= 4:
            Data['check_weekend'].values[i] = 0
        elif int(Week_value) > 4:
            Data['check_weekend'].values[i] = 1



    for i in Holiday_index:
        Data['check_weekend'].values[i] =1

    if mode ==1:
        Data = pd.get_dummies(Data, columns=['time_index', 'Weekday'])
        filtered_df = Data.drop(['Unnamed: 0', 'check_weekend'], axis=1)
    elif mode == 2:
        Data = pd.get_dummies(Data, columns=['time_index', 'Weekday', 'check_weekend'])
        filtered_df = Data.drop(['Unnamed: 0'], axis=1)
    elif mode == 3:
        Data = pd.get_dummies(Data, columns=['time_index', 'Weekday'])
        filtered_df = Data.drop(['Unnamed: 0'], axis=1)




    return filtered_df


def Make_input(data, time_scale = 24, Window_day = 12, train_day = 700, Val_day = 91, test_day = 91):
    '''
    :param data: 모델을 만들기 위한 입력 데이터
    :param time_scale: Multi-timestep 수 ==> 얼마나 예측 할 것인지를 정의
    :param Window_day: 배치사이즈
    :param train_day: 훈련 날짜(단위는 전체 데이터의 %)
    :param Val_day: 검증 날짜(단위는 전체 데이터의 %)
    :param test_day: 테스트 날짜(단위는 전체 데이터의 %)
    :return: train, val, test 데이터
    '''
    Data_dividing_test = (Window_day+train_day+Val_day+test_day)*time_scale
    if Data_dividing_test > len(data):
        raise Exception('Training Day를 줄이세요')

    time_scale = time_scale
    window_length = Window_day*time_scale
    Make_data = np.concatenate((data[window_length:, [0]], data[:-window_length*1, 1:],  data[:-window_length,[0]]), axis=1)

    train = Make_data[:time_scale*train_day]
    val = Make_data[time_scale*train_day:time_scale * (train_day+Val_day)]
    test = Make_data[-time_scale*test_day:]
    return train, val, test


def train_test(train_data, val_data,test_data, time_scale = 24, Window_day=12,train_day = 791, Val_day = 20, test_day = 91):
    data = np.concatenate((train_data, val_data,test_data))

    x = list()
    y = list()
    for i in range(0, int((len(data))/time_scale)-Window_day):
        window = data[time_scale*i:time_scale*(i + Window_day), :]
        x.append(window[:,1:])
        y.append(window[:24, 0])

    x = np.array(x)
    y = np.array(y)

    x_train = x[:train_day]
    y_train = y[:train_day]

    x_val = x[train_day:(train_day+Val_day)]
    y_val = y[train_day:(train_day+Val_day)]

    x_test = x[-test_day:]
    y_test = y[-test_day:]

    return x_train, y_train, x_val, y_val, x_test, y_test