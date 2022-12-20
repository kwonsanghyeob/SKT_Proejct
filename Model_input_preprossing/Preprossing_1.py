import numpy as np


def Make_input(data, time_scale = 24, Window_day = 12, train_day = 638, Val_day = 20, test_day = 262):
    '''
    :param data: 모델을 만들기 위한 입력 데이터
    :param time_scale: Multi-timestep 수 ==> 얼마나 예측 할 것인지를 정의
    :param Window_day: 배치사이즈
    :param train_day: 훈련 날짜(단위는 전체 데이터의 %)
    :param Val_day: 검증 날짜(단위는 전체 데이터의 %)
    :param test_day: 테스트 날짜(단위는 전체 데이터의 %)
    :return: train, val, test 데이터
    '''
    time_scale = time_scale
    window_length = Window_day*time_scale


    train = data[:time_scale*train_day]
    test = data[-time_scale*test_day:]
    return train, test

#TODO 입력데이터를 정의
#배치사이즈를 288개로 만들기

def train_test(train_data, test_data, Window_day=12,train_day = 638, Val_day = 20, test_day = 261, time_scale = 24):
    data = np.concatenate((train_data,test_data))

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

    x_test = x[-test_day:]
    y_test = y[-test_day:]

    return x_train, y_train, x_test, y_test