import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from Forecasting_model.SKT_LSTM import Mymodel
from Model_Input_define.Input_preproceessing_1 import Date_extraction, Onehot_encoding
from Model_input_preprossing.Preprossing_1 import Make_input, train_test


def Mode_1():


    for i in range(1, 6):
        globals()[f'Zone{i}_target'.format(i)] = Date_extraction(EV_data, globals()[f'Zone{i}'])
        globals()[f'Target_onehot_{i}'.format(i)] = Onehot_encoding(globals()[f'Zone{i}_target'])
        globals()[f'Target{i}_train'.format(i)], globals()[f'Target{i}_test'], = Make_input(
            np.array(globals()[f'Target_onehot_{i}']))
        globals()[f'MinMaxScaler_{i}'.format(i)] = MinMaxScaler()
        # 문제발생코드
        globals()[f'MinMaxScaler_{i}_Traget{i}_train'] = globals()[f'MinMaxScaler_{i}'].fit_transform(
            globals()[f'Target{i}_train'])
        globals()[f'MinMaxScaler_{i}_Traget{i}_test'] = globals()[f'MinMaxScaler_{i}'].transform(
            globals()[f'Target{i}_test'])
        globals()[f'x_Traget{i}_train'], globals()[f'y_Traget{i}_train'], globals()[f'x_Traget{i}_test'], globals()[
            f'y_Traget{i}_test'] = train_test(globals()[f'MinMaxScaler_{i}_Traget{i}_train'],
                                              globals()[f'MinMaxScaler_{i}_Traget{i}_test'])
        globals()[f'model_Traget{i}'] = Mymodel()
        globals()[f'model_Traget{i}'].compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.00002),
                                              metrics=['mae'])
        globals()[f'history_Target{i}'] = globals()[f'model_Traget{i}'].fit(globals()[f'x_Traget{i}_train'],
                                                                            globals()[f'y_Traget{i}_train'], epochs=150)
        globals()[f'Target{i}_y_predict'] = globals()[f'model_Traget{i}'].predict(globals()[f'x_Traget{i}_test'])
        globals()[f'Target{i}_y_predict_inverse'] = globals()[f'MinMaxScaler_{i}'].inverse_transform(np.hstack((globals()[
                                                                                                                    f'Target{i}_y_predict'].reshape(
            -1, 1), np.zeros(
            [globals()[f'Target{i}_y_predict'].reshape(-1).shape[0], globals()[f'Target{i}_test'].shape[1] - 1]))))[:, 0]
        globals()[f'Target{i}_y_test_inverse'] = globals()[f'MinMaxScaler_{i}'].inverse_transform(np.hstack((globals()[
                                                                                                                 f'y_Traget{i}_test'].reshape(
            -1, 1), np.zeros(
            [globals()[f'y_Traget{i}_test'].reshape(-1).shape[0], globals()[f'Target{i}_test'].shape[1] - 1]))))[:, 0]


    #
    def NMAE(true, pred):
        '''
        true: np.array
        pred: np.array
        '''
        return np.mean(np.abs(true - pred) / (max(true) - min(true))) * 100


    a = NMAE(Target1_y_predict_inverse, Target1_y_test_inverse)
    b = NMAE(Target2_y_predict_inverse, Target2_y_test_inverse)
    c = NMAE(Target3_y_predict_inverse, Target3_y_test_inverse)
    d = NMAE(Target4_y_predict_inverse, Target4_y_test_inverse)
    e = NMAE(Target5_y_predict_inverse, Target5_y_test_inverse)
    #
    print(NMAE(Target1_y_predict_inverse, Target1_y_test_inverse))
    print(NMAE(Target2_y_predict_inverse, Target2_y_test_inverse))
    print(NMAE(Target3_y_predict_inverse, Target3_y_test_inverse))
    print(NMAE(Target4_y_predict_inverse, Target4_y_test_inverse))
    print(NMAE(Target5_y_predict_inverse, Target5_y_test_inverse))
    # print((a+b+c+d)/3)




if __name__ == '__main__':
    mode = 1
    if mode ==1:
        warnings.filterwarnings('ignore')

        os.getcwd()
        # os.chdir('/home/hy/PycharmProjects/sanghyeob_test/EV_forecasting')

        a = pd.read_csv('../SKT_Month_power_22.12.12.csv', encoding='cp949')
        EV_data = pd.read_csv('../SKT_Timeseries_station.csv', encoding='cp949')

        # 상위 충전소만 추출
        # 1. 종합경기장
        # 2. 종합경기장, LH제주본부, 신성로 공영주차장, 현일아파트
        # 3. 종합경기장, LH제주본부, 신성로 공영주차장, 현일아파트, 제주공항, 제주특별자치도복지이음마루, 정원파인즈10차
        # 4. 종합경기장, LH제주본부, 신성로 공영주차장, 현일아파트, 제주공항, 제주특별자치도복지이음마루, 정원파인즈10차, 제주도교육청, 제주직할, 베라체공영주차장

        Zone1 = ['종합경기장']
        Zone2 = ['종합경기장', 'LH제주본부', '신성로 공영주차장', '현일 아파트']
        Zone3 = ['종합경기장', 'LH제주본부', '신성로 공영주차장', '현일 아파트', '제주공항', '제주특별자치도 복지이음마루', '정원파인즈 10차']
        Zone4 = ['종합경기장', 'LH제주본부', '신성로 공영주차장', '현일 아파트', '제주공항', '제주특별자치도 복지이음마루', '정원파인즈 10차', '제주도교육청', '제주직할',
                 '베라체 공영주차장']
        Zone5 = EV_data.columns[1:]

        Mode_1()




    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rc('font', size=40)        # 기본 폰트 크기
    plt.rc('axes', labelsize=20)   # x,y축 label 폰트 크기
    plt.rc('xtick', labelsize=20)  # x축 눈금 폰트 크기
    plt.rc('ytick', labelsize=20)  # y축 눈금 폰트 크기
    plt.rc('legend', fontsize=20)  # 범례 폰트 크기
    plt.rc('figure', titlesize=20) # figure title 폰트 크기



    plt.figure(num =1, figsize=[20,10])
    plt.plot(Target1_y_predict_inverse[-24*7:], linewidth='5',label='Forecast')
    plt.plot(Target1_y_test_inverse[-24*7:], linewidth='5',label='Real')
    plt.legend()
    # plt.savefig('sun_y.png', dpi=300)
    # plt.show()

    plt.figure(num =2, figsize=[20,10])
    plt.plot(Target2_y_predict_inverse[-24*7:], linewidth='5',label='Forecast')
    plt.plot(Target2_y_test_inverse[-24*7:], linewidth='5',label='Real')
    # plt.savefig('sun_y.png', dpi=300)
    plt.legend()
    # plt.show()

    plt.figure(num =3, figsize=[20,10])
    plt.plot(Target3_y_predict_inverse[-24*7:], linewidth='5',label='Forecast')
    plt.plot(Target3_y_test_inverse[-24*7:], linewidth='5',label='Real')
    plt.legend()
    # plt.savefig('eng_y.png', dpi=300)
    # plt.show()


    plt.figure(num =4, figsize=[20,10])
    plt.plot(Target4_y_predict_inverse[-24*7:], linewidth='5',label='Forecast')
    plt.plot(Target4_y_test_inverse[-24*7:], linewidth='5',label='Real')
    plt.legend()
    # plt.savefig('sol_y.png', dpi=300)
    # plt.show()

    plt.figure(num =5, figsize=[20,10])
    plt.plot(Target5_y_predict_inverse[-24*7:], linewidth='5',label='Forecast')
    plt.plot(Target5_y_test_inverse[-24*7:], linewidth='5',label='Real')
    plt.legend()
    # plt.savefig('sol_y.png', dpi=300)
    plt.show()