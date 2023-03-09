import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from Data_preproessing.Model_Input_define_1 import Date_extraction, Onehot_encoding, Make_input, train_test
from EV_station_Type_seperating.Zone_sperate_code import Zone_sperate
from Evaluation_Index.NMAE import NMAE
from Forecasting_model.SKT_LSTM import Mymodel_LSTM
from Loss_function.Loss import Loss_1

warnings.filterwarnings('ignore')

EV_data = pd.read_csv('SKT_최종(이상치제거후)_개별충전소_시계열_한전_환경부_제주도청__210101_220622_23.01.13.csv', encoding='cp949')
# Zone_Seperate_1 = EV_data.columns[1:]
Zone_Seperate = pd.read_csv('SKT_충전소별_정보_통계량_한전_환경부_제주도청_210101_220622_23.01.17.csv', encoding='cp949')

a = pd.read_csv('SKT_월평균충전량_충전소_22.12.12.csv', encoding='cp949')
Extra_Feature_1 = pd.read_csv('SKT_충전소별_월별데이터존재유무_23.01.03.csv', encoding='cp949')
Extra_Feature_2 = pd.read_csv('SKT_시계열_고급휘발유가격.csv', encoding='cp949')

############################## 기본설정값 ################################
# Zone_Senario 1 제주전체, 2 (제주시, 서귀포쉬), 3 (변전소) 4.(읍면동)
Zone_Senario = 1
Selected_Zone = Zone_sperate(Zone_Seperate, Zone_Senario)
number_of_zone = len(Selected_Zone)  # Zone의 갯수
Mode = 1  # mode 1 휴일미포함, 2 휴일 포함
Zone_seperate = False  # Zone을 공용, 아파트용으로 나눠서 예측 할것인지를 판단
window_day = 14  # 배치사이즈 결정
train_day = 365  # 훈련데이터 수
val_day = 60  # 검증데이터 수
test_day = 60  # 테스트데이터 수
Epoch = 500  # 에폭
lr = 0.00002  # 훈련률

Month_predict = []
Day_predict = []

#########################Main코드######################################
Zone_name = Selected_Zone
for i in range(len(Zone_name)):
    globals()[f'Zone_target_{i}'] = Date_extraction(EV_data, Zone_name[i])


os.getcwd()
os.chdir(r'C:\\Users\\PESL_RTDS\\PycharmProjects\\SKT_Project\\Result\\Result_1')

result_1 = pd.read_csv('결과값_전체_0.csv')
result_2 = pd.read_csv('결과값_제주서귀포_0.csv')
result_3 = pd.read_csv('결과값_제주서귀포_1.csv')

print(NMAE(result_1['Real_Value'],result_1['Prediction Value']))
print(NMAE(result_2['Real_Value'],result_2['Prediction Value']))
print(NMAE(result_3['Real_Value'],result_3['Prediction Value']))






np.arrage(312)
plt.plot(Zone_target_0[(Zone_target_0['Unnamed: 0'] > '2021-12-25') & (Zone_target_0['Unnamed: 0'] < '2022-01-08')]['target'])
plt.show()