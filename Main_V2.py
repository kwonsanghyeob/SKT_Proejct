import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from Data_preproessing.Model_Input_define_1 import Date_extraction, Onehot_encoding, Make_input, train_test
from EV_station_Type_seperating.Poblic_or_Apart import divider
from EV_station_Type_seperating.Zone_sperate_code import Zone_sperate
from Evaluation_Index.NMAE import NMAE
from Forecasting_model.SKT_LSTM import Mymodel_LSTM
from Loss_function.Loss import Loss_1

warnings.filterwarnings('ignore')

EV_data = pd.read_csv('SKT_최종개별충전소_시계열_22.12.12_V2.csv', encoding='cp949')
Zone_Seperate_1 = EV_data.columns[1:]

Zone_Seperate_2 = pd.read_csv('SKT_Zone_1_제주시서귀포시.csv', encoding='cp949')
Zone_Seperate_3 = pd.read_csv('SKT_Zone_2_동서남북.csv', encoding='cp949')
Zone_Seperate_4 = pd.read_csv('SKT_Zone_3_읍면동.csv', encoding='cp949')
Zone_Seperate_5 = pd.read_csv('SKT_Zone_4_변전소.csv', encoding='cp949')

a = pd.read_csv('SKT_월평균충전량_충전소_22.12.12.csv', encoding='cp949')
Extra_Feature_1 = pd.read_csv('SKT_충전소별_월별데이터존재유무_23.01.03.csv', encoding='cp949')
Extra_Feature_2 = pd.read_csv('SKT_시계열_고급휘발유가격.csv', encoding='cp949')


############################## 기본설정값 ################################
Zone_Senario = 1
# for ZONE_SENARIO in Zone_Senario:
    #나중에는 바꿔야됨

Selected_Zone = Zone_sperate(globals()[f'Zone_Seperate_{Zone_Senario}'], Zone_Senario)
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
Zone_name = list()
if Zone_seperate:
    for i in range(number_of_zone):
        globals()[f'Pb_Zone_{i}'], globals()[f'AP_zone_{i}'] = divider(globals()[f'Zone_{i}'], a)
        Zone_name.append(globals()[f'Pb_Zone_{i}'])
        Zone_name.append(globals()[f'AP_zone_{i}'])

elif not Zone_seperate:
    Zone_name = Selected_Zone


for i in range(len(Zone_name)):
    # 입력데이터 정의
    globals()[f'Zone_target_{i}'] = Date_extraction(EV_data, Zone_name[i])



    globals()[f'Target_onehot_{i}'] = Onehot_encoding(globals()[f'Zone_target_{i}'], mode=Mode)
    globals()[f'Target_onehot_{i}']['oil'] = Extra_Feature_2
    # 모델 입력데이터 재배열
    globals()[f'Target_train_{i}'], globals()[f'Target_Val_{i}'], globals()[f'Target_Test_{i}'] = Make_input(np.array(globals()[f'Target_onehot_{i}']), Window_day=window_day, train_day=train_day, Val_day=val_day, test_day=test_day)
    globals()[f'MinMaxScaler_{i}'] = MinMaxScaler()
    globals()[f'MinMaxScaler_Traget_train_{i}'] = globals()[f'MinMaxScaler_{i}'].fit_transform(globals()[f'Target_train_{i}'])
    globals()[f'MinMaxScaler_Traget_Val_{i}'] = globals()[f'MinMaxScaler_{i}'].transform(globals()[f'Target_Val_{i}'])
    globals()[f'MinMaxScaler_Traget_test_{i}'] = globals()[f'MinMaxScaler_{i}'].transform(globals()[f'Target_Test_{i}'])

    # Train, Test 나누기
    globals()[f'x_Target_train_{i}'], globals()[f'y_Target_train_{i}'], globals()[f'x_Target_val_{i}'], globals()[f'y_Target_val_{i}'], globals()[f'x_Target_test_{i}'], globals()[f'y_Target_test_{i}'] = \
        train_test(globals()[f'MinMaxScaler_Traget_train_{i}'], globals()[f'MinMaxScaler_Traget_Val_{i}'], globals()[f'MinMaxScaler_Traget_test_{i}'], Window_day=window_day, train_day=train_day, Val_day=val_day, test_day=test_day)

    # 차원수 확인
    print(globals()[f'x_Target_train_{i}'].shape, globals()[f'y_Target_train_{i}'].shape, globals()[f'x_Target_val_{i}'].shape, globals()[f'y_Target_val_{i}'].shape, globals()[f'x_Target_test_{i}'].shape, globals()[f'y_Target_test_{i}'].shape)

    # 모델 훈련시작
    globals()[f'model_Target_{i}'] = Mymodel_LSTM()
    checkpointer = ModelCheckpoint(filepath=f'./tmp/weights_{i}', verbose=1, save_best_only=True, monitor='val_loss')
    globals()[f'model_Target_{i}'].compile(loss=Loss_1, optimizer=Adam(learning_rate=lr), metrics=['mae'])
    globals()[f'history_Target_{i}'] = globals()[f'model_Target_{i}'].fit(globals()[f'x_Target_train_{i}'], globals()[f'y_Target_train_{i}'], epochs=Epoch, validation_data=(globals()[f'x_Target_val_{i}'], globals()[f'y_Target_val_{i}']), callbacks=checkpointer)

    globals()[f'y_Target_predict_{i}'] = globals()[f'model_Target_{i}'].predict(globals()[f'x_Target_test_{i}'])

    globals()[f'y_Target_predict_inverse_{i}'] = globals()[f'MinMaxScaler_{i}'].inverse_transform(np.hstack((globals()[f'y_Target_predict_{i}'].reshape(-1, 1), np.zeros([globals()[f'y_Target_predict_{i}'].reshape(-1).shape[0], globals()[f'Target_Test_{i}'].shape[1] - 1]))))[:, 0]
    globals()[f'y_Target_test_inverse_{i}'] = globals()[f'MinMaxScaler_{i}'].inverse_transform(np.hstack((globals()[f'y_Target_test_{i}'].reshape(-1, 1), np.zeros([globals()[f'y_Target_test_{i}'].reshape(-1).shape[0], globals()[f'Target_Test_{i}'].shape[1] - 1]))))[:, 0]

result = []
for i in range(len(Zone_name)):
    result.append(NMAE(globals()[f'Target_Test_{i}'][:,0], globals()[f'y_Target_predict_inverse_{i}']))
print(result)


# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rc('font', size=15)  # 기본 폰트 크기
# plt.rc('axes', labelsize=15)  # x,y축 label 폰트 크기
# plt.rc('xtick', labelsize=15)  # x축 눈금 폰트 크기
# plt.rc('ytick', labelsize=15)  # y축 눈금 폰트 크기
# plt.rc('legend', fontsize=15)  # 범례 폰트 크기
# plt.rc('figure', titlesize=15)  # figure title 폰트 크기
#
# import matplotlib.pyplot as plt
# for i in range(len(Zone_name)):
#     plt.figure(figsize=(20,5))
#     plt.plot(globals()[f'Target_Test_{i}'][:,0])
#     plt.plot(globals()[f'y_Target_predict_inverse_{i}'])
#     plt.show()
#
# result = pd.DataFrame({'예측 값':globals()[f'Target_Test_{i}'][:,0], '실제값' : globals()[f'y_Target_predict_inverse_{i}']})
# result.to_csv('결과값.csv')
