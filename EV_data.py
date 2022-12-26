import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from Model_Input_define.Input_preproceessing_1 import Date_extraction, Onehot_encoding
from Model_input_preprossing.Preprossing_1 import Make_input, train_test
from Forecasting_model.SKT_LSTM import Mymodel
import warnings

warnings.filterwarnings('ignore')

os.getcwd()
# os.chdir('/home/hy/PycharmProjects/sanghyeob_test/EV_forecasting')


a = pd.read_csv('SKT_Month_power_22.12.12.csv', encoding = 'cp949')
EV_data = pd.read_csv('SKT_Timeseries_station.csv', encoding = 'cp949')

#상위 충전소만 추출
#1. 종합경기장
#2. 종합경기장, LH제주본부, 신성로 공영주차장, 현일아파트
#3. 종합경기장, LH제주본부, 신성로 공영주차장, 현일아파트, 제주공항, 제주특별자치도복지이음마루, 정원파인즈10차
#4. 종합경기장, LH제주본부, 신성로 공영주차장, 현일아파트, 제주공항, 제주특별자치도복지이음마루, 정원파인즈10차, 제주도교육청, 제주직할, 베라체공영주차장



Zone1 = ['종합경기장']
Zone2 = ['종합경기장', 'LH제주본부', '신성로 공영주차장', '현일 아파트']
Zone3 = ['종합경기장', 'LH제주본부', '신성로 공영주차장', '현일 아파트', '제주공항', '제주특별자치도 복지이음마루', '정원파인즈 10차']
Zone4 = ['종합경기장', 'LH제주본부', '신성로 공영주차장', '현일 아파트', '제주공항', '제주특별자치도 복지이음마루', '정원파인즈 10차', '제주도교육청', '제주직할', '베라체 공영주차장']
Zone5 = EV_data.columns[1:]

Zone1_target = Date_extraction(EV_data, Zone1)
Zone2_target = Date_extraction(EV_data, Zone2)
Zone3_target = Date_extraction(EV_data, Zone3)
Zone4_target = Date_extraction(EV_data, Zone4)
Zone5_target = Date_extraction(EV_data, Zone5)

Target_onehot_1= Onehot_encoding(Zone1_target)
Target_onehot_2= Onehot_encoding(Zone2_target)
Target_onehot_3= Onehot_encoding(Zone3_target)
Target_onehot_4= Onehot_encoding(Zone5_target)
Target_onehot_5= Onehot_encoding(Zone5_target)


Target1_train, Target1_Test = Make_input(np.array(Target_onehot_1))
Target2_train, Target2_Test = Make_input(np.array(Target_onehot_2))
Target3_train, Target3_Test = Make_input(np.array(Target_onehot_3))
Target4_train, Target4_Test = Make_input(np.array(Target_onehot_4))
Target5_train, Target5_Test = Make_input(np.array(Target_onehot_5))


MinMaxScaler_1 = MinMaxScaler()
MinMaxScaler_2 = MinMaxScaler()
MinMaxScaler_3 = MinMaxScaler()
MinMaxScaler_4 = MinMaxScaler()
MinMaxScaler_5 = MinMaxScaler()

MinMaxScaler_1_Traget1_train = MinMaxScaler_1.fit_transform(Target1_train)
MinMaxScaler_2_Traget2_train = MinMaxScaler_2.fit_transform(Target2_train)
MinMaxScaler_3_Traget3_train = MinMaxScaler_3.fit_transform(Target3_train)
MinMaxScaler_4_Traget4_train = MinMaxScaler_4.fit_transform(Target4_train)
MinMaxScaler_5_Traget5_train = MinMaxScaler_5.fit_transform(Target5_train)


MinMaxScaler_1_Traget1_test = MinMaxScaler_1.transform(Target1_Test)
MinMaxScaler_2_Traget2_test = MinMaxScaler_2.transform(Target2_Test)
MinMaxScaler_3_Traget3_test = MinMaxScaler_3.transform(Target3_Test)
MinMaxScaler_4_Traget4_test = MinMaxScaler_4.transform(Target4_Test)
MinMaxScaler_5_Traget5_test = MinMaxScaler_5.transform(Target5_Test)

x_Traget1_train, y_Traget1_train, x_Traget1_test, y_Traget1_test = train_test(MinMaxScaler_1_Traget1_train,MinMaxScaler_1_Traget1_test)
x_Traget2_train, y_Traget2_train, x_Traget2_test, y_Traget2_test = train_test(MinMaxScaler_2_Traget2_train,MinMaxScaler_2_Traget2_test)
x_Traget3_train, y_Traget3_train, x_Traget3_test, y_Traget3_test = train_test(MinMaxScaler_3_Traget3_train,MinMaxScaler_3_Traget3_test)
x_Traget4_train, y_Traget4_train, x_Traget4_test, y_Traget4_test = train_test(MinMaxScaler_4_Traget4_train,MinMaxScaler_4_Traget4_test)
x_Traget5_train, y_Traget5_train, x_Traget5_test, y_Traget5_test = train_test(MinMaxScaler_5_Traget5_train,MinMaxScaler_5_Traget5_test)

print(x_Traget1_train.shape, y_Traget1_train.shape, x_Traget1_test.shape, y_Traget1_test.shape)
print(x_Traget2_train.shape, y_Traget2_train.shape, x_Traget2_test.shape, y_Traget2_test.shape)
print(x_Traget3_train.shape, y_Traget3_train.shape, x_Traget3_test.shape, y_Traget3_test.shape)
print(x_Traget4_train.shape, y_Traget4_train.shape, x_Traget4_test.shape, y_Traget4_test.shape)
print(x_Traget5_train.shape, y_Traget5_train.shape, x_Traget5_test.shape, y_Traget5_test.shape)

model_Traget1 = Mymodel()
model_Traget2 = Mymodel()
model_Traget3 = Mymodel()
model_Traget4 = Mymodel()
model_Traget5 = Mymodel()


model_Traget1.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.00002), metrics=['mae'])
model_Traget2.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.00002), metrics=['mae'])
model_Traget3.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.00002), metrics=['mae'])
model_Traget4.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.00002), metrics=['mae'])
model_Traget5.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.00002), metrics=['mae'])

print(x_Traget1_train.shape)
print(y_Traget2_train.shape)

history_Target1 = model_Traget1.fit(x_Traget1_train, y_Traget1_train, epochs=150)
history_Target2 = model_Traget2.fit(x_Traget2_train, y_Traget2_train, epochs=150)
history_Target3 = model_Traget3.fit(x_Traget3_train, y_Traget3_train, epochs=150)
history_Target4 = model_Traget4.fit(x_Traget4_train, y_Traget4_train, epochs=150)
history_Target5 = model_Traget5.fit(x_Traget5_train, y_Traget5_train, epochs=150)


Target1_y_predict = model_Traget1.predict(x_Traget1_test)
Target2_y_predict = model_Traget2.predict(x_Traget2_test)
Target3_y_predict = model_Traget3.predict(x_Traget3_test)
Target4_y_predict = model_Traget4.predict(x_Traget4_test)
Target5_y_predict = model_Traget5.predict(x_Traget5_test)


Target1_y_predict_inverse = MinMaxScaler_1.inverse_transform(np.hstack((Target1_y_predict.reshape(-1,1), np.zeros([Target1_y_predict.reshape(-1).shape[0], Target1_train.shape[1]-1]))))[:,0]
Target2_y_predict_inverse = MinMaxScaler_2.inverse_transform(np.hstack((Target2_y_predict.reshape(-1,1), np.zeros([Target2_y_predict.reshape(-1).shape[0], Target2_train.shape[1]-1]))))[:,0]
Target3_y_predict_inverse = MinMaxScaler_3.inverse_transform(np.hstack((Target3_y_predict.reshape(-1,1), np.zeros([Target3_y_predict.reshape(-1).shape[0], Target3_train.shape[1]-1]))))[:,0]
Target4_y_predict_inverse = MinMaxScaler_4.inverse_transform(np.hstack((Target4_y_predict.reshape(-1,1), np.zeros([Target4_y_predict.reshape(-1).shape[0], Target4_train.shape[1]-1]))))[:,0]
Target5_y_predict_inverse = MinMaxScaler_5.inverse_transform(np.hstack((Target5_y_predict.reshape(-1,1), np.zeros([Target5_y_predict.reshape(-1).shape[0], Target5_train.shape[1]-1]))))[:,0]


Target1_y_test_inverse = MinMaxScaler_1.inverse_transform(np.hstack((y_Traget1_test.reshape(-1,1), np.zeros([y_Traget1_test.reshape(-1).shape[0], Target1_Test.shape[1]-1]))))[:,0]
Target2_y_test_inverse = MinMaxScaler_2.inverse_transform(np.hstack((y_Traget2_test.reshape(-1,1), np.zeros([y_Traget2_test.reshape(-1).shape[0], Target2_Test.shape[1]-1]))))[:,0]
Target3_y_test_inverse = MinMaxScaler_3.inverse_transform(np.hstack((y_Traget3_test.reshape(-1,1), np.zeros([y_Traget3_test.reshape(-1).shape[0], Target3_Test.shape[1]-1]))))[:,0]
Target4_y_test_inverse = MinMaxScaler_4.inverse_transform(np.hstack((y_Traget4_test.reshape(-1,1), np.zeros([y_Traget4_test.reshape(-1).shape[0], Target4_Test.shape[1]-1]))))[:,0]
Target5_y_test_inverse = MinMaxScaler_5.inverse_transform(np.hstack((y_Traget5_test.reshape(-1,1), np.zeros([y_Traget5_test.reshape(-1).shape[0], Target5_Test.shape[1]-1]))))[:,0]


plt.rcParams['font.family'] = 'Times New Roman'
plt.rc('font', size=40)        # 기본 폰트 크기
plt.rc('axes', labelsize=20)   # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=20)  # x축 눈금 폰트 크기
plt.rc('ytick', labelsize=20)  # y축 눈금 폰트 크기
plt.rc('legend', fontsize=20)  # 범례 폰트 크기
plt.rc('figure', titlesize=20) # figure title 폰트 크기


plt.figure(num =1, figsize=[20,10])
plt.plot(Target1_y_predict[0])
plt.plot(y_Traget1_test[3])
# plt.savefig('sun_y.png', dpi=300)
plt.show()



plt.figure(num =1, figsize=[20,10])
plt.plot(Target1_y_predict_inverse[:24*7], linewidth='5')
plt.plot(Target1_y_test_inverse[:24*7], linewidth='5')
# plt.savefig('sun_y.png', dpi=300)
plt.show()

plt.figure(num =2, figsize=[20,10])
plt.plot(Target2_y_predict_inverse[:24*7], linewidth='5')
plt.plot(Target2_y_test_inverse[:24*7], linewidth='5')
# plt.savefig('sun_y.png', dpi=300)
plt.show()

plt.figure(num =3, figsize=[20,10])
plt.plot(Target3_y_predict_inverse[:24*7], linewidth='5')
plt.plot(Target3_y_test_inverse[:24*7], linewidth='5')
# plt.savefig('eng_y.png', dpi=300)
plt.show()


plt.figure(num =4, figsize=[20,10])
plt.plot(Target4_y_predict_inverse[:24*7], linewidth='5')
plt.plot(Target4_y_test_inverse[:24*7], linewidth='5')
# plt.savefig('sol_y.png', dpi=300)
plt.show()

plt.figure(num =5, figsize=[20,10])
plt.plot(Target5_y_predict_inverse[:24*30], linewidth='3')
plt.plot(Target5_y_test_inverse[:24*30], linewidth='3')
# plt.savefig('sol_y.png', dpi=300)
plt.show()

#
def NMAE(true, pred):
    '''
    true: np.array
    pred: np.array
    '''
    return np.mean(np.abs(true-pred)/(max(true)-min(true)))*100

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
