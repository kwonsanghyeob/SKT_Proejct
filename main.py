import warnings

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from Data_preproessing.Model_Input_define_1 import Date_extraction, Onehot_encoding, Make_input, train_test
from EV_station_Type_seperating.Poblic_or_Apart import divider
from Evaluation_Index.NMAE import NMAE
from Forecasting_model.SKT_LSTM import Mymodel_LSTM
from Loss_function.Loss import Loss_1

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    a = pd.read_csv('SKT_월평균충전량_충전소_22.12.12.csv', encoding='cp949')
    EV_data = pd.read_csv('SKT_최종개별충전소_시계열_22.12.12.csv', encoding='cp949')

    Zone_0 = ['제주 동부보건소', '스마트그리드 홍보관', '구좌119센터', '종달리 종합복지회관', '동부농업기술센타', '제주해녀박물관',
              '테크노파크 용암해수센터', '조천119센터', '에덴빌리지2차', '베스트힐 주차장(조천읍)', '물영아리주차장', '남원공영주차장',
              '남원 의례회관 주차장', '서귀포남원LH 아파트', '성산하수처리장', '수산1리 복지회관', '동부소방서', '김영갑갤러리',
              '난산리 다목적회관', '표선 해양경비단', '가시리 조랑말체험공원', '표선공영주차장']

    Zone_1 = ['한림체육관', '김창열미술관', '서부소방서', '서부농업기술센터', '제주한림LH 아파트', '한수풀도서관', '생각하는정원',
              '유리의성', '한경농협 하나로마트', '한경119센터', '저지리사무소', '대정읍사무소', '영어교육도시119센터', '마라도 가는 여객선',
              '서귀포대정LH 아파트', '용머리해안', '안덕계곡 주차장', '안덕생활체육관', '건강과 성박물관', '방주교회', '서커스월드공연장',
              '동광리버스차고지', '청소년 문화의 집', '안덕면종합복지회관', '토이파크', '예래동주민센터']

    Zone_2 = ['제주국제평화센터', '영실매표소', '정방폭포', '천지공영주차장', '서귀포시교육지원청', '번개과학체험관', '서귀포개인택시조합',
              '제주국가생약자원관리센터', '서귀포의료원', '서귀포향토오일장', '제주서귀포동홍6단지LH 아파트', '서귀포동홍3LH 아파트',
              '서귀포지사', '서귀포해양경찰서', '서귀포혁신LH2단지', '서귀포혁신LH1단지', '강창학경기장', '농업기술원', '대천동주민센터',
              '강정상록 아파트', '강정생명평화교회']

    Zone_3 = ['LH제주본부', '현일 아파트', '건입현대 아파트', '국립제주박물관', '제주해양경찰서', '제주 교육대학교', '화북4 아파트',
              '화북1 아파트', '화북주공2 아파트', '화북2동 공영주차장', '제주삼화1단지LH 아파트', '제주화북3LH 아파트', '제주 삼양 유적지',
              '삼양2동 공영주차장', '농어촌공사 제주지역본부', '제주삼화3단지LH 아파트', '제주도련LH 아파트', '동물위생시험소(동물보호센터)',
              '제주대학교', '아라원신 아파트', '국제대학교 버스차고지', '아라아이파크', '제주의료원', '제주산학융합원',
              '한라산국립공원 관음사탐방안내소', '제주아라스위첸 아파트', '제주아라LH 아파트', '제주경찰교육센터', '종합경기장',
              '제주아트센터', '제주탐라교육원', '제주특별자치도 복지이음마루', '제주교도소']

    Zone_4 = ['제주공항', '행복날개 주유소', '현대3차 아파트', '공항입구 공영주차장', '제주도교육청', '제주직할', '연동대림1차아파트', '연동한일시티',
              '노형 제2공영주차장', '웅전공영주차장', '아트리움공연장', '현대3노형부영5차 아파트', '제주정든마을1단지LH 아파트', '노형2차부영 아파트',
              '제주정든마을3단지LH 아파트', '한라산국립공원 어리목탐방안내소', '노형지구중흥S클래스 아파트', '제일일출연립주택', '제주부영1차 아파트',
              '제주으뜸마을LH 아파트', '도로교통공단 제주지부', '해오름 아파트']

    Zone_5 = ['애월119센터', '서부경찰서', '유수암', '새마을금고제주연수원', '제주고성LH 아파트', '제주하귀휴먼시아1단지', '제주하귀휴먼시아2단지',
              '고스트타운', '용흥리사무소', '제주소년원', '제주외도아름마을LH 아파트', '제주수산연구소']

    number_of_zone = 6  # Zone의 갯수
    Mode = 3  # mode 1 휴일미포함, 2 휴일 포함, 3 각각 모델로 설계
    Zone_seperate = True  # Zone을 공용, 아파트용으로 나눠서 예측 할것인지를 판단
    window_day = 14  # 배치사이즈 결정
    train_day = 681  # 훈련데이터 수
    val_day = 120  # 검증데이터 수
    test_day = 61  # 테스트데이터 수
    Epoch = 150  # 에폭
    lr = 0.00002  # 훈련률

    window_weekday = 10
    train_weekday = 440
    val_weekday = 64
    test_weekday = 64

    train_weekend = 238
    val_weekend = 26
    test_weekend = 27

    #########################Main코드#####################################
    Zone_name = list()
    if Zone_seperate:
        for i in range(number_of_zone):
            globals()[f'Pb_Zone_{i}'], globals()[f'AP_zone_{i}'] = divider(globals()[f'Zone_{i}'], a)
            Zone_name.append(globals()[f'Pb_Zone_{i}'])
            Zone_name.append(globals()[f'AP_zone_{i}'])
    elif Zone_seperate:
        for i in range(number_of_zone):
            Zone_name.append(globals()[f'Zone_{i}'])

    if Mode != 3:
        for i in range(len(Zone_name)):
            # 입력데이터 정의
            globals()[f'Zone_target_{i}'] = Date_extraction(EV_data, Zone_name[i])
            globals()[f'Target_onehot_{i}'] = Onehot_encoding(globals()[f'Zone_target_{i}'], mode=Mode)

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
            result.append(NMAE(globals()[f'y_Target_test_inverse_{i}'], globals()[f'y_Target_predict_inverse_{i}']))
        print(result)

    if Mode == 3:
        for i in range(len(Zone_name)):
            # 입력데이터 정의
            globals()[f'Zone_target_{i}'] = Date_extraction(EV_data, Zone_name[i])
            globals()[f'Target_onehot_{i}'] = Onehot_encoding(globals()[f'Zone_target_{i}'], mode=Mode)

            # 요일 나누기
            globals()[f'Target_onehot_weekday_{i}'] = globals()[f'Target_onehot_{i}'][globals()[f'Target_onehot_{i}']['check_weekend'] == 0]
            globals()[f'Target_onehot_weekday_{i}'] = globals()[f'Target_onehot_weekday_{i}'].loc[:, globals()[f'Target_onehot_weekday_{i}'].columns != 'check_weekend']
            globals()[f'Target_onehot_weekend_{i}'] = globals()[f'Target_onehot_{i}'][globals()[f'Target_onehot_{i}']['check_weekend'] == 1]
            globals()[f'Target_onehot_weekend_{i}'] = globals()[f'Target_onehot_weekend_{i}'].loc[:, globals()[f'Target_onehot_weekend_{i}'].columns != 'check_weekend']

            #
            globals()[f'Target_train_week_{i}'], globals()[f'Target_val_week_{i}'], globals()[f'Target_test_week_{i}'] = Make_input(np.array(globals()[f'Target_onehot_{i}']), Window_day=window_weekday, train_day=train_weekday, Val_day=val_weekday, test_day=test_weekday)
            globals()[f'Target_train_end_{i}'], globals()[f'Target_val_end_{i}'], globals()[f'Target_test_end_{i}'] = Make_input(np.array(globals()[f'Target_onehot_{i}']), Window_day=window_weekday, train_day=train_weekday, Val_day=val_weekday, test_day=test_weekday)

            globals()[f'MinMaxScaler_week_{i}'] = MinMaxScaler()
            globals()[f'MinMaxScaler_end_{i}'] = MinMaxScaler()

            globals()[f'MinMaxScaler_week_Target_train_{i}'] = globals()[f'MinMaxScaler_week_{i}'].fit_transform(globals()[f'Target_train_week_{i}'])
            globals()[f'MinMaxScaler_end_Target_train_{i}'] = globals()[f'MinMaxScaler_end_{i}'].fit_transform(globals()[f'Target_train_end_{i}'])
            globals()[f'MinMaxScaler_week_Target_val_{i}'] = globals()[f'MinMaxScaler_week_{i}'].transform(globals()[f'Target_val_week_{i}'])
            globals()[f'MinMaxScaler_end_Target_val_{i}'] = globals()[f'MinMaxScaler_end_{i}'].transform(globals()[f'Target_val_end_{i}'])
            globals()[f'MinMaxScaler_week_Target_val_{i}'] = globals()[f'MinMaxScaler_week_{i}'].transform(globals()[f'Target_test_week_{i}'])
            globals()[f'MinMaxScaler_end_Target_test_{i}'] = globals()[f'MinMaxScaler_end_{i}'].transform(globals()[f'Target_test_end_{i}'])

            globals()[f'x_Target_week_train_{i}'], globals()[f'y_Target_week_train_{i}'], globals()[f'x_Target_week_val_{i}'], globals()[f'y_Target_week_val_{i}'], globals()[f'x_Target_week_test_{i}'], globals()[f'y_Target_week_test_{i}'] = train_test(
                globals()[f'MinMaxScaler_week_Target_train_{i}'],
                globals()[f'MinMaxScaler_week_Target_val_{i}'],
                globals()[f'MinMaxScaler_week_Target_val_{i}'],
                Window_day=window_weekday, train_day=train_weekday,
                Val_day=val_weekday, test_day=test_weekday)

            globals()[f'x_Target_end_train_{i}'], globals()[f'y_Target_end_train_{i}'], globals()[f'x_Target_end_val_{i}'], globals()[f'y_Target_end_val_{i}'], globals()[f'x_Target_end_test_{i}'], globals()[f'y_Target_end_test_{i}'] = train_test(globals()[f'MinMaxScaler_end_Target_train_{i}'],
                                                                                                                                                                                                                                                      globals()[f'MinMaxScaler_end_Target_val_{i}'],
                                                                                                                                                                                                                                                      globals()[f'MinMaxScaler_end_Target_val_{i}'],
                                                                                                                                                                                                                                                      Window_day=window_weekday, train_day=train_weekend,
                                                                                                                                                                                                                                                      Val_day=val_weekend, test_day=test_weekend)

            globals()[f'model_Traget_week_{i}'] = Mymodel_LSTM()
            globals()[f'model_Traget_end_{i}'] = Mymodel_LSTM()

            checkpointer_week = ModelCheckpoint(filepath=f'./tmp/weights_week_{i}', verbose=1, save_best_only=True, monitor='val_loss')
            checkpointer_end = ModelCheckpoint(filepath=f'./tmp/weights_end_{i}', verbose=1, save_best_only=True, monitor='val_loss')

            globals()[f'model_Traget_week_{i}'].compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr), metrics=['mae'])
            globals()[f'model_Traget_end_{i}'].compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr), metrics=['mae'])

            globals()[f'history_Target_week_{i}'] = globals()[f'model_Traget_week_{i}'].fit(globals()[f'x_Target_week_train_{i}'], globals()[f'y_Target_week_train_{i}'], epochs=Epoch, validation_data=(globals()[f'x_Target_week_val_{i}'], globals()[f'y_Target_week_val_{i}']),
                                                                                            callbacks=checkpointer_week)
            globals()[f'history_Target_end_{i}'] = globals()[f'model_Traget_end_{i}'].fit(globals()[f'x_Target_end_train_{i}'], globals()[f'y_Target_end_train_{i}'], epochs=Epoch, validation_data=(globals()[f'x_Target_end_val_{i}'], globals()[f'y_Target_end_val_{i}']), callbacks=checkpointer_end)

            globals()[f'y_Target_week_predict_{i}'] = globals()[f'model_Traget_week_{i}'].predict(globals()[f'x_Target_week_test_{i}'])
            globals()[f'y_Target_end_predict_{i}'] = globals()[f'model_Traget_end_{i}'].predict(globals()[f'x_Target_end_test_{i}'])

            globals()[f'y_Target_predict_week_inverse_{i}'] = globals()[f'MinMaxScaler_week_{i}'].inverse_transform(
                np.hstack((globals()[f'y_Target_week_predict_{i}'].reshape(-1, 1), np.zeros([globals()[f'y_Target_week_predict_{i}'].reshape(-1).shape[0], globals()[f'Target_test_week_{i}'].shape[1] - 1]))))[:, 0]
            globals()[f'y_Target_predict_end_inverse_{i}'] = globals()[f'MinMaxScaler_end_{i}'].inverse_transform(
                np.hstack((globals()[f'y_Target_end_predict_{i}'].reshape(-1, 1), np.zeros([globals()[f'y_Target_end_predict_{i}'].reshape(-1).shape[0], globals()[f'Target_test_end_{i}'].shape[1] - 1]))))[:, 0]

            globals()[f'y_Target_test_week_inverse_{i}'] = globals()[f'MinMaxScaler_week_{i}'].inverse_transform(np.hstack((globals()[f'y_Target_week_test_{i}'].reshape(-1, 1), np.zeros([globals()[f'y_Target_week_test_{i}'].reshape(-1).shape[0], globals()[f'Target_test_week_{i}'].shape[1] - 1]))))[
                                                           :, 0]
            globals()[f'y_Target_test_end_inverse_{i}'] = globals()[f'MinMaxScaler_end_{i}'].inverse_transform(np.hstack((globals()[f'y_Target_end_test_{i}'].reshape(-1, 1), np.zeros([globals()[f'y_Target_end_test_{i}'].reshape(-1).shape[0], globals()[f'Target_test_end_{i}'].shape[1] - 1]))))[:, 0]

        result = []
        for i in range(number_of_zone):
            result.append(NMAE(globals()[f'y_Target_test_week_inverse_{i}'], globals()[f'y_Target_predict_week_inverse_{i}']))
            result.append(NMAE(globals()[f'y_Target_predict_end_inverse_{i}'], globals()[f'y_Target_test_end_inverse_{i}']))
        print(result)
