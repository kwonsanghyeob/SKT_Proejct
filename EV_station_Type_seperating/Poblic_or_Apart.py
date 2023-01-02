import numpy as np
import pandas


def divider(Zone_1, a):
    Poblic_Zone_1 = list()
    Apart_Zone_1 = list()
    for i in Zone_1:
        if a[a['충전소명']==i]['충전소구분'].values[0] == '공용' or a[a['충전소명']==i]['충전소구분'].values[0] == '업무용':
            Poblic_Zone_1.append(i)
        elif a[a['충전소명']==i]['충전소구분'].values[0] == '아파트용':
            Apart_Zone_1.append(i)
    if np.array(Zone_1).shape[0] != np.array(Poblic_Zone_1).shape[0]+np.array(Apart_Zone_1).shape[0]:
        raise Exception("분류 안된 충전소가 있습니다.")
    return Poblic_Zone_1, Apart_Zone_1
