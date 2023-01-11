import numpy as np


def Zone_sperate(data, Zone_Seperate_number=3):
    Zone_name = list()

    if Zone_Seperate_number == 2:
        for i in range(len(data['Zone'].unique())):
            Zone_name.append(np.array(data[data['Zone'] == data['Zone'].unique()[i]]['충전소명']))

    elif Zone_Seperate_number == 1:
        Zone_name.append(data)
    else:
        zone_name_set = data['Zone'].values
        for z in zone_name_set:
            k = data.loc[data['Zone'] == z, '충전소명'].values
            Zone_name.append(k[0].split(', '))

    return Zone_name
