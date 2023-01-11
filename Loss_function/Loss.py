from keras import backend as K

def Loss_1(y_real, y_pred, e = 0.4):
    return e*K.mean((y_real-y_pred)**2) + (1-e)*K.max((y_real-y_pred)**2)