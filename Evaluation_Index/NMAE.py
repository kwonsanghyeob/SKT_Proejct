import numpy as np

def NMAE(true, pred):
    '''
    true: np.array
    pred: np.array
    '''
    return np.mean(np.abs(true - pred) / (max(true) - min(true))) * 100