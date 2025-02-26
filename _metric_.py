import numpy as np



def WRMSPE(pred, true):
    numerator = np.sqrt(np.mean((pred - true) ** 2))
    denominator = np.mean(np.abs(true))
    return numerator/denominator

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def WAPE(pred, true):
    numerator = np.sum(np.abs(pred - true))
    denominator = np.sum(np.abs(true))
    return numerator/denominator

def sMAPE(pred, true, constant=200):
    return np.mean(constant * np.abs(pred - true) / (np.abs(pred) + np.abs(true)))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true + 1e-8))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true + 1e-8))


def metric(true, pred):
    rmse = RMSE(pred, true)
    wrmspe = WRMSPE(pred, true)
    mae = MAE(pred, true)
    wape = WAPE(pred, true)
    mse = MSE(pred, true)
    smape = sMAPE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return rmse.mean(), wrmspe.mean(), mae.mean(), wape.mean(), mse.mean(), smape.mean(), mape.mean(), mspe.mean()
   