import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def WMAPE(pred,true):
    pred = pred.ravel()
    true = true.ravel()

    weight = np.abs(true) / np.sum(np.abs(true))
    weight_res = np.abs((pred - true) / true) * weight
    return np.sum(weight_res)
def calculate_wmape(pred, true, weights=None):
    """
    计算加权平均绝对百分比误差（WMAPE）

    参数：
    pred: numpy数组，预测值，维度为[128, 5, 1]
    true: numpy数组，真实值，维度为[128, 5, 1]
    weights: numpy数组，权重，维度为[128, 5, 1]，默认为均等权重

    返回：
    wmape: float，加权平均绝对百分比误差
    """
    if weights is None:
        weights = np.ones_like(pred)

    numerator = np.sum(np.abs(pred - true) * weights)
    denominator = np.sum(np.abs(true) * weights)

    # wmape = numerator / denominator * 100
    wmape = numerator / denominator
    return wmape

import torch
class MAPE_Fund():
    def __init__(self, args=None):
        self.args=args
        self.li_loss=torch.nn.L1Loss()
    def cal_fund_val(self,pred, true):
        pred=pred.detach().cpu().numpy()
        true=true.detach().cpu().numpy()
        return calculate_wmape(pred, true)

def metric(pred, true):

    metric_dict={}
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    # metric_dict['wmape']=WMAPE(pred,true)
    metric_dict['wmape']=calculate_wmape(pred,true)
    metric_dict['mae']=mae
    metric_dict['mse'] = mse
    metric_dict['rmse'] = rmse
    metric_dict['mape'] = mape
    metric_dict['mspe'] = mspe
    metric_dict['rse'] = rse
    metric_dict['corr'] = np.mean(corr)

    return mae, mse, rmse, mape, mspe, rse, corr,metric_dict
