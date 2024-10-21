import time

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


def MAPE_fund(pred, true,args=None):
    fund_metric = {}
    me_name = ['apply', 'redeem']
    for k in range(pred.shape[-1]):
        fund_metric[me_name[k]] = calculate_wmape(np.expand_dims(pred[:, :, k], axis=-1),
                                                  np.expand_dims(true[:, :, k], axis=-1)) * 100

    fund_metric['mean']=(fund_metric['apply']+fund_metric['redeem'])/2
    fund_metric['sum'] =fund_metric['apply'] + fund_metric['redeem']
    return fund_metric

    fund_metric={}
    metric_name=['apply','redeem','netin']

    if args.model == 'DeepAr' and args.mode == 'univariate' or args.features=='S':
        # print(pred.shape,true.shape)
        # time.sleep(500)
        pred_all = [pred]
        true_all = [true]
    else:

        pred_apply=pred[:,:,0]
        true_apply=true[:,:,0]

        pred_redeem= pred[:, :, 1]
        true_redeem= true[:, :, 1]

        if args.cal_net_in:
            if args.dived:
                pred_netin=pred_apply-pred_redeem
            else:
                pred_netin = pred[:, :, 2]
            pred_netin = pred_apply - pred_redeem
            true_netin = true[:, :, 2]

            pred_all=[pred_apply,pred_redeem,pred_netin]
            true_all=[true_apply,true_redeem,true_netin]
        else:
            pred_all=[pred_apply,pred_redeem]
            true_all=[true_apply,true_redeem]
            fund_metric['netin']=0

    for i, (pred, true) in enumerate(list(zip(pred_all, true_all))):
        pred=pred.ravel()
        true=true.ravel()

        weight=np.abs(true)/np.sum(np.abs(true))
        weight_res=np.abs((pred - true) / true)*weight

        fund_metric[metric_name[i]]=np.sum(weight_res)

    fund_metric['mean']=(fund_metric['apply']+fund_metric['redeem']+fund_metric['netin'])/2
    fund_metric['sum'] =fund_metric['apply'] + fund_metric['redeem'] + fund_metric['netin']
    for k,v in fund_metric.items():
        fund_metric[k]=np.round(v*100,2)
    return fund_metric
import torch

def calculate_wmape(pred, true, weights=None):

    if weights is None:
        weights = np.ones_like(pred)

    numerator = np.sum(np.abs(pred - true) * weights)
    denominator = np.sum(np.abs(true) * weights)

    # wmape = numerator / denominator * 100
    wmape = numerator / denominator
    return wmape
def WMAPE(pred,true):

    pred = pred.ravel()
    true = true.ravel()

    weight = np.abs(true) / np.sum(np.abs(true))
    weight_res = np.abs((pred - true) /true) * weight
    return np.sum(weight_res)
class MAPE_Fund():
    def __init__(self, args=None):
        self.args=args
        self.li_loss=torch.nn.L1Loss()
    def cal_fund_val(self,pred, true):
        fund_metric = {}
        if 'Fund' in self.args.data:
            mae = MAE(pred, true)
            mse = MSE(pred, true)
            rmse = RMSE(pred, true)
            corr = CORR(pred, true)
            rse = RSE(pred, true)
            # pmae = PMAE(pred, true)
            wmape = calculate_wmape(pred, true)
            fund_metric['mae'] = mae
            fund_metric['mse'] = mse
            fund_metric['rmse'] = rmse
            fund_metric['corr'] = np.mean(corr)
            fund_metric['rse'] = rse
            # fund_metric['pmae'] = pmae
            fund_metric['wmape'] = wmape
            me_name=['apply','redeem']
            for k in range(pred.shape[-1]):
                fund_metric[me_name[k]]=calculate_wmape(np.expand_dims(pred[:,:,k],axis=-1),np.expand_dims(true[:,:,k],axis=-1))*100
            fund_metric['sum']=fund_metric['apply'] + fund_metric['redeem']
            # print(fund_metric)
            # time.sleep(500)
        else:
            mae = MAE(pred, true)
            mse = MSE(pred, true)
            rmse = RMSE(pred, true)
            corr = CORR(pred, true)
            rse = RSE(pred, true)
            # pmae = PMAE(pred, true)
            wmape=calculate_wmape(pred,true)
            fund_metric['mae']=mae
            fund_metric['mse']=mse
            fund_metric['rmse']=rmse
            fund_metric['corr']=np.mean(corr)
            fund_metric['rse']=rse
            # fund_metric['pmae'] = pmae
            fund_metric['wmape']=wmape

        for k, v in fund_metric.items():
            # print(type(v))
            # fund_metric[k] = float(np.round(np.float32(v),5))

            fund_metric[k] =float(v)

        return fund_metric


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):

    return np.mean(np.square((pred - true) / true))


def PMAE(pred,true,args=None):
    abs_error = np.abs(pred - true)

    pmae = (abs_error / true) * 100
    # print(pmae.shape)
    # inf_indices = np.where(np.any(np.isinf(pmae), axis=(1, 2)))
    inf_indices=np.any(np.isinf(pmae), axis=(1, 2))
    pmae=pmae[~inf_indices]
    # if 'Fund' in args.data_type:
    #     condition = np.any(pmae[:, :, :] > 200, axis=(1, 2))
    #     pmae=pmae[~condition]
    mean_pmae = np.mean(pmae)
    return mean_pmae
def metric(pred, true,args=None):
    # np.save(args.path+'/'+'test_pred.npy',pred)
    # np.save(args.path + '/' + 'true_pred.npy', true)

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred,true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    pmae = PMAE(pred, true)
    corr = CORR(pred, true)
    mape_fund = MAPE_fund(pred, true, args)

    mape_fund['mae']=mae
    mape_fund['mse']=mse
    mape_fund['rmse']=rmse
    mape_fund['rse']=rse
    mape_fund['pmae'] = pmae
    for k,v in mape_fund.items():

        mape_fund[k]=np.float(v)

    print('******',pred.shape,true.shape)
    return mae, mse, rmse, mape, mspe, rse, corr,mape_fund
