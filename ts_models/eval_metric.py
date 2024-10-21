import os
import numpy as np
import pickle
# import torch
import random
from datetime import datetime
import torch
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, auc, roc_curve, precision_recall_curve
def eval_cus(y_classes, yhat_classes):
    """

    :param y_classes:  true labels
    :param yhat_classes: predicted labels
    :return:
    """
    # y_classes = np.argmax(y_test, axis=1)
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_classes, yhat_classes)
    # print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    preci = precision_score(y_classes, yhat_classes,average='weighted')
    # print('Precision: %f' % preci)
    # recall: tp / (tp + fn)
    recal = recall_score(y_classes, yhat_classes,average='weighted')
    # print('Recall: %f' % recal)
    # f1: 2 tp / (2 tp + fp + fn)
    mcc = matthews_corrcoef(y_classes,yhat_classes)
    # print('mcc: %f' % mcc)
    f1 = f1_score(y_classes, yhat_classes,average='weighted')
    # print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_classes, yhat_classes)

    # return accuracy, preci, recal, mcc, f1, kappa
    return {'acc':np.round(accuracy,3),'preci': np.round(preci,3),'recal':np.round(recal,3),'mcc':np.round(mcc,3) ,'f1':np.round(f1,3),'kappa':np.round(kappa,3)}