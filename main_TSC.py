import time
import numpy as np
import random
from ts_models import datautils
from ts_models.Easy_use_trainner import Easy_use_trainner
import os
from os.path import dirname
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from ts_models.OmniScaleCNN import OmniScaleCNN
from ts_models.InceptionNet import InceptionNet
from ts_models.FormerTime import FormerTime
# from utils_log import *
import json
from ts_models.FCN_ResNet import FCN, ResNet

device = "cuda:1"

def get_files(path):
    with open(path, mode='r', encoding='utf-8') as file:
        data = file.read()
        data_dict = json.loads(data)
    return data_dict
# example_data_set_list = ['BME','CBF','ECG200','FiftyWords', 'ShapeletSim']

parser = argparse.ArgumentParser()
# 不加--会报错required
parser.add_argument('--dataset', default='GunPoint', help='The dataset name')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_rate', type=float, default=1.)
parser.add_argument('--lr_decay_steps', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0.01)

# FormerTime args
parser.add_argument('--stages', type=int, default=3)
parser.add_argument('--layer_per_stage', type=int, default=[6, 6, 6])
parser.add_argument('--hidden_size_per_stage', type=list, default=[64, 64, 64])
parser.add_argument('--slice_per_stage', type=list, default=[16, 2, 2])  # changes according to the dataset
parser.add_argument('--stride_per_stage', type=int, default=[8, 2, 2])
parser.add_argument('--tr', type=list, default=[2, 1, 1])  # temporal reduction ratio
parser.add_argument('--position_location', type=str, default='top', choices=['top', 'middle'])
parser.add_argument('--position_type', type=str, default='cond',
                    choices=['cond', 'relative', 'static', 'none', 'conv_static'])

parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--attn_heads', type=int, default=4)
parser.add_argument('--eval_per_steps', type=int, default=16)
parser.add_argument('--enable_res_parameter', type=int, default=1)
parser.add_argument('--loss', type=str, default='ce', choices=['bce', 'ce'])
parser.add_argument('--pooling_type', type=str, default='mean', choices=['mean', 'max', 'last_token', 'cat'])
parser.add_argument('--save_model', type=int, default=1)

parser.add_argument('--cost_type', default='cosine', type=str, help='cosine | dotprod | euclidean')
parser.add_argument('--pooling_op', default='max', type=str, help='avg | sum | max')
parser.add_argument('--n_segments', default=4, type=int, help='# of segments')
parser.add_argument('--gamma', default=1.0, type=float, help='smoothing for differentiable DTW')

args = parser.parse_args()
def pad_sequeence(original_array):
    # target_shape = (original_array.shape[0], 100, 1)
    original_array=original_array.squeeze(-1)
    padded_arr = np.pad(original_array, ((0, 0), (0, 60-original_array.shape[1])), mode='constant', constant_values=0)
    return np.expand_dims(padded_arr,axis=-1)

UCR_datasets_all=['SyntheticControl', 'HouseTwenty', 'SonyAIBORobotSurface2', 'SemgHandGenderCh2', 'ProximalPhalanxOutlineCorrect', 'Rock', 'DodgerLoopGame', 'BME', 'SonyAIBORobotSurface1', 'ToeSegmentation2', 'Beef', 'ArrowHead', 'DodgerLoopWeekend', 'ProximalPhalanxTW', 'Trace', 'Haptics', 'ToeSegmentation1', 'OSULeaf', 'CricketZ', 'Strawberry', 'Ham', 'PigArtPressure', 'ECG200', 'GestureMidAirD2', 'GunPointMaleVersusFemale', 'Worms', 'SmallKitchenAppliances', 'FaceFour', 'InlineSkate', 'AllGestureWiimoteY', 'InsectEPGSmallTrain', 'LargeKitchenAppliances', 'Fish', 'GestureMidAirD3', 'RefrigerationDevices', 'ACSF1', 'GunPoint', 'GesturePebbleZ1', 'GestureMidAirD1', 'EOGHorizontalSignal', 'Herring', 'Plane', 'ECGFiveDays', 'Lightning7', 'GunPointAgeSpan', 'Coffee', 'SmoothSubspace', 'DistalPhalanxOutlineAgeGroup', 'InsectEPGRegularTrain', 'GesturePebbleZ2', 'BeetleFly', 'DistalPhalanxOutlineCorrect', 'Car', 'SemgHandSubjectCh2', 'OliveOil', 'Lightning2', 'Symbols', 'ShapeletSim', 'BirdChicken', 'EOGVerticalSignal', 'GunPointOldVersusYoung', 'ProximalPhalanxOutlineAgeGroup', 'Chinatown', 'SemgHandMovementCh2', 'CricketY', 'Meat', 'AllGestureWiimoteX', 'WormsTwoClass', 'FiftyWords', 'ShakeGestureWiimoteZ', 'PickupGestureWiimoteZ', 'PigCVP', 'WordSynonyms', 'ScreenType', 'PowerCons', 'DiatomSizeReduction', 'CricketX', 'Computers', 'DodgerLoopDay', 'Earthquakes', 'AllGestureWiimoteZ', 'Wine', 'PigAirwayPressure', 'MiddlePhalanxOutlineAgeGroup', 'EthanolLevel', 'UMD', 'Adiac', 'DistalPhalanxTW', 'MiddlePhalanxOutlineCorrect', 'Fungi', 'CBF', 'MiddlePhalanxTW']

args.loader='UCR'

seeds_all=[2024, 2025,2026,2027,2028]
args.model='resnet'
# args.model='inception'
# args.model='formertime'
# args.model='oscnn'
# args.model=='fcn'
args.hierarchical_bucketing=False
device = "cuda:1"
for seed in seeds_all:
    for dataset_name in UCR_datasets_all:
        args.task='TSC'
        args.debug=True
        args.seed=seed
        args.dataset=dataset_name
        if args.loader == 'UCR':
            task_type = 'classification'
            train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)

        elif args.loader == 'UEA':
            task_type = 'classification'
            train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)

            # train_data, train_labels, test_data, test_labels=UCR_UEA_datasets().load_dataset(args.dataset)
            train_labels = np.array([int(float(v)) for v in train_labels.tolist()])
            test_labels = np.array([int(float(v)) for v in test_labels.tolist()])
        args.num_classes = len(np.unique(train_labels))
        args.num_class=len(np.unique(train_labels))
        args.device=device
        if args.model == 'formertime':
            ks = args.slice_per_stage[0]
            if train_data.shape[1] < ks:
                train_data = pad_sequeence(train_data)
                test_data = pad_sequeence(test_data)
            if dataset_name == 'Chinatown':
                train_data = pad_sequeence(train_data)
                test_data = pad_sequeence(test_data)

        train_data = torch.from_numpy(train_data)
        train_data = torch.where(torch.isnan(train_data), torch.full_like(train_data, 0),
                                 train_data).cpu().numpy()
        args.data_shape = (train_data.shape[1],train_data.shape[2])
        args.var_num=train_data.shape[-1]
        test_data = torch.from_numpy(test_data)
        test_data = torch.where(torch.isnan(test_data), torch.full_like(test_data, 0), test_data).cpu().numpy()
        X_train, y_train, X_test, y_test = train_data,train_labels,test_data, test_labels
        fix_seed = seed
        torch.manual_seed(fix_seed)
        torch.cuda.manual_seed(seed)
        random.seed(fix_seed)

        np.random.seed(fix_seed)

        Result_log_path = './Example_Results/Results_of_InceptionNet/'

        if args.loader == 'UCR':
            if args.model == 'resnet':
                fname = 'logs_TSC/resnet'
            elif args.model == 'inception':
                fname = 'logs_TSC/inception'

            elif args.model=='formertime':
                fname = 'logs_TSC/formertime'

            elif args.model == 'oscnn':
                fname = 'logs_TSC/OSCNN'
        args.epoch_inteval = 75 #[1,0.1]
        args.weights_sub = [1, 0.1]



        args.predictability_aware_training = True

        args.amortization=True

        fname=fname+'/'+dataset_name
        if not os.path.exists(fname):
            os.makedirs(fname)

        args.metrics_path=fname

        if args.model == 'fcn':
            TSC_model = FCN(num_classes=args.num_classes,
                        num_segments=args.n_segments,
                        input_size=args.var_num,
                        cost_type=args.cost_type,
                        pooling_op=args.pooling_op,
                        gamma=args.gamma)

            TSC_model2 = FCN(num_classes=args.num_classes,
                        num_segments=args.n_segments,
                        input_size=args.var_num,
                        cost_type=args.cost_type,
                        pooling_op=args.pooling_op,
                        gamma=args.gamma)

        elif args.model == 'resnet':
            TSC_model = ResNet(num_classes=args.num_classes,
                            num_segments=args.n_segments,
                            input_size=args.var_num,
                            cost_type=args.cost_type,
                            pooling_op=args.pooling_op,
                            gamma=args.gamma)

            TSC_model2 = ResNet(num_classes=args.num_classes,
                             num_segments=args.n_segments,
                             input_size=args.var_num,
                             cost_type=args.cost_type,
                             pooling_op=args.pooling_op,
                             gamma=args.gamma)
        elif args.model=='inception':
            TSC_model = InceptionNet(input_channle_size=X_train.shape[-1], nb_classes=max(y_train) + 1)
            TSC_model2 = InceptionNet(input_channle_size=X_train.shape[-1], nb_classes=max(y_train) + 1)
        elif args.model == 'formertime':
            TSC_model = FormerTime(args)
            TSC_model2 = FormerTime(args)
        elif args.model=='oscnn':
            TSC_model = OmniScaleCNN(c_in=X_train.shape[-1],c_out= max(y_train)+1,seq_len= X_train.shape[1])
            TSC_model2 = OmniScaleCNN(c_in=X_train.shape[-1],c_out= max(y_train)+1,seq_len= X_train.shape[1])
        easy_use_trainner = Easy_use_trainner(Result_log_folder = Result_log_path,
                                              dataset_name = dataset_name,
                                              device = device,args=args,max_epoch=300,print_result_every_x_epoch=1)
        # put model to trainner
        easy_use_trainner.get_model(TSC_model,TSC_model2)

        # fit data
        easy_use_trainner.fit(X_train, y_train, X_test, y_test)
