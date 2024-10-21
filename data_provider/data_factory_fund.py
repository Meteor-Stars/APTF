import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from data_provider.data_loader_fund import Dataset_Fund
from torch.utils.data import DataLoader
import pandas as pd
import os
import numpy as np

data_dict = {
    'Fund':Dataset_Fund,}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    if args.state=='data_process':
        shuffle_flag = False
        drop_last=False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    print(flag,shuffle_flag,drop_last)
    cal_scaler=args.cal_scaler
    args.flag=flag
    args.scaler_custom=None

    if args.state == 'data_process':

        data_set_all=[]
        data_loader_all=[]

        all_efective_dataset=[]
        for data_path in args.data_path_list:
            # print(data_path)
            # time.sleep(500)
            data_set = Data(
                root_path=args.root_path,
                data_path=data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq, args=args
            )

            drop_last = False

            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=0,  # args.num_workers
                drop_last=drop_last)
            data_set_all.append(data_set)
            data_loader_all.append(data_loader)
            all_efective_dataset.append(data_path)

        return data_set_all,data_loader_all
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,args=args
        )
        drop_last=False
        # print(flag, len(data_set),batch_size)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=0, #args.num_workers
            drop_last=drop_last)
        return data_set, data_loader
