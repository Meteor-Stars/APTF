# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2021 THUML @ Tsinghua University
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Autoformer (https://arxiv.org/pdf/2106.13008.pdf) implementation
# from https://github.com/thuml/Autoformer by THUML @ Tsinghua University
####################################################################################

import os
import time

import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import numpy as np
from math import floor


warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # print(size)
        # time.sleep(400)
        # self.seq_len=120
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Syn(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        self.scaler = StandardScaler()
        self.add_noise = False
        lambdaf = lambda x: 0.2*x/(1+x**10) 
        def Mackey_Glass(N, T):
            t, x = np.zeros((N,)), np.zeros((N,))
            x[0] = 1.2
            for k in range(N-1):	
                t[k+1] = t[k]+1
                if k < T:
                    k1=-0.1*x[k]
                    k2=-0.1*(x[k]+k1/2); 
                    k3=-0.1*(x[k]+k2/2); 
                    k4=-0.1*(x[k]+k3);
                    x[k+1]=x[k]+(k1+2*k2+2*k3+k4)/6; 
                else:
                    n=floor((t[k]-T-t[0])+1); 
                    k1=lambdaf(x[n])-0.1*x[k]; 
                    k2=lambdaf(x[n])-0.1*(x[k]+k1/2); 
                    k3=lambdaf(x[n])-0.1*(x[k]+2*k2/2); 
                    k4=lambdaf(x[n])-0.1*(x[k]+k3); 
                    x[k+1]=x[k]+(k1+2*k2+2*k3+k4)/6; 
            return t, x
        def add_outliers(signal, perc=0.00001):
            median = np.median(signal, 0)
            stdev = signal.std(0)
            outliers_sign = np.random.randint(0, 2, signal.shape)*2 - 1
            outliers_mask = np.random.rand(*signal.shape)<perc
            outliers = (np.random.rand(*signal.shape)*50+50) * stdev + median
            outliers = outliers * outliers_sign * outliers_mask
            return signal+outliers
        len = 10000
        t, x1 = Mackey_Glass(len, 18)
        x1 = np.array([x1]).T
        if flag=='train' and self.add_noise:
            x1 = add_outliers(x1)
        _, x2 = Mackey_Glass(len, 12)
        time = np.arange(len)
        values = np.where(time < 10, time**3, (time-9)**2)
        seasonal = []
        for i in range(40):
            for j in range(250):
                seasonal.append(values[j])
        seasonal_upward = seasonal + np.arange(len)*10
        big_event = np.zeros(len)
        big_event[-2000:] = np.arange(2000)*-2000
        non_stationary = np.array([seasonal_upward ]).T
        self.scaler.fit(non_stationary)
        non_stationary = self.scaler.transform(non_stationary)
        x2 = np.array([x2]).T * 2 + non_stationary
        if flag=='train' and self.add_noise:
            x2 = add_outliers(x2)
        _, x3 = Mackey_Glass(len, 9)
        time = np.arange(len)
        values = np.where(time < 10, time**3, (time-9)**2)
        seasonal = []
        for i in range(40):
            for j in range(250):
                seasonal.append(values[j])
        seasonal_upward = seasonal + np.arange(len)*10
        big_event = np.zeros(len)
        big_event[-2000:] = np.arange(2000)*-10
        non_stationary = np.array([seasonal_upward + big_event]).T
        self.scaler.fit(non_stationary)
        non_stationary = self.scaler.transform(non_stationary)
        x3 = np.array([x3]).T * 2 + non_stationary
        if flag=='train' and self.add_noise:
            x3 = add_outliers(x3)
        x = np.concatenate([x1, x2, x3], 1)
        t = (np.array(t)%30)/30
        t = np.concatenate([[t], [t], [t], [t]], 0).T
        np.save('Mackey_Glass.npy', x)
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        num_train = int(x.shape[0] * 0.7)
        num_test = int(x.shape[0] * 0.2)
        num_vali = x.shape[0] - num_train - num_test
        border1s = [0, num_train - self.seq_len, x.shape[0] - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, x.shape[0]]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.data_x = x[border1:border2]
        self.data_y = x[border1:border2]
        self.data_stamp = t

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Stock(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',args=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.args=args

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    def get_data(self,df_raw,border1s,border2s,cof=1):
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            # print('111')
            # time.sleep(500)
            train_data = df_data[border1s[0]*cof:border2s[0]*cof]
            # if cof==1:
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        return data
    def get_data_time_stamp(self,df_raw,border1,border2,cof=1):
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        return data_stamp
    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # start_date=pd.to_datetime('2012-01-01')
        # # end_date=pd.to_datetime('2013-01-01')
        #
        # # start_date = pd.to_datetime('2014-03-07')
        # # end_date = pd.to_datetime('2013-07-01')
        # df_raw['date']=pd.to_datetime(df_raw['date'])
        # df_raw=df_raw[df_raw['date']>=start_date]
        #
        #
        # data_all = df_raw.values
        # data_start = (data_all != 0).argmax(axis=0)
        # bool_1 = data_start == 0
        # df_raw = df_raw.loc[:, bool_1]

        # print(df_raw.shape)
        # time.sleep(500)
        # print(self.args.finest)
        # time.sleep(500)
        files_all=os.listdir(self.root_path)
        print(files_all)
        print(self.args.finest)
        if self.args.finest=='15m':
            df_raw_1h = pd.read_csv(os.path.join(self.root_path,
                                                 'stock_15m.csv'))
        else:
            df_raw_1h = pd.read_csv(os.path.join(self.root_path,
                                              'stock_30m.csv'))

        df_raw_4h = pd.read_csv(os.path.join(self.root_path,
                                             'stock_1h.csv'))

        df_raw_12h = pd.read_csv(os.path.join(self.root_path,
                                             'stock_3h.csv'))

        # df_raw_1h['date'] = pd.to_datetime(df_raw_1h['date'])
        # df_raw_4h['date'] = pd.to_datetime(df_raw_4h['date'])
        # df_raw_12h['date'] = pd.to_datetime(df_raw_12h['date'])
        #
        # df_raw_1h = df_raw_1h[df_raw_1h['date'] >= start_date]
        # df_raw_1h = df_raw_1h.loc[:, bool_1]
        #
        # df_raw_4h = df_raw_4h[df_raw_4h['date'] >= start_date]
        # df_raw_4h = df_raw_4h.loc[:, bool_1]
        #
        # df_raw_12h = df_raw_12h[df_raw_12h['date'] >= start_date]
        # df_raw_12h = df_raw_12h.loc[:, bool_1]
        # start_date = pd.to_datetime('2012-01-02')

        # print(df_raw.shape,df_raw_12h.shape,df_raw_4h.shape,df_raw_1h.shape)
        # time.sleep(500)
        #5 10 30 120

        # df_raw = df_raw[df_raw['date'] <= end_date]

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # df_raw.rename(columns={'Date': 'date'}, inplace=True)

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]


        # num_vali=185
        # num_test = 185
        # num_train = len(df_raw) - num_vali - num_test

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # if self.features == 'M' or self.features == 'MS':
        #     cols_data = df_raw.columns[1:]
        #     df_data = df_raw[cols_data]
        # elif self.features == 'S':
        #     df_data = df_raw[[self.target]]
        #
        # if self.scale:
        #     # print('111')
        #     # time.sleep(500)
        #     train_data = df_data[border1s[0]:border2s[0]]
        #     self.scaler.fit(train_data.values)
        #     data = self.scaler.transform(df_data.values)
        # else:
        #     data = df_data.values
        data=self.get_data(df_raw,border1s,border2s)
        data_12h=self.get_data(df_raw_12h,border1s,border2s,cof=2)
        data_4h = self.get_data(df_raw_4h, border1s, border2s, cof=5)
        if self.args.finest == '15m':
            data_1h = self.get_data(df_raw_1h, border1s, border2s, cof=16)
        else:
            data_1h = self.get_data(df_raw_1h, border1s, border2s, cof=8)

        data_stamp=self.get_data_time_stamp(df_raw,border1,border2)
        data_stamp_12h = self.get_data_time_stamp(df_raw_12h, border1*2, border2*2)
        data_stamp_4h = self.get_data_time_stamp(df_raw_4h, border1*5, border2*5)

        #15m
        if self.args.finest == '15m':
            data_stamp_1h = self.get_data_time_stamp(df_raw_1h, border1 * 16, border2 * 16)
        else:
            # 30m
            data_stamp_1h = self.get_data_time_stamp(df_raw_1h, border1 * 8, border2 * 8)

        self.data_x = data[border1:border2]
        self.data_x_12h=data_12h[border1*2:border2*2]
        self.data_x_4h = data_4h[border1 * 5:border2 * 5]

        if self.args.finest == '15m':
            self.data_x_1h = data_1h[border1 * 16:border2 * 16]
        else:
            self.data_x_1h = data_1h[border1 * 8:border2 * 8]
        # if self.set_type == 1:
        #     print(self.data_x.shape,self.data_x_12h.shape)
        #     time.sleep(500)
        self.data_y = data[border1:border2]
        self.data_y_12h = data_12h[border1*2:border2*2]
        self.data_y_4h = data_4h[border1*5:border2*5]

        if self.args.finest == '15m':
            self.data_y_1h = data_1h[border1 * 16:border2 * 16]
        else:
            self.data_y_1h = data_1h[border1 * 8:border2 * 8]
        self.data_stamp = data_stamp
        self.data_stamp_12h = data_stamp_12h
        self.data_stamp_4h = data_stamp_4h
        self.data_stamp_1h = data_stamp_1h

        self.single=True
    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len

        s_begin_12h = s_begin*2
        s_end_12h = s_end*2

        s_begin_4h = s_begin * 5
        s_end_4h = s_end * 5
        if self.args.finest == '15m':
            s_begin_1h = s_begin * 16
            s_end_1h = s_end * 16
        else:
            s_begin_1h = s_begin * 8
            s_end_1h = s_end * 8

        if self.single:
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            r_begin_12h = s_end_12h - self.label_len*2
            r_end_12h = r_begin_12h + self.label_len*2 + self.pred_len*2

            r_begin_4h = s_end_4h - self.label_len*5
            r_end_4h = r_begin_4h + self.label_len*5 + self.pred_len*5
            if self.args.finest == '15m':
                r_begin_1h = s_end_1h - self.label_len*16
                r_end_1h = r_begin_1h + self.label_len*16 + self.pred_len*16
            else:
                r_begin_1h = s_end_1h - self.label_len * 8
                r_end_1h = r_begin_1h + self.label_len * 8 + self.pred_len * 8
        else:
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x_12h=self.data_x_12h[s_begin_12h:s_end_12h]
        seq_x_4h=self.data_x_4h[s_begin_4h:s_end_4h]
        seq_x_1h=self.data_x_1h[s_begin_1h:s_end_1h]

        seq_y_12h = self.data_y_12h[r_begin_12h:r_end_12h]
        seq_y_4h = self.data_y_4h[r_begin_4h:r_end_4h]
        seq_y_1h = self.data_y_1h[r_begin_1h:r_end_1h]

        seq_x_mark_12h = self.data_stamp_12h[s_begin_12h:s_end_12h]
        seq_y_mark_12h = self.data_stamp_12h[r_begin_12h:r_end_12h]

        seq_x_mark_4h = self.data_stamp_4h[s_begin_4h:s_end_4h]
        seq_y_mark_4h = self.data_stamp_4h[r_begin_4h:r_end_4h]

        seq_x_mark_1h = self.data_stamp_1h[s_begin_1h:s_end_1h]
        seq_y_mark_1h = self.data_stamp_1h[r_begin_1h:r_end_1h]
        if self.args.pre_single_grand and self.args.grand=='1h':
            return seq_x_4h, seq_y, seq_x_mark_4h, seq_y_mark
        elif self.args.pre_single_grand and self.args.grand=='3h':
            return seq_x_12h, seq_y, seq_x_mark_12h, seq_y_mark
        elif self.args.pre_single_grand and self.args.grand=='15m' or self.args.grand=='30m':
            return seq_x_1h, seq_y, seq_x_mark_1h, seq_y_mark
        elif self.args.pre_single_grand and self.args.grand=='1D':
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        # print(seq_x.shape, self.data_x_12h[s_begin_12h:s_end_12h].shape, self.data_x_4h[s_begin_4h:s_end_4h].shape, self.data_x_1h[s_begin_1h:s_end_1h].shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark,self.data_x_12h[s_begin_12h:s_end_12h], \
               self.data_x_4h[s_begin_4h:s_end_4h],self.data_x_1h[s_begin_1h:s_end_1h]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Elect(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',args=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.args=args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    def get_data(self,df_raw,border1s,border2s,cof=1):
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            # print('111')
            # time.sleep(500)
            train_data = df_data[border1s[0]*cof:border2s[0]*cof]
            # if cof==1:
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        return data
    def get_data_time_stamp(self,df_raw,border1,border2,cof=1):
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        return data_stamp
    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        start_date=pd.to_datetime('2012-01-01')
        # end_date=pd.to_datetime('2013-01-01')

        # start_date = pd.to_datetime('2014-03-07')
        # end_date = pd.to_datetime('2013-07-01')
        df_raw['date']=pd.to_datetime(df_raw['date'])
        df_raw=df_raw[df_raw['date']>=start_date]


        data_all = df_raw.values
        data_start = (data_all != 0).argmax(axis=0)
        bool_1 = data_start == 0
        df_raw = df_raw.loc[:, bool_1]
        # print(df_raw.shape)
        # time.sleep(500)
        files_all=os.listdir(self.root_path)
        print(files_all)
        df_raw_1h = pd.read_csv(os.path.join(self.root_path,
                                          'elect_1h.csv'))

        df_raw_4h = pd.read_csv(os.path.join(self.root_path,
                                             'elect_4h.csv'))

        df_raw_12h = pd.read_csv(os.path.join(self.root_path,
                                             'elect_12h.csv'))
        df_raw_1h['date'] = pd.to_datetime(df_raw_1h['date'])
        df_raw_4h['date'] = pd.to_datetime(df_raw_4h['date'])
        df_raw_12h['date'] = pd.to_datetime(df_raw_12h['date'])

        df_raw_1h = df_raw_1h[df_raw_1h['date'] >= start_date]
        df_raw_1h = df_raw_1h.loc[:, bool_1]

        df_raw_4h = df_raw_4h[df_raw_4h['date'] >= start_date]
        df_raw_4h = df_raw_4h.loc[:, bool_1]

        df_raw_12h = df_raw_12h[df_raw_12h['date'] >= start_date]
        df_raw_12h = df_raw_12h.loc[:, bool_1]
        start_date = pd.to_datetime('2012-01-02')
        # print(df_raw.shape,df_raw_12h.shape,df_raw_4h.shape,df_raw_1h.shape)
        # time.sleep(500)
        #5 10 30 120

        # df_raw = df_raw[df_raw['date'] <= end_date]

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # df_raw.rename(columns={'Date': 'date'}, inplace=True)

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]


        num_vali=185
        num_test = 185
        num_train = len(df_raw) - num_vali - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # if self.features == 'M' or self.features == 'MS':
        #     cols_data = df_raw.columns[1:]
        #     df_data = df_raw[cols_data]
        # elif self.features == 'S':
        #     df_data = df_raw[[self.target]]
        #
        # if self.scale:
        #     # print('111')
        #     # time.sleep(500)
        #     train_data = df_data[border1s[0]:border2s[0]]
        #     self.scaler.fit(train_data.values)
        #     data = self.scaler.transform(df_data.values)
        # else:
        #     data = df_data.values
        data=self.get_data(df_raw,border1s,border2s)
        data_12h=self.get_data(df_raw_12h,border1s,border2s,cof=2)
        data_4h = self.get_data(df_raw_4h, border1s, border2s, cof=6)
        data_1h = self.get_data(df_raw_1h, border1s, border2s, cof=24)

        data_stamp=self.get_data_time_stamp(df_raw,border1,border2)
        data_stamp_12h = self.get_data_time_stamp(df_raw_12h, border1*2, border2*2)
        data_stamp_4h = self.get_data_time_stamp(df_raw_4h, border1*6, border2*6)
        data_stamp_1h = self.get_data_time_stamp(df_raw_1h, border1*24, border2*24)


        self.data_x = data[border1:border2]
        self.data_x_12h=data_12h[border1*2:border2*2]
        self.data_x_4h = data_4h[border1 * 6:border2 * 6]
        self.data_x_1h = data_1h[border1 * 24:border2 * 24]
        # if self.set_type == 1:
        #     print(self.data_x.shape,self.data_x_12h.shape)
        #     time.sleep(500)
        self.data_y = data[border1:border2]
        self.data_y_12h = data_12h[border1*2:border2*2]
        self.data_y_4h = data_4h[border1*6:border2*6]
        self.data_y_1h = data_1h[border1*24:border2*24]
        self.data_stamp = data_stamp
        self.data_stamp_12h = data_stamp_12h
        self.data_stamp_4h = data_stamp_4h
        self.data_stamp_1h = data_stamp_1h

        self.single=True
    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len

        s_begin_12h = s_begin*2
        s_end_12h = s_end*2

        s_begin_4h = s_begin * 6
        s_end_4h = s_end * 6

        s_begin_1h = s_begin * 24
        s_end_1h = s_end * 24

        if self.single:
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            r_begin_12h = s_end_12h - self.label_len*2
            r_end_12h = r_begin_12h + self.label_len*2 + self.pred_len*2

            r_begin_4h = s_end_4h - self.label_len*6
            r_end_4h = r_begin_4h + self.label_len*6 + self.pred_len*6

            r_begin_1h = s_end_1h - self.label_len*24
            r_end_1h = r_begin_1h + self.label_len*24 + self.pred_len*24
        else:
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x_12h=self.data_x_12h[s_begin_12h:s_end_12h]
        seq_x_4h=self.data_x_4h[s_begin_4h:s_end_4h]
        seq_x_1h=self.data_x_1h[s_begin_1h:s_end_1h]

        seq_y_12h = self.data_y_12h[r_begin_12h:r_end_12h]
        seq_y_4h = self.data_y_4h[r_begin_4h:r_end_4h]
        seq_y_1h = self.data_y_1h[r_begin_1h:r_end_1h]

        seq_x_mark_12h = self.data_stamp_12h[s_begin_12h:s_end_12h]
        seq_y_mark_12h = self.data_stamp_12h[r_begin_12h:r_end_12h]

        seq_x_mark_4h = self.data_stamp_4h[s_begin_4h:s_end_4h]
        seq_y_mark_4h = self.data_stamp_4h[r_begin_4h:r_end_4h]

        seq_x_mark_1h = self.data_stamp_1h[s_begin_1h:s_end_1h]
        seq_y_mark_1h = self.data_stamp_1h[r_begin_1h:r_end_1h]
        if self.args.pre_single_grand and self.args.grand=='4h':
            return seq_x_4h, seq_y, seq_x_mark_4h, seq_y_mark
        elif self.args.pre_single_grand and self.args.grand=='12h':
            return seq_x_12h, seq_y, seq_x_mark_12h, seq_y_mark
        elif self.args.pre_single_grand and self.args.grand=='1h':
            return seq_x_1h, seq_y, seq_x_mark_1h, seq_y_mark
        elif self.args.pre_single_grand and self.args.grand=='1D':
            return seq_x, seq_y, seq_x_mark, seq_y_mark

        return seq_x, seq_y, seq_x_mark, seq_y_mark,self.data_x_12h[s_begin_12h:s_end_12h], \
               self.data_x_4h[s_begin_4h:s_end_4h],self.data_x_1h[s_begin_1h:s_end_1h]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
