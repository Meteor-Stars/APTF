import os
import time

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_Fund(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',args=None):
        self.data_path=data_path
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.flag=flag
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

        self.args=args

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        df_raw.rename(columns={'transaction_date':'date'},inplace=True)
        date_inf=pd.DataFrame()
        date_inf['date']=df_raw['date']

        date_inf['date'] = pd.to_datetime(date_inf['date'])

        date_inf['day_of_year'] = date_inf['date'].dt.dayofyear
        date_inf['week_of_year'] = date_inf['date'].dt.week
        date_inf['month'] = date_inf['date'].dt.month
        date_inf['year'] = date_inf['date'].dt.year
        date_inf['year'] = pd.Categorical(date_inf['year']).codes


        date_inf.drop(['date'],axis=1,inplace=True)

        df_raw['embid']=df_raw['is_summarydate'].astype('string')+df_raw['is_weekend_delay'].astype('string')+df_raw['holiday_num'].astype('string')
        df_raw['embid'] = pd.Categorical(df_raw['embid']).codes
        embed_clos=['product_pid','is_summarydate','embid', 'holiday_num','is_weekend_delay']
        self.df_embed_id = df_raw[embed_clos].to_numpy()
        self.df_embed_id=np.concatenate([self.df_embed_id ,date_inf],axis=-1)

        df_raw_y=df_raw[['date', 'apply_amt', 'redeem_amt']]
        df_raw_x=df_raw_y

        test_ratio = np.round(self.args.test_point_num / len(df_raw), 2)
        num_test = int(len(df_raw) * test_ratio)
        train_ratio = np.round(1 - test_ratio - 0.1, 2)
        num_train = int(len(df_raw) * train_ratio)

        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw_x.columns[1:]
            df_data_x = df_raw_x[cols_data]

            cols_data_y = df_raw_y.columns[1:]
            df_data_y = df_raw_y[cols_data_y]
        elif self.features == 'S':
            df_data_x=df_data_y = df_raw_x[[self.target]]

        self.scale=False
        if self.scale:
            train_data = df_data_x[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_x = self.scaler.transform(df_data_x.values)
            data_y = self.scaler.transform(df_data_y.values)
        elif self.args.scaler_custom!=None:
            if self.flag!='test':
                data_x = self.args.scaler_custom.transform(df_data_x.values)
                data_y = self.args.scaler_custom.transform(df_data_y.values)
            else:
                data_x = self.args.scaler_custom.transform(df_data_x.values)
                data_y = df_data_y.values
        else:
            data_x = df_data_x.values
            data_y=df_data_y.values

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
        # print(border1,border2,self.set_type)
        # time.sleep(500)
        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]

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

        seq_emb_id_x=self.df_embed_id[s_begin:s_end]
        seq_emb_id_y = self.df_embed_id[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark,seq_emb_id_x,seq_emb_id_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

