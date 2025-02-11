from exp.exp_basic import Exp_Basic

from utils.tools import adjust_learning_rate
from utils.metrics import MAPE_Fund
from models import PatchTST,NHits,Autoformer,Informer,NLinear,TimeMixer,NSformer,Scaleformer

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
from data_provider.data_factory_fund import data_provider
from torch.utils.data import TensorDataset
from exp.exp_main_FundDataset_process import Data_Process
import json

import matplotlib.pyplot as plt
import matplotlib as mpl
import copy

from Amortized_hierarchical_predictability_aware_loss import AHPLoss
from Loss_WaveBound import EMAUpdater
from Loss_WaveBound import  reset_batchnorm_statistics
from Loss_WaveBound import compute_loss_wavebound
mpl.use('Agg')
warnings.filterwarnings('ignore')
class WMAPELoss(nn.Module):
    def __init__(self):
        super(WMAPELoss, self).__init__()

    def forward(self, pred, true, weights=None):
        if weights is None:
            weights = torch.ones_like(pred)

        numerator = torch.sum(torch.abs(pred - true) * weights)
        denominator = torch.sum(torch.abs(true) * weights)

        wmape = numerator / (denominator)  # 添加一个小的常数，避免分母为0

        return wmape
class moving_avg(nn.Module):
    def __init__(self):
        super(moving_avg, self).__init__()
    def forward(self, x, kernel_size):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            convert_numpy = True
            x = torch.tensor(x)
        else:
            convert_numpy = False
        x = nn.functional.avg_pool1d(x.permute(0, 2, 1), kernel_size, kernel_size)
        x = x.permute(0, 2, 1)
        if convert_numpy:
            x = x.numpy()
        return x

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.mv = moving_avg()
        self.folder_path = './dataset/processed_fund_data'
        self.data_pre_procss=Data_Process(args,self.folder_path)
        self.configs=args
        self.criterion_rec = nn.MSELoss()
        if self.args.wavebound:
            self.ema_updater = EMAUpdater(self.model_2, self.model, self.args.moving_average_decay,
                                          self.args.start_iter)
    def _build_model(self):
        model_dict = {

            'PatchTST':PatchTST,
            'NHits':NHits,
            'Autoformer':Autoformer,
            'Informer':Informer,
            'NLinear': NLinear,
            'TimeMixer': TimeMixer,
            'NSformer':NSformer,
            'Scaleformer':Scaleformer,
        }
        model = model_dict[self.args.model].Model(configs=self.args).float()

        print(f"NUMBER OF PARAMETERS IN MODEL: {self.args.model}: {sum(p.numel() for p in model.parameters())}")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, additional_params=None):
        if additional_params is not None:
            model_optim = optim.AdamW(list(self.model.parameters())+additional_params, lr=self.args.learning_rate)
            return model_optim
        else:
            if self.args.predictability_aware_training:
                model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
                model_optim_2 = optim.AdamW(self.model_2.parameters(), lr=self.args.learning_rate)
                return model_optim, model_optim_2
            else:
                model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
                return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion,mode=None):
        total_loss = []
        self.model.eval()
        preds=[]
        trues=[]
        flag=False
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.model in self.args.models_1:
                    if self.args.predictability_aware_training:
                        outputs = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)
                elif self.args.model == 'Scaleformer':
                    if self.args.predictability_aware_training:
                        outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs_all[-1]
                    else:
                        outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs_all[-1]
                else:

                    if self.args.predictability_aware_training:

                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        # outputs_2 = self.model_2(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                preds.append(pred)
                trues.append(true)

        pred_all = np.array(np.concatenate(preds,axis=0))
        trues_all = np.array(np.concatenate(trues,axis=0))
        total_loss=self.criterion(pred=pred_all,true=trues_all)

        self.model.train()
        return total_loss

    def load_process_data(self,mode=None,args=None):

        batch_x_all=np.load(self.folder_path+'/'+mode+'_x_all.npy',allow_pickle=True)
        batch_y_all=np.load(self.folder_path+'/'+mode+'_y_all.npy',allow_pickle=True)
        batch_x_mark_all=np.load(self.folder_path+'/'+mode+'_x_mark_all.npy',allow_pickle=True)
        batch_y_mark_all=np.load(self.folder_path+'/'+mode+'_y_mark_all.npy',allow_pickle=True)
        train_dataset = TensorDataset( torch.from_numpy(batch_x_all), torch.from_numpy(batch_y_all), \
                torch.from_numpy(batch_x_mark_all), torch.from_numpy(batch_y_mark_all))

        if mode=='train':
            data_loader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=self.args.batch_size,
                                                      shuffle=True)
        else:
            bs = self.args.batch_size*2
            sf=False

            data_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=bs,
                                                       shuffle=sf)


        return data_loader
    def train_one_epoch(self,i,batch_x,batch_y,batch_x_mark,batch_y_mark,iter_count):
        # print(batch_x.shape) #torch.Size([128, 30, 2])

        epoch_time = time.time()
        # iter_count += 1
        if self.args.predictability_aware_training:
            self.model_optim_2.zero_grad()
        self.model_optim.zero_grad()
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        outputs_2 = None
        if self.args.model in self.args.models_1:
            if self.args.predictability_aware_training:
                outputs = self.model(batch_x)
                if self.args.Amortization:
                    outputs_2 = self.model_2(batch_x)
            else:
                outputs = self.model(batch_x)
                if self.args.wavebound:
                    outputs_2 = self.model_2(batch_x)
        elif self.args.model =='Scaleformer':
            if self.args.predictability_aware_training:
                outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs_all[-1]
                if self.args.Amortization:
                    outputs_all_2 = self.model_2(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs_2 = outputs_all_2[-1]
            else:
                outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs_all[-1]
                if self.args.wavebound:
                    outputs_all_2 = self.model_2(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs_2 = outputs_all_2[-1]
        else:

            if self.args.predictability_aware_training:

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.Amortization:
                    outputs_2 = self.model_2(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self.args.wavebound:
                    outputs_2 = self.model_2(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        if self.args.predictability_aware_training:
            loss,loss_2=AHPLoss(outputs, batch_y,outputs_2,self.criterion_tmp,self.args.epoch,self.args)

        elif self.args.wavebound:
            loss = compute_loss_wavebound(outputs_2, outputs, batch_y, self.args)

        else:
            loss=self.criterion_tmp(outputs, batch_y).mean()

        if self.args.predictability_aware_training:

            loss.backward()
            self.model_optim.step()
            if self.args.Amortization:
                loss_2.backward()
                self.model_optim_2.step()

        else:
            if self.args.wavebound:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.model_optim)
                self.scaler.update()

            else:
                loss.backward()
                self.model_optim.step()

        if self.args.wavebound:
            self.ema_updater.update(self.iter_count)
        self.iter_count += 1

        return loss.item()
    def save_checkpoint(self, val_loss, model, path):
        # if self.verbose:
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + self.args.script_id+'checkpoint.pth')
        self.val_loss_min = val_loss
    def train(self, setting):


        if self.args.preprocess_data:
            self.data_pre_procss.process_data(self.args)
        self.criterion =MAPE_Fund(self.args).cal_fund_val
        train_loader_ori=self.load_process_data(mode='train',args=self.args)

        vali_loader=self.load_process_data(mode='valid',args=self.args)
        test_loader=self.load_process_data(mode='test',args=self.args)
        self.test_loader=test_loader
        self.train_loader=train_loader_ori
        if self.args.loss=='mse':
            self.criterion_tmp = torch.nn.MSELoss(reduction='none')
        elif self.args.loss=='huber':
            self.criterion_tmp = torch.nn.HuberLoss(reduction='none', delta=0.5)
        elif self.args.loss=='l1':
            self.criterion_tmp = torch.nn.L1Loss(reduction='none')

        if not self.args.wmape:
            pass
        else:
            self.criterion_tmp = WMAPELoss()

        path = os.path.join(self.args.checkpoints, setting)
        self.args.path=path
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps=0

        if self.args.predictability_aware_training:
            self.model_optim,self.model_optim_2 = self._select_optimizer()

        else:
            self.model_optim = self._select_optimizer()
        train_loss_all_dict={}
        valid_loss_all_dict={}
        test_loss_all_dict={}
        time_now = time.time()
        epoch_time_all=[]
        self.val_loss_min = 100
        self.iter_count = 0

        if self.args.wavebound:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):

            train_loss = []
            self.args.epoch = epoch
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader_ori):
                self.args.test = False
                self.args.i=i
                loss=self.train_one_epoch(i,batch_x, batch_y, batch_x_mark, batch_y_mark,self.iter_count)
                train_loss.append(loss)

            if self.args.wavebound:
                self.apply_standing_statistics()

            epoch_time_all.append(time.time() - epoch_time)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_all_dict[epoch]=train_loss
            vali_data=None
            test_data=None
            self.args.mode='valid'
            self.args.test = True
            vali_loss_dict = self.vali(vali_data, vali_loader, self.criterion,mode='valid')
            valid_loss_all_dict[epoch]=vali_loss_dict
            if self.args.wmape:
                vali_loss = vali_loss_dict['sum']

            else:
                vali_loss = vali_loss_dict['mse']

            self.args.mode = 'test'
            test_loss_dict = self.vali(test_data, self.test_loader, self.criterion,mode='test')
            test_loss_all_dict[epoch]=test_loss_dict
            if self.args.wmape:
                test_loss=test_loss_dict['sum']
            else:
                test_loss = test_loss_dict['mse']

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print('All metric',test_loss_dict['apply'],test_loss_dict['redeem'],test_loss_dict['sum'])
            if self.args.speed_mode:
                pass
            else:
                self.val_loss_min = vali_loss
                self.save_checkpoint(vali_loss, self.model, path)
            adjust_learning_rate(self.model_optim, epoch + 1, self.args)
            json_record_loss_train = json.dumps(train_loss_all_dict, indent=4)
            json_record_loss_val = json.dumps(valid_loss_all_dict, indent=4)
            json_record_loss_test = json.dumps(test_loss_all_dict, indent=4)
            if self.args.record:
                with open(path + '/record_all_loss_train' + '.json', 'w') as json_file:
                    json_file.write(json_record_loss_train)
                with open(path + '/record_all_loss_val' + '.json', 'w') as json_file:
                    json_file.write(json_record_loss_val)
                with open(path + '/record_all_loss_test' + '.json', 'w') as json_file:
                    json_file.write(json_record_loss_test)
        train_cost = time.time() - time_now
        train_loss_all_dict['train_cost_time'] = train_cost
        train_loss_all_dict['train_mean_epoch_time'] = np.mean(epoch_time_all)

        json_record_loss_train = json.dumps(train_loss_all_dict, indent=4)
        json_record_loss_val = json.dumps(valid_loss_all_dict, indent=4)
        json_record_loss_test = json.dumps(test_loss_all_dict, indent=4)

        if self.args.record:
            with open(path + '/record_all_loss_train' + '.json', 'w') as json_file:
                json_file.write(json_record_loss_train)
            with open(path + '/record_all_loss_val' + '.json', 'w') as json_file:
                json_file.write(json_record_loss_val)
            with open(path + '/record_all_loss_test' + '.json', 'w') as json_file:
                json_file.write(json_record_loss_test)
        best_model_path = path + '/' + self.args.script_id+'checkpoint.pth'
        return self.model
    def get_decoder_input(self, batch_y):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
        return dec_inp
    def get_output(self, model, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y=None):
        if self.args.model in self.args.models_1:
            output =model(batch_x)
        elif self.args.model == 'Scaleformer':
            outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            output = outputs_all[-1]
        else:
            output = self.model(batch_x,batch_x_mark, dec_inp, batch_y_mark)
        return output
    def apply_standing_statistics(self):
        self.model_2.train()
        self.model_2.apply(reset_batchnorm_statistics)

        with torch.no_grad():
            loader_iter = iter(self.train_loader)
            for _ in range(self.args.standing_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)

                batch_x, batch_y, batch_x_mark, batch_y_mark = batch

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = self.get_decoder_input(batch_y)
                _ = self.get_output(self.model_2, batch_x, batch_x_mark, dec_inp, batch_y_mark)
    def test(self, setting, test=0):
        if self.args.loss=='mse':
            # self.criterion_tmp = torch.nn.MSELoss(reduction='none')
            self.criterion_tmp = torch.nn.MSELoss()
        elif self.args.loss=='huber':
            self.criterion_tmp = torch.nn.HuberLoss(reduction='none', delta=0.5)
        elif self.args.loss=='l1':
            self.criterion_tmp = torch.nn.L1Loss(reduction='none')
        self.args.device='cuda:'+str(self.args.gpu)
        torch.cuda.empty_cache()
        path = os.path.join(self.args.checkpoints, setting)
        self.configs.path=path
        if self.args.preprocess_data:
            self.data_pre_procss.process_data(self.args)
        test_loader = self.load_process_data(mode='test', args=self.args)
        torch.cuda.empty_cache()
        path = os.path.join(self.args.checkpoints, setting)
        self.configs.path=path
        best_model_path = path + '/' + self.args.script_id + 'checkpoint.pth'
        self.criterion = MAPE_Fund(self.args).cal_fund_val
        self.args.mode = 'test'
        model_init=copy.deepcopy(self.model)
        if test:
            print('loading model',best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        running_times = []
        test_mse=[]
        time_now=time.time()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark)in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                start_time = time.time()
                if self.args.model in self.args.models_1:
                    if self.args.predictability_aware_training:
                        outputs = self.model(batch_x)
                        outputs_2 = self.model_2(batch_x)
                    else:
                        outputs = self.model(batch_x)
                else:

                    if self.args.predictability_aware_training:

                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs_2 = self.model_2(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                running_times.append(time.time()-start_time)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                pred = outputs.numpy()
                true = batch_y.numpy()
                preds.append(pred)
                trues.append(true)


        preds = np.array(np.concatenate(preds,axis=0))
        trues = np.array(np.concatenate(trues,axis=0))
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        total_loss_dict=self.criterion(preds,trues)
        print('num test batch {} {} mean metric {}'.format(i,len(test_mse),np.mean(test_mse)))
        test_cost_time=time.time()-time_now
        total_loss_dict['test_time']=test_cost_time
        print('final test',total_loss_dict)

        return
