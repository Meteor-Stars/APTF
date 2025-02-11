from exp.exp_basic import Exp_Basic
from models import PatchTST,NHits,Autoformer,Informer,NLinear,TimeMixer,NSformer,Scaleformer
from utils.tools import  adjust_learning_rate
from utils.metrics import MAPE_Fund
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
from data_provider.data_factory_pubilc import data_provider


import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import json
from Amortized_hierarchical_predictability_aware_loss import AHPLoss
from Loss_WaveBound import compute_loss_wavebound
from Loss_WaveBound import EMAUpdater
from Loss_WaveBound import  reset_batchnorm_statistics
warnings.filterwarnings('ignore')

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
        self.configs=args
        self.criterion_rec = nn.MSELoss()

        if self.args.wavebound:
            self.ema_updater = EMAUpdater(self.model_2, self.model, self.args.moving_average_decay,
                                          self.args.start_iter)

    def _build_model(self):
        #PatchTST,NHits,Autoformer,Informer,DLinear,NLinear
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
                        # outputs_2 = self.model_2(batch_x)
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

    def train_one_epoch(self,i,batch_x,batch_y,batch_x_mark,batch_y_mark,iter_count):
        loss=0
        epoch_time = time.time()

        if self.args.predictability_aware_training:
            self.model_optim_2.zero_grad()
        self.model_optim.zero_grad()

        # self.model_optim.zero_grad()
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        outputs_2=None
        if self.args.model in self.args.models_1:
            if self.args.predictability_aware_training:
                outputs = self.model(batch_x)
                if self.args.Amortization:
                    outputs_2= self.model_2(batch_x)
            else:
                outputs = self.model(batch_x)
                if self.args.wavebound:
                    outputs_2 = self.model_2(batch_x)

        elif self.args.model == 'Scaleformer':
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
                outputs = self.model(batch_x,batch_x_mark, dec_inp, batch_y_mark)
                if self.args.Amortization:
                    outputs_2= self.model_2(batch_x,batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = self.model(batch_x,batch_x_mark, dec_inp, batch_y_mark)
                if self.args.wavebound:
                    outputs_2 = self.model_2(batch_x,batch_x_mark, dec_inp, batch_y_mark)

        rec_loss = 0


        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        if self.args.predictability_aware_training:

            loss, loss_2 = AHPLoss(outputs, batch_y, outputs_2, self.criterion_tmp,
                                                    self.args.epoch,self.args)

        elif self.args.wavebound:
            loss=compute_loss_wavebound(outputs_2,outputs,batch_y,self.args)

        else:
            loss = self.criterion_tmp(outputs, batch_y).mean()

        if self.args.predictability_aware_training:

            loss.backward()
            self.model_optim.step()
            if self.args.Amortization:
                # self.model_optim_2.zero_grad()
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
            self.ema_updater.update(iter_count)
        self.iter_count += 1

        return loss.item()
    def save_checkpoint(self, val_loss, model, path):
        # if self.verbose:
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + self.args.script_id+'checkpoint.pth')
        self.val_loss_min = val_loss
    def train(self, setting):

        self.criterion =MAPE_Fund(self.args).cal_fund_val
        train_data, train_loader = self._get_data(flag='train')

        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        self.test_loader=test_loader
        self.test_loader=test_loader
        self.train_loader=train_loader
        self.criterion_tmp = torch.nn.MSELoss(reduction='none')

        path = os.path.join(self.args.checkpoints, setting)
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
        self.val_loss_min=100
        self.iter_count = 0

        if self.args.wavebound:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):

            train_loss = []
            self.args.epoch = epoch
            self.model.train()
            if self.args.predictability_aware_training:
                self.model_2.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                self.args.test = False
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
            vali_loss = vali_loss_dict['mse']
            self.args.mode = 'test'
            test_loss_dict = self.vali(test_data, self.test_loader, self.criterion,mode='test')
            test_loss_all_dict[epoch]=test_loss_dict
            test_loss = test_loss_dict['mse']

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            if self.args.speed_mode:
                pass
            else:
                self.val_loss_min = vali_loss
                self.save_checkpoint(vali_loss, self.model, path)
            if self.args.predictability_aware_training:
                adjust_learning_rate(self.model_optim_2, epoch + 1, self.args)
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

        if not self.args.speed_mode:
            best_model_path = path + '/' + self.args.script_id+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

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
        self.criterion_tmp = torch.nn.MSELoss()
        self.args.device='cuda:'+str(self.args.gpu)
        torch.cuda.empty_cache()
        path = os.path.join(self.args.checkpoints, setting)
        self.configs.path=path
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        self.test_loader = test_loader
        torch.cuda.empty_cache()
        path = os.path.join(self.args.checkpoints, setting)
        self.configs.path=path
        best_model_path = path + '/' + self.args.script_id + 'checkpoint.pth'
        self.criterion = MAPE_Fund(self.args).cal_fund_val
        self.args.mode = 'test'
        model_init=copy.deepcopy(self.model)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path)) #,map_location={'cuda:0':'cuda:2'}

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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark)in enumerate(train_loader):

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
