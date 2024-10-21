import os
import time

from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import json
from torch.optim.lr_scheduler import LambdaLR
from Amortized_unpredictability_evolution_aware_loss import AUE_Loss
from .eval_metric import eval_cus
def eval_condition(iepoch,print_result_every_x_epoch):
    if (iepoch + 1) % print_result_every_x_epoch == 0:
        return True
    else:
        return False


def eval_model(model, dataloader,criterion):
    predict_list = np.array([])
    label_list = np.array([])
    test_loss_temp=[]
    for sample in dataloader:
        y_predict = model(sample[0].float())
        loss_1 = criterion(y_predict, sample[1]).mean().item()
        test_loss_temp.append(loss_1)
        y_predict = y_predict.detach().cpu().numpy()
        y_predict = np.argmax(y_predict, axis=1)
        predict_list = np.concatenate((predict_list, y_predict), axis=0)
        label_list = np.concatenate((label_list, sample[1].detach().cpu().numpy()), axis=0)
    # acc = accuracy_score(predict_list, label_list)
    acc=eval_cus(label_list,predict_list)
    return acc,np.mean(test_loss_temp)


def save_to_log(sentence, Result_log_folder, dataset_name):
    father_path = Result_log_folder + dataset_name
    if not os.path.exists(father_path):
        os.makedirs(father_path)
    path = father_path + '/' + dataset_name + '_.txt'
    print(path)
    with open(path, "a") as myfile:
        myfile.write(sentence + '\n')

class Easy_use_trainner():
    
    def __init__(self,
                 Result_log_folder, 
                 dataset_name, 
                 device, 
                 max_epoch = 2000, 
                 batch_size=16,
                 print_result_every_x_epoch = 50,
                 minium_batch_size = 2,
                 lr = None,args=None,
                ):
        
        super(Easy_use_trainner, self).__init__()
        self.args=args
        if not os.path.exists(Result_log_folder +dataset_name+'/'):
            os.makedirs(Result_log_folder +dataset_name+'/')
        Initial_model_path = Result_log_folder +dataset_name+'/'+dataset_name+'initial_model'
        model_save_path = Result_log_folder +dataset_name+'/'+dataset_name+'Best_model'
        

        self.Result_log_folder = Result_log_folder
        self.dataset_name = dataset_name        
        self.model_save_path = model_save_path
        self.Initial_model_path = Initial_model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.print_result_every_x_epoch = print_result_every_x_epoch
        self.minium_batch_size = minium_batch_size

        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps
        self.step = 0

        if lr == None:
            self.lr = 0.001
            # self.lr = 0.01
        else:
            self.lr = lr
        self.Model = None
    
    def get_model(self, model1,model2=None):
        self.Model = model1.to(self.device)
        if model2!=None:
            self.Model2=model2.to(self.device)

        
        
    def fit(self, X_train, y_train, X_val, y_val):

        print('code is running on ',self.device)
        
        # covert numpy to pytorch tensor and put into gpu
        X_train = torch.from_numpy(X_train)
        X_train.requires_grad = False
        X_train = X_train.to(self.device)
        y_train = torch.from_numpy(y_train).to(self.device)
        
        X_test = torch.from_numpy(X_val)
        X_test.requires_grad = False
        X_test = X_test.to(self.device)
        y_test = torch.from_numpy(y_val).to(self.device)

        # add channel dimension to time series data
        if len(X_train.shape) == 2:
            X_train = X_train.unsqueeze_(1)
            X_test = X_test.unsqueeze_(1)
        
        # loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.Adam(self.Model.parameters(),lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=7, min_lr=0.0001) #patience 50
        if self.args.unpredictability_aware_training:
            optimizer2 = optim.Adam(self.Model2.parameters(),lr=self.lr)
            scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'min', factor=0.5, patience=7, min_lr=0.0001) #patience 50

        # build dataloader
        
        train_dataset = TensorDataset(X_train, y_train)
        # train_loader = DataLoader(train_dataset, batch_size=max(int(min(X_train.shape[0] / 10, self.batch_size)),self.minium_batch_size), shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=64*2, shuffle=True)
        test_dataset = TensorDataset(X_test, y_test)

        test_loader = DataLoader(test_dataset, batch_size=128*2, shuffle=False)

        loss_all_train={}
        metric_all_train={}
        loss_all_test={}
        metric_all_test={}

        self.Model.train()   
        best_acc=0
        for i in range(self.max_epoch):
            self.args.epoch=i
            train_loss_temp=[]
            predict_list = np.array([])
            label_list = np.array([])
            for sample in train_loader:
                optimizer.zero_grad()


                if self.args.unpredictability_aware_training:
                    optimizer2.zero_grad()
                    y_predict = self.Model(sample[0].float())
                    y_predict2 = self.Model2(sample[0].float())
                    loss_1,loss_2=AUE_Loss(y_predict,sample[1],y_predict2,criterion,i,self.args)

                else:
                    y_predict = self.Model(sample[0].float())
                    loss_1 = criterion(y_predict, sample[1]).mean()
                loss_1.backward()
                optimizer.step()

                if self.args.unpredictability_aware_training:
                    loss_2.backward()
                    optimizer2.step()
                train_loss_temp.append(loss_1.item())

                y_predict = y_predict.detach().cpu().numpy()
                y_predict = np.argmax(y_predict, axis=1)
                predict_list = np.concatenate((predict_list, y_predict), axis=0)
                label_list = np.concatenate((label_list, sample[1].detach().cpu().numpy()), axis=0)
            acc_train = accuracy_score(predict_list, label_list)
            scheduler.step(loss_1)

            loss_all_train[i]=np.mean(train_loss_temp)
            metric_all_train[i]=acc_train
            if self.args.unpredictability_aware_training:
                scheduler2.step(loss_2)
            if eval_condition(i,self.print_result_every_x_epoch):
                if self.args.debug:
                    for param_group in optimizer.param_groups:
                        print('epoch =',i, 'lr = ', param_group['lr'])
                torch.set_grad_enabled(False)
                self.Model.eval()
                # acc_train = eval_model(self.Model, train_loader)
                acc_test,loss_test = eval_model(self.Model, test_loader,criterion)
                acc_temp=acc_test['acc']
                if self.args.debug:
                    if acc_temp>best_acc:

                        best_acc=acc_temp
                if self.args.debug:
                    print('Epoch {} Test Acc {} Best {}'.format(i,acc_test,best_acc))
                loss_all_test[i] = loss_test
                metric_all_test[i] = acc_test
                self.Model.train()
                torch.set_grad_enabled(True)


        json_record_loss_train = json.dumps(loss_all_train, indent=4)
        json_record_metric_train = json.dumps(metric_all_train, indent=4)

        json_record_loss_test = json.dumps(loss_all_test, indent=4)
        json_record_metric_test = json.dumps(metric_all_test, indent=4)

        # json_record_acc = json.dumps(acc_test_all, indent=4)
        with open(self.args.metrics_path + '/record_all_loss_train_' + str(self.args.seed) + '.json', 'a') as json_file:
            json_file.write(json_record_loss_train)
        with open(self.args.metrics_path + '/record_all_metric_train_' + str(self.args.seed) + '.json', 'a') as json_file:
            json_file.write(json_record_metric_train)

        with open(self.args.metrics_path + '/record_all_loss_test_' + str(self.args.seed) + '.json', 'a') as json_file:
            json_file.write(json_record_loss_test)
        with open(self.args.metrics_path + '/record_all_metric_test_' + str(self.args.seed) + '.json',
                  'a') as json_file:
            json_file.write(json_record_metric_test)

        
        
    def predict(self, X_test):
        
        X_test = torch.from_numpy(X_test)
        X_test.requires_grad = False
        X_test = X_test.to(self.device)
        
        if len(X_test.shape) == 2:
            X_test = X_test.unsqueeze_(1)
        
        test_dataset = TensorDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=max(int(min(X_test.shape[0] / 10, self.batch_size)),2), shuffle=False)
        
        self.Model.eval()
        
        predict_list = np.array([])
        for sample in test_loader:
            y_predict = self.Model(sample[0])
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            
        return predict_list