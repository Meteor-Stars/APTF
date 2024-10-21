
from data_provider.data_factory_fund import data_provider
import numpy as np
import torch
import os
import warnings

warnings.filterwarnings('ignore')



class Data_Process():
    def __init__(self, args,folder_path):
        self.args=args
        self.args.state = 'data_process'
        self.folder_path=folder_path
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def vali(self, vali_loader_all,mode=None,val_data_all=None):
        batch_x_all = []
        batch_y_all = []
        batch_x_mark_all = []
        batch_y_mark_all = []

        with torch.no_grad():
            for d_i,vali_loader in enumerate(vali_loader_all):
                try:
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_x_embid,batch_y_embid) in enumerate(vali_loader):

                        batch_x_all.append(batch_x.reshape(-1,batch_x.shape[-2],batch_x.shape[-1]))
                        batch_y_all.append(batch_y.reshape(-1,batch_y.shape[-2],batch_y.shape[-1]))
                        batch_x_mark_all.append(batch_x_mark.reshape(-1,batch_x_mark.shape[-2],batch_x_mark.shape[-1]))
                        batch_y_mark_all.append(batch_y_mark.reshape(-1,batch_y_mark.shape[-2],batch_y_mark.shape[-1]))

                except:
                    print('Problem dataset {}'.format(mode,val_data_all[d_i].data_path))

            batch_x_all=torch.cat(batch_x_all,dim=0).numpy() #torch.Size([269, 40, 3])
            batch_y_all=torch.cat(batch_y_all,dim=0).numpy()
            batch_x_mark_all=torch.cat(batch_x_mark_all,dim=0).numpy()
            batch_y_mark_all=torch.cat(batch_y_mark_all,dim=0).numpy()

            np.save(self.folder_path+'/'+mode+'_x_all.npy',batch_x_all)
            np.save(self.folder_path+'/'+mode+'_y_all.npy',batch_y_all)
            np.save(self.folder_path+'/'+mode+'_x_mark_all.npy',batch_x_mark_all)
            np.save(self.folder_path+'/'+mode+'_y_mark_all.npy',batch_y_mark_all)

        return batch_x_all.shape


    def process_data(self, setting,mode='train'):
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        train_data_all, train_loader_all = self._get_data(flag='train')

        if not self.args.train_only:
            vali_data_all, vali_loader_all = self._get_data(flag='val')
            test_data_all, test_loader_all = self._get_data(flag='test')


        for epoch in range(self.args.train_epochs):

            batch_x_all=[]
            batch_y_all=[]
            batch_x_mark_all=[]
            batch_y_mark_all=[]
            batch_x_embid_all=[]
            batch_y_embid_all=[]

            for i_da,train_loader in enumerate(train_loader_all) :
                # print(len(train_loader))
                try:
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_x_embid,batch_y_embid) in enumerate(train_loader):

                        batch_x_all.append(batch_x.reshape(-1,batch_x.shape[-2],batch_x.shape[-1]))
                        batch_y_all.append(batch_y.reshape(-1,batch_y.shape[-2],batch_y.shape[-1]))
                        batch_x_mark_all.append(batch_x_mark.reshape(-1,batch_x_mark.shape[-2],batch_x_mark.shape[-1]))
                        batch_y_mark_all.append(batch_y_mark.reshape(-1,batch_y_mark.shape[-2],batch_y_mark.shape[-1]))
                        batch_x_embid_all.append(batch_x_embid.reshape(-1,batch_x_embid.shape[-2],batch_x_embid.shape[-1]))
                        batch_y_embid_all.append(batch_y_embid.reshape(-1,batch_y_embid.shape[-2],batch_y_embid.shape[-1]))


                except:
                    print('Problem dataset {}'.format(mode, train_data_all[i_da].data_path))
                    pass
            batch_x_all=torch.cat(batch_x_all,dim=0).numpy() #torch.Size([269, 40, 3])

            batch_y_all=torch.cat(batch_y_all,dim=0).numpy()

            batch_x_mark_all=torch.cat(batch_x_mark_all,dim=0).numpy()
            batch_y_mark_all=torch.cat(batch_y_mark_all,dim=0).numpy()

            np.save(self.folder_path+'/'+mode+'_x_all.npy',batch_x_all)
            np.save(self.folder_path+'/'+mode+'_y_all.npy',batch_y_all)
            np.save(self.folder_path+'/'+mode+'_x_mark_all.npy',batch_x_mark_all)
            np.save(self.folder_path+'/'+mode+'_y_mark_all.npy',batch_y_mark_all)

            shape_valid = self.vali( vali_loader_all,mode='valid',val_data_all=vali_data_all)

            shape_test = self.vali( test_loader_all,mode='test',val_data_all=test_data_all)
            print('Fund Train shape X',batch_x_all.shape)
            print('Fund Valid shape X', shape_valid)
            print('Fund Test shape X', shape_test)
            return None


