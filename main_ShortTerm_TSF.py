import argparse
import os
import time
import torch
from exp.exp_main_Fund import Exp_Main
import random
import numpy as np
import pandas as pd


def get_file_info(directory):
    file_info_list = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            parent_dir = os.path.basename(os.path.dirname(file_path))
            grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            file_info_list.append((grandparent_dir, parent_dir, filename))
    return file_info_list


def main(seed):




    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # parser.add_argument('--is_training', type=int, default=1, help='status')
    # parser.add_argument('--use_multi_scale', action='store_true', help='using mult-scale')
    # parser.add_argument('--prob_forecasting', action='store_true', help='using probabilistic forecasting')
    # parser.add_argument('--scales', default=[3, 2, 1], help='scales in mult-scale') #Scaleformer
    # parser.add_argument('--scale_factor', type=int, default=2, help='scale factor for upsample')

    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    parser.add_argument('--reconstruct_loss', action='store_true',
                        help='whether to use reconstruction loss for patch squeeze', default=False)
    parser.add_argument('--LWI', action='store_true',
                        help='Learnable Weighted-average Integration', default=True)
    parser.add_argument('--MAP', action='store_true',
                        help='Multi-period self-Adaptive Patching', default=False)

    # supplementary config for FiLM model
    parser.add_argument('--modes1', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--mode_type',type=int,default=0)

    # supplementary config for Reformer model
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--film_ours', default=True, action='store_true')
    parser.add_argument('--ab', type=int, default=2, help='ablation version')
    parser.add_argument('--ratio', type=float, default=0.5, help='dropout')
    parser.add_argument('--film_version', type=int, default=0, help='compression')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--use_multi_scale', action='store_true', help='using mult-scale')
    parser.add_argument('--prob_forecasting', action='store_true', help='using probabilistic forecasting')
    parser.add_argument('--scales', default=[16, 8, 4, 2, 1], help='scales in mult-scale')
    parser.add_argument('--scale_factor', type=int, default=2, help='scale factor for upsample')
    # parser.add_argument('--model', type=str, required=True, default='Autoformer',
    #         help='model name, options: [Autoformer, Informer, Transformer, Reformer, FEDformer] and their MS versions: [AutoformerMS, InformerMS, etc]')
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--moving_average_decay', type=float, default=0.99)
    parser.add_argument('--start_iter', type=int, default=250)
    parser.add_argument('--standing_steps', type=int, default=100)
    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')


    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                        help='whether to use future_temporal_feature; True 1 False 0')

    # Pyraformer parameters.
    parser.add_argument('-window_size', type=str,
                        default=[4, 4, 4])  # The number of children of a parent node.
    parser.add_argument('-inner_size', type=int, default=3)  # The number of ajacent nodes.
    # CSCM structure. selection: [Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct]
    parser.add_argument('-CSCM', type=str, default='Bottleneck_Construct')
    parser.add_argument('-truncate', action='store_true',
                        default=False)  # Whether to remove coarse-scale nodes from the attention structure
    parser.add_argument('-use_tvm', action='store_true', default=False)  # Whether to use TVM.

    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    parser.add_argument('-decoder', type=str, default='FC')  # selection: [FC, attention]
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=128)
    parser.add_argument('-d_v', type=int, default=128)
    parser.add_argument('-d_bottleneck', type=int, default=128)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layer', type=int, default=3)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    args = parser.parse_args()

    fix_seed = seed
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)

    np.random.seed(fix_seed)

    pred_len_ = 1  # 5 8 10
    data_type='Fund1'
    args.data = 'Fund'
    args.speed_mode = True
    args.seq_len = 30
    model='PatchTST' # TimeMixer Autoformer NSformer PatchTST NHits NLinear TimeMixer Informer Scaleformer
    args.unpredictability_aware_training=True
    args.Amortization=False
    args.wavebound = False
    args.woEvolution = True
    args.woEvolution=False
    args.patch_pad = True
    args.context_window = None
    args.e_layers = 4
    args.redundancy_scaling = False
    args.activation_tag = True
    args.model = model
    args.train_only = False
    args.dived = True

    if model=='Scaleformer':
        args.scales=[3,2,1]

    args.shared_num = 3
    args.e_layers = 4

    if data_type=='Fund1':
        data_path = './dataset/' + 'Fund1'
    elif data_type=='Fund2':
        data_path = './dataset/' + 'Fund2'
    elif data_type=='Fund3':
        data_path = './dataset/' + 'Fund3'
    args.root_path = data_path
    args.data_path_list = os.listdir(data_path)
    args.target = 'redeem_amt'
    args.features = 'M'
    args.learning_rate = 1e-4
    args.train_epochs = 16
    args.batch_size = 32 * 4
    args.test_point_num = 67  # 50 67
    args.script_id = '1_'
    args.preprocess_data = True
    args.is_training = True
    args.model_id = 0
    args.cal_scaler = False
    model_act = args.model
    args.patch_len = 5
    args.stride = 4
    args.D_norm = True
    args.revin_norm = False
    args.explore_fund_memory = False
    args.pred_len = pred_len_

    args.state = 'train'

    args.task = 'TSF'

    args.gpu = 0
    args.checkpoints = './checkpoints_new1008/' + data_type + '/' + args.model + '/' + 'random_seed_' + str(seed)

    if args.model=='Pyraformer':
        args.input_size = args.seq_len
        args.predict_step = args.pred_len
        args.device='cuda:'+str(args.gpu)

        args.window_size = [3, 3, 3]
        # args.window_size = [2, 2, 2]
        args.input_size = args.seq_len
        args.predict_step = args.pred_len

        args.embed_type='Normal'
    args.individual = 0
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 2
    args.label_len = 0
    args.is_training = True
    args.only_test = False
    args.train_epochs = 15
    args.wmape = False
    args.record = True
    args.revin_norm = False
    args.learning_rate = 0.001
    args.models_1=['PatchTST','DLinear','NLinear','FiLM']
    args.models_2=['Autoformer','Informer','NHites','TimeMixer','Scaleformer','nsAutoformer','Pyraformer']
    args.speed_mode=True
    args.loss = 'mse'
    args.loss_real = 'mse'

    if model in args.models_1:
        args.label_len = 0
    else:
        args.label_len = 10

    args.itr = 1
    args.device = 'cuda:' + str(args.gpu)
    args.dec_in=args.enc_in
    args.c_out=args.enc_in
    if args.model=='TimeMixer':
        args.e_layers = 2
        args.down_sampling_layers = 3
        args.down_sampling_window = 2
        args.d_model = 16
        args.d_ff = 32
        args.seq_len=32
    print('Args in experiment:')
    print(args)
    Exp = Exp_Main
    args.n_heads = 4
    args.d_model = 128
    args.d_ff = 128
    args.task_name = 'short_term_forecast'

    if args.is_training:
        for ii in range(args.itr):
            setting = f'{args.data}_{args.model}_seq{args.seq_len}_pl{args.pred_len}'

            args.save_path = os.path.join(args.checkpoints, setting)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            exp = Exp(args)  # set experiments

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)


    else:

        setting = f'{args.data}_{args.model}_seq{args.seq_len}_pl{args.pred_len}'
        setting += extra
        args.save_path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)

if __name__ == "__main__":
    seed_all=[1986,2021, 2023, ]
    for seed in seed_all:
        main(seed)

