import argparse
import os
import time
import torch
from exp.exp_main_public import Exp_Main
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


def generate_weights(k):

    if k <= 1:
        raise ValueError("k 必须大于1")

    # 计算每个间隔的大小
    interval = 1 / (k - 1)

    # 生成序列
    result = [1 - i * interval for i in range(k)]

    # 确保最后一个元素为0.1
    result[-1] = result[-2]/2

    return result


def main(seed):
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

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

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--use_multi_scale', action='store_true', help='using mult-scale')
    parser.add_argument('--prob_forecasting', action='store_true', help='using probabilistic forecasting')
    parser.add_argument('--scales', default=[16, 8, 4, 2, 1], help='scales in mult-scale')
    parser.add_argument('--scale_factor', type=int, default=2, help='scale factor for upsample')

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

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
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


    # Formers
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')

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

    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--moving_average_decay', type=float, default=0.99)
    parser.add_argument('--start_iter', type=int, default=300)

    # supplementary config for FiLM model
    parser.add_argument('--modes1', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--mode_type',type=int,default=0)
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--film_ours', default=True, action='store_true')
    parser.add_argument('--ab', type=int, default=2, help='ablation version')
    parser.add_argument('--ratio', type=float, default=0.5, help='dropout')
    parser.add_argument('--film_version', type=int, default=0, help='compression')
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    # parser.add_argument('--epsilon', type=float, default=0.01)
    # parser.add_argument('--moving_average_decay', type=float, default=0.99)
    # parser.add_argument('--start_iter', type=int, default=300) #300
    parser.add_argument('--standing_steps', type=int, default=100)

    args = parser.parse_args()

    args.speed_mode = True
    data_type = 'ETTh1'
    model='PatchTST' # TimeMixer Autoformer NSformer PatchTST NHits NLinear TimeMixer Informer Scaleformer
    args.model = model
    pred_len_ = 96 #192,336,720
    args.unpredictability_aware_training=True
    args.Amortization=True
    args.wavebound = False
    args.woEvolution = True
    args.seq_len = 96 #96 for TimeMixer

    seed = seed
    fix_seed = seed
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    args.mode = 'long-term'
    args.data_type = data_type
    args.train_only = False
    args.loss = 'mse'

    args.target = 'OT'

    args.root_path = './dataset/'
    args.embed_dim = 3
    if args.data_type == 'ETTh1':
        args.data_path = 'ETTh1.csv'
        args.data = 'ETTh1'
        args.model_id = 'ETTh1'
        args.data_real = args.data
        args.enc_in = 7
        args.n_heads = 4
        args.d_model = 64 * 2
        args.d_ff = 128
        args.dropout = 0.3
        args.fc_dropout = 0.3
        args.head_dropout = 0
        args.activation_tag = False
        args.D_norm = True
        args.revin_norm = False
        args.embed_dim = 3
        if args.model=='TimeMixer':
            args.down_sampling_layers = 3
            args.down_sampling_window = 2
            args.d_model = 16
            args.d_ff = 32
    elif args.data_type == 'ETTh2':
        args.data_path = 'ETTh2.csv'
        args.data = 'ETTh2'
        args.data_real = args.data
        args.model_id = 'ETTh2'
        args.enc_in = 7
        args.n_heads = 4
        args.d_model = 64 * 2
        args.d_ff = 128
        args.dropout = 0.3
        args.fc_dropout = 0.3
        args.head_dropout = 0
        args.activation_tag = False
        args.redundancy_scaling = True
        args.embed_dim = 3
        args.D_norm = False
        args.revin_norm = True
        if args.model=='TimeMixer':
            args.down_sampling_layers = 3
            args.down_sampling_window = 2
            args.d_model = 16
            args.d_ff = 32

    elif args.data_type == 'ETTm1':
        args.data_path = 'ETTm1.csv'
        args.data = 'ETTm1'
        args.model_id = 'ETTm1'
        args.data_real = 'ETTm1'
        args.enc_in = 7
        args.n_heads = 16
        args.d_model = 64 * 2
        args.d_ff = 256
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.D_norm = False
        args.revin_norm = True
        args.activation_tag = False
        args.embed_dim = 3
        if args.model=='TimeMixer':
            args.down_sampling_layers = 3
            args.down_sampling_window = 2
            args.d_model = 16
            args.d_ff = 32


    elif args.data_type == 'ETTm2':
        args.data_path = 'ETTm2.csv'
        args.data = 'ETTm2'
        args.data_real = 'ETTm2'
        args.model_id = 'ETTm2'
        args.enc_in = 7
        args.e_layers = 1  # 1
        args.n_heads = 16
        args.d_model = 64
        args.d_ff = 256
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.D_norm = False
        args.revin_norm = True
        args.embed_dim = 3
        if args.model=='TimeMixer':
            args.down_sampling_layers = 3
            args.down_sampling_window = 2
            args.d_model = 32
            args.d_ff = 32


    elif args.data_type == 'weather':
        args.data_path = 'weather.csv'
        args.model_id = 'weather'
        args.data = 'custom'
        args.data_real = 'weather'
        args.enc_in = 21
        args.n_heads = 16
        args.d_model = 128
        args.d_ff = 256
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.D_norm = False
        args.revin_norm = True

        args.embed_dim = 3
        if args.model=='TimeMixer':
            args.down_sampling_layers = 3
            args.down_sampling_window = 2
            args.d_model = 16
            args.d_ff = 32
    args.learning_rate = 1e-4
    args.batch_size = 32 * 4

    args.is_training = True
    args.model_id = 0
    args.patch_pad = True
    args.shared_num = 3
    args.context_window = None
    args.threshold_patch_num = 7  # When the number of patches is  less than or equal to this value, no patch squeeze is performed

    args.pred_len = pred_len_

    args.state = 'train'

    args.checkpoints = './checkpoints/' + args.data_real + '/' + args.model + '/' + 'random_seed_' + str(seed)

    args.gpu=0
    args.individual = 0
    args.d_layers = 1
    args.factor = 3
    args.c_out=args.enc_in
    args.dec_in=args.enc_in
    args.is_training = True
    args.only_test = False
    args.train_epochs = 30

    data_fil=['ETTm1', 'ETTm2', 'weather']
    if args.data_type in data_fil:
        args.train_epochs = 10+2
    if args.data_type=='electricity':
        args.train_epochs = 10+2
    args.train_epochs =30
    args.task='TSF'
    args.record = True
    args.D_norm=False
    args.models_1=['PatchTST','DLinear','NLinear','FiLM']
    args.models_2=['Autoformer','Informer','NHites','TimeMixer','Scaleformer','nsAutoformer','Pyraformer']
    if model in args.models_1:
        args.label_len = 0
    else:
        args.label_len = 48
    extra=''

    args.itr = 1
    args.e_layers = 3


    if args.model=='Scaleformer':
        args.seq_len = 256*2
        args.e_layers = 2
    elif args.model=='nsAutoformer':
        args.e_layers=2


    args.script_id = '0_'  # model checkpoint id
    args.device = 'cuda:' + str(args.gpu)
    print('Args in experiment:')

    args.task_name='long_term_forecast'
    print(vars(args))
    Exp = Exp_Main
    args.is_training = True
    args.noise=False
    if args.is_training:
        for ii in range(args.itr):
            setting = f'{args.data_real}_{args.model}_seq{args.seq_len}_pl{args.pred_len}'
            setting+=extra
            args.save_path = os.path.join(args.checkpoints, setting)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            exp = Exp(args)  # set experiments

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)

            # if args.do_predict:
            #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #     exp.predict(setting, True)
            #
            # torch.cuda.empty_cache()
            # best_model_path = args.save_path + '/' + args.script_id + 'checkpoint.pth'
            # os.remove(best_model_path)


if __name__ == "__main__":

    seeds_all = [2024, 2025, 2026, 2027, 2028]
    for seed in seeds_all:
        main(seed)
