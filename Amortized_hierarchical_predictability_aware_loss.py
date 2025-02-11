import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy

def divide_list_equally(input_list,bucket_num):
    k = len(input_list)
    base_size = k // bucket_num
    remainder = k % bucket_num
    divided_lists = []
    for i in range(bucket_num-1):
        divided_lists.append(input_list[i * base_size:(i + 1) * base_size])
    divided_lists.append(input_list[(bucket_num-1) * base_size:])

    return divided_lists

def generate_weights(k,ind):
    if k <= 1:
        raise ValueError("k shoubl be bigger than 1")
    interval = 1 / (k - 1)
    result = [1 - i * interval for i in range(k)]
    result[-1] = result[-2]/2
    return result


def AHPLoss(outputs, batch_y,outputs_2,criterion,epoch,args):
    """
    Amortized hierarchical predictability-aware_loss
    """

    def sub_loss_TSC(outputs, batch_y,outputs_2,criterion,epoch,p_n,args):

        loss_1 = criterion(outputs, batch_y)
        ind_1_sorted = torch.argsort(loss_1)

        bs=outputs.shape[0]

        normal_samples_1=ind_1_sorted[:(bs-int(p_n*bs))]
        potenital_noise_samples_1=ind_1_sorted[-int(p_n*bs):]

        divided_lists_1 = [normal_samples_1, potenital_noise_samples_1]
        weights_all = [args.weights_sub]

        if args.amortization:
            loss_2 = criterion(outputs_2, batch_y)
            ind_2_sorted = torch.argsort(loss_2)
            normal_samples_2 = ind_2_sorted[:(bs - int(p_n * bs))]
            potenital_noise_samples_2 = ind_2_sorted[-int(p_n * bs):]
            divided_lists_2=[normal_samples_2,potenital_noise_samples_2]
        if not args.amortization:
            loss_1_updated = 0
            for i, bucket in enumerate(divided_lists_1):
                if len(bucket) == 0:
                    continue
                for weights in weights_all:
                    loss_1_updated+=criterion(outputs[bucket],batch_y[bucket]).mean()*weights[i]
                loss_1_updated /= len(weights_all)
            loss_2_updated = 0
            return loss_1_updated, loss_2_updated
        else:
            loss_1_updated=0
            for i, bucket in enumerate(divided_lists_2):
                if len(bucket)==0:
                    continue
                for weights in weights_all:
                    loss_1_updated+=criterion(outputs[bucket],batch_y[bucket]).mean()*weights[i]
                loss_1_updated/=len(weights_all)
            loss_2_updated=0
            for i, bucket in enumerate(divided_lists_1):
                if len(bucket)==0:
                    continue
                for weights in weights_all:
                    loss_2_updated+=criterion(outputs_2[bucket],batch_y[bucket]).mean()*weights[i]
                loss_2_updated/=len(weights_all)
            return loss_1_updated,loss_2_updated

    def sub_loss_TSF(outputs, batch_y,outputs_2,criterion,epoch,weights,args):

        loss_1=criterion(outputs,batch_y).mean(-1).mean(-1)

        ind_1_sorted = torch.argsort(loss_1)
        divided_lists_1=divide_list_equally(ind_1_sorted,bucket_num=len(weights))
        if args.amortization:

            loss_2 = criterion(outputs_2,batch_y).mean(-1).mean(-1)


            ind_2_sorted = torch.argsort(loss_2)

            divided_lists_2 = divide_list_equally(ind_2_sorted,bucket_num=len(weights))

        if not args.amortization:
            loss_1_updated = 0
            for i, bucket in enumerate(divided_lists_1):
                if len(bucket) == 0:
                    continue
                loss_1_updated += criterion(outputs[bucket], batch_y[bucket]).mean() * weights[i]
            loss_2_updated = 0
            return loss_1_updated, loss_2_updated
        else:
            loss_1_updated=0
            for i, bucket in enumerate(divided_lists_2):
                if len(bucket)==0:
                    continue
                loss_1_updated+=criterion(outputs[bucket],batch_y[bucket]).mean()*weights[i]

            loss_2_updated=0
            for i, bucket in enumerate(divided_lists_1):
                if len(bucket)==0:
                    continue
                loss_2_updated+=criterion(outputs_2[bucket],batch_y[bucket]).mean()*weights[i]

            return loss_1_updated,loss_2_updated
    loss_1_updated_f=0
    loss_2_updated_f=0

    if args.task == 'TSC':
        start=args.start
        end=args.end
        buckets_num_all=list(range(10))
        penalize_rates=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
        penalize_rates=[0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]
        epochs_all=[i*args.epoch_inteval for i in range(len(buckets_num_all))]

        if args.epoch in epochs_all:
            args.ep_id = epochs_all.index(args.epoch)

        if not args.hierarchical_bucketing:
            "Using fixed multiple bucket groups instead of dynamically changing as training epochs evolve"
            for i,p_n in enumerate([penalize_rates[args.ep_id]]):
                loss_1_updated, loss_2_updated = sub_loss_TSC(outputs, batch_y, outputs_2, criterion, epoch, p_n, args)
                loss_1_updated_f += loss_1_updated
                loss_2_updated_f += loss_2_updated
        else:
            for i, p_n in enumerate(penalize_rates[:args.ep_id + 1]):
                loss_1_updated, loss_2_updated = sub_loss_TSC(outputs, batch_y, outputs_2, criterion, epoch, p_n, args)
                loss_1_updated_f += loss_1_updated
                loss_2_updated_f += loss_2_updated

    elif args.task == 'TSF':
        start = 1
        end = args.bucket_num_K
        if not args.hierarchical_bucketing:
            buckets_num_all = list(range(start, end))
            buckets_num_all.reverse()
            epochs_all = [i * 2 for i in range(len(buckets_num_all))]
            if args.epoch in epochs_all:
                args.ep_id = epochs_all.index(args.epoch)
            ##don't consider previous bucketing strategy
            for i, k in enumerate(buckets_num_all[args.ep_id + 1]):
                loss_1_updated, loss_2_updated = sub_loss_TSF(outputs, batch_y, outputs_2, criterion, epoch,
                                                          generate_weights(k, i), args)
                loss_1_updated_f += loss_1_updated
                loss_2_updated_f += loss_2_updated
        else:
            buckets_num_all = list(range(start, end))
            buckets_num_all.reverse()
            epochs_all = [i * 2 for i in range(len(buckets_num_all))]
            if args.epoch in epochs_all:
                args.ep_id = epochs_all.index(args.epoch)
            ##consider previous bucketing strategy
            for i, k in enumerate(buckets_num_all[:args.ep_id + 1]):
                loss_1_updated, loss_2_updated = sub_loss_TSF(outputs, batch_y, outputs_2, criterion, epoch,
                                                          generate_weights(k, i), args)
                loss_1_updated_f += loss_1_updated
                loss_2_updated_f += loss_2_updated


    return loss_1_updated_f/(end-start),loss_2_updated_f/(end-start)





