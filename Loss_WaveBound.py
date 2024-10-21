import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch
import copy
from utils.tools import set_requires_grad

def compute_loss_wavebound(target_output, source_output, batch_y,args):
    f_dim = -1 if args.features == 'MS' else 0
    batch_y = batch_y[:, -args.pred_len:, f_dim:]

    target_output = target_output[:, -args.pred_len:, f_dim:]
    source_output = source_output[:, -args.pred_len:, f_dim:]
    loss = torch.abs(((source_output - batch_y) ** 2).mean(0) - ((target_output - batch_y) ** 2).mean(
        0) + args.epsilon).mean()

    return loss


def reset_batchnorm_statistics(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.reset_running_stats()


class EMAUpdater:
    def __init__(self, target_model, source_model, moving_average_decay, start_iter=0):
        super().__init__()
        self.target_model = target_model
        self.source_model = source_model
        self.moving_average_decay = moving_average_decay
        self.start_iter = start_iter

        set_requires_grad(self.target_model, False)
        self._update_moving_average(0.0)

    def _update_moving_average(self, moving_average_decay):
        with torch.no_grad():
            for target_params, source_params in zip(self.target_model.parameters(), self.source_model.parameters()):
                target_params.copy_(source_params.lerp(target_params, moving_average_decay))
            for target_buffers, source_buffers in zip(self.target_model.buffers(), self.source_model.buffers()):
                target_buffers.copy_(source_buffers)

    def update(self, iter_count):
        if iter_count < self.start_iter:
            self._update_moving_average(0.0)
        else:
            self._update_moving_average(self.moving_average_decay)