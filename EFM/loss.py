import torch
from torch import device
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter 
import bottleneck as bn

class _BPR_LOSS(nn.Module):
    def __init__(self, device):
        super(_BPR_LOSS, self).__init__()
        self.m_device = device

    def forward(self, user_attr_p, user_attr_t, item_attr_p, item_attr_t, rating):
        ### logits: batch_size

        a = 1
        b = 1
        c = 1

        # user-feature matrix loss
        loss1 = torch.mean((user_attr_p - user_attr_t) ** 2) ** 0.5

        # item-feature matrix loss
        loss2 = torch.mean((item_attr_p - item_attr_t) ** 2) ** 0.5

        # rating loss
        loss3 = torch.mean((rating) ** 2) ** 0.5
        loss = a * loss1 + b * loss2

        return loss