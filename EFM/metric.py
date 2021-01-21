
import torch
from torch import device
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter
import bottleneck as bn


def get_precision_recall(preds_user, preds_item, targets, mask, k=1):
    preds = preds_user
    preds = preds.view(-1, preds.size(1))

    ### indices: batch_size*k
    _, indices = torch.topk(preds, k, -1)
    precisoin_list = []
    recall_list = []
    pop_correct_num = 0
    non_pop_correct_num = 0

    for i, pred_index in enumerate(indices):
        pred_i = list(pred_index.numpy())
        target_i = list(targets[i].numpy())
        true_pos = set(target_i) & set(pred_i)
        true_pos_num = len(true_pos)

        precision = true_pos_num / k
        # print('mask:', mask[i])
        # print('mask:', sum(mask[i]).item())
        recall = true_pos_num / (sum(mask[i]).item())
        # break

        # print(precision, "precision")
        # print(recall, "recall")

        precisoin_list.append(precision)
        recall_list.append(recall)

    precision = np.mean(precisoin_list)

    recall = np.mean(recall_list)
    # exit()
    return precision, recall
