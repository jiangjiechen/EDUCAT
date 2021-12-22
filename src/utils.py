# -*- coding: utf-8 -*-


import numpy as np
from copy import copy
import math


def normalize(x, e=0.05):
    tem = copy(x)
    if max(tem) == 0:
        tem += e
    return tem / tem.sum()


def mask_sentence(input, pos_set, action_set, mask_idx):
    sent = np.array(input)  # copy
    adjusted_pos_set = np.array(pos_set)
    for idx, (pos, act) in enumerate(zip(pos_set, action_set)):
        if act == 0:  # replace
            sent[pos] = mask_idx
            adjusted_pos_set[:idx] += 0
        elif act == 1:  # insert
            sent = np.concatenate([sent[:pos], [mask_idx], sent[pos:]])
            adjusted_pos_set[:idx] += 1
        elif act == 2:  # delete
            sent = np.concatenate([sent[:pos], sent[pos + 1:]])
            adjusted_pos_set[:idx] -= 1
    return sent, adjusted_pos_set


def sample_from_dist(dist, size=1, replace=True):
    return np.random.choice(list(range(len(dist))), size=size, replace=replace, p=dist)


def annealing(parameter, num_iter=1):
    return 1 / parameter ** math.floor(num_iter / 5)


def just_acc(just_acc_rate):
    r = np.random.random()
    if r < just_acc_rate:
        return 0
    else:
        return 1
