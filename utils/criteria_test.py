from tqdm import tqdm
import torch
import numpy as np
import random
import Library.Utility as utility


def get_usage(idx, n_embed):
    n_steps = idx.shape[-1] if idx.ndim > 1 else 1
    usage = np.zeros((n_steps, n_embed))
    for i in range(n_steps):
        np.add.at(usage[i], idx[:, ], 1)
    return usage


def get_combinatorial_usage(idx, n_embed):
    n_steps = idx.shape[-1] if idx.ndim > 1 else 1
    usage = np.zeros((n_embed, ) * n_steps)
    np.add.at(usage, tuple(idx.T), 1)
    return usage


def get_dataset_usage(idx, used):
    usages = []
    for i in range(used.shape[0]):
        # usage = used.astype(np.float32)[idx].mean()
        usage = used[i].astype(np.float32)[idx[:, i]].mean()
        usages.append(usage)
    return usages


def get_combinatorial_dataset_usage(idx, used):
    usage = used[tuple(idx.T)].astype(np.float32).mean()
    return usage
