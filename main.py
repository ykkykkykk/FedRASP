#!/usr/bin/env python
"""Run the public FedRASP CIFAR-10/VGG-16 example."""

import random
import numpy as np
import torch

from fedrasp.config import fedrasp_parse_args
from fedrasp.data import fedrasp_load_cifar10
from fedrasp.training import fedrasp_train


def fedrasp_set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fedrasp_resolve_device(gpu):
    if gpu < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu >= torch.cuda.device_count():
        gpu = 0
    return torch.device(f"cuda:{gpu}")


def fedrasp_main():
    args = fedrasp_parse_args()
    args.device = fedrasp_resolve_device(args.gpu)
    fedrasp_set_seed(args.seed)
    training, testing, client_indices = fedrasp_load_cifar10(args)
    fedrasp_train(args, training, testing, client_indices)


if __name__ == "__main__":
    fedrasp_main()

