"""CIFAR-10 loading and federated partitioning."""

import numpy as np
from torchvision import datasets, transforms


def fedrasp_iid_partition(num_samples, num_clients, seed):
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(num_samples)
    return {client_id: split.astype(np.int64) for client_id, split in enumerate(np.array_split(shuffled, num_clients))}


def fedrasp_dirichlet_partition(targets, num_clients, alpha, seed):
    generator = np.random.default_rng(seed)
    targets = np.asarray(targets)
    while True:
        client_indices = [[] for _ in range(num_clients)]
        for label in range(10):
            label_indices = np.where(targets == label)[0]
            generator.shuffle(label_indices)
            proportions = generator.dirichlet(np.full(num_clients, alpha))
            boundaries = (np.cumsum(proportions)[:-1] * len(label_indices)).astype(int)
            for client_id, split in enumerate(np.split(label_indices, boundaries)):
                client_indices[client_id].extend(split.tolist())
        if min(map(len, client_indices)) >= 10:
            break
    return {client_id: np.asarray(indices, dtype=np.int64) for client_id, indices in enumerate(client_indices)}


def fedrasp_load_cifar10(args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    training = datasets.CIFAR10(args.data_root, train=True, download=True, transform=train_transform)
    testing = datasets.CIFAR10(args.data_root, train=False, download=True, transform=test_transform)
    if args.iid:
        clients = fedrasp_iid_partition(len(training), args.num_clients, args.seed)
    else:
        clients = fedrasp_dirichlet_partition(
            training.targets, args.num_clients, args.dirichlet_alpha, args.seed)
    return training, testing, clients

