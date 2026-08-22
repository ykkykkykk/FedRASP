"""Local client optimization used by FedRASP."""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from . import constants


class FedRASPDatasetView(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


class FedRASPLocalTrainer:
    def __init__(self, args, dataset, indices):
        self.args = args
        self.loader = DataLoader(
            FedRASPDatasetView(dataset, indices),
            batch_size=args.local_batch_size,
            shuffle=True,
            drop_last=True,
        )

    def fedrasp_train(self, round_index, model):
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.args.learning_rate * (constants.LEARNING_RATE_DECAY ** round_index),
            momentum=constants.SGD_MOMENTUM,
            weight_decay=constants.SGD_WEIGHT_DECAY,
        )
        criterion = nn.CrossEntropyLoss()
        total_loss, batches = 0.0, 0
        for _ in range(self.args.local_epochs):
            for images, labels in self.loader:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                loss = criterion(model(images)["output"], labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                batches += 1
        return model.state_dict(), total_loss / max(1, batches)


def fedrasp_evaluate(model, dataset, args):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(args.device), labels.to(args.device)
            predictions = model(images)["output"].argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return 100.0 * correct / max(1, total)

