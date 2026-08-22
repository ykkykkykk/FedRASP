"""Lightweight FLOP/parameter profiling and client-resource simulation."""

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


@dataclass
class FedRASPLayerStat:
    name: str
    params: int
    flops: int


@dataclass
class FedRASPClientResource:
    flops_per_second: float
    uplink_bits_per_second: float
    downlink_bits_per_second: float


class FedRASPProfiler:
    def profile(self, model, example_input, device="cpu"):
        stats, handles = [], []

        def fedrasp_register(name, module):
            parameters = int(sum(item.numel() for item in module.parameters(recurse=False)))

            def fedrasp_hook(_module, inputs, output):
                flops = 0
                if torch.is_tensor(output) and isinstance(module, nn.Conv2d):
                    batch, output_channels, height, width = output.shape
                    kernel_h, kernel_w = module.kernel_size
                    macs = batch * output_channels * height * width * (module.in_channels // module.groups) * kernel_h * kernel_w
                    flops = 2 * macs + (output.numel() if module.bias is not None else 0)
                elif torch.is_tensor(output) and isinstance(module, nn.Linear):
                    batches = output.numel() // module.out_features
                    flops = 2 * batches * module.in_features * module.out_features
                    flops += batches * module.out_features if module.bias is not None else 0
                elif torch.is_tensor(output) and isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    flops = 4 * output.numel()
                elif torch.is_tensor(output) and isinstance(module, nn.ReLU):
                    flops = output.numel()
                stats.append(FedRASPLayerStat(name, parameters, int(flops)))

            handles.append(module.register_forward_hook(fedrasp_hook))

        model = model.to(device).eval()
        for name, module in model.named_modules():
            if name and not list(module.children()):
                fedrasp_register(name, module)
        with torch.no_grad():
            model(example_input.to(device))
        for handle in handles:
            handle.remove()
        return stats


def fedrasp_simulate_client_resources(num_clients, seed=7):
    generator = np.random.default_rng(seed)
    tiers = (["fast"] * int(round(num_clients * 0.4)) +
             ["medium"] * int(round(num_clients * 0.3)) +
             ["slow"] * (num_clients - int(round(num_clients * 0.4)) - int(round(num_clients * 0.3))))
    generator.shuffle(tiers)
    ranges = {
        "fast": ((1e11, 5e11), (1e8, 5e8), (2e8, 1e9)),
        "medium": ((5e10, 1e11), (5e7, 1e8), (1e8, 2e8)),
        "slow": ((5e9, 5e10), (2e7, 5e7), (5e7, 1e8)),
    }
    resources = {}
    for client_id, tier in enumerate(tiers):
        values = [10 ** generator.uniform(np.log10(low), np.log10(high)) for low, high in ranges[tier]]
        resources[client_id] = FedRASPClientResource(*map(float, values))
    return resources


def fedrasp_round_time_seconds(total_flops, total_parameters, n_samples, client, round_spec):
    iterations = round_spec.local_epochs * n_samples / float(round_spec.batch_size)
    compute = iterations * total_flops / client.flops_per_second
    bits = round_spec.bits_per_parameter * total_parameters
    communication = bits / client.downlink_bits_per_second + bits / client.uplink_bits_per_second
    return float(compute + communication)

