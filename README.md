# FedRASP

Reference implementation of **FedRASP: Federated
Resource-Aware Structured Pruning via Adaptive Channel Selection**.


## Released scope

This repository intentionally provides a focused example with:

- **Dataset:** CIFAR-10.
- **Model:** VGG-16.
- **Federated partitions:** IID or Dirichlet non-IID.
- **Resource model:** fixed synthetic computation and bandwidth profiles.
- **Method path:** FedRASP.



## Requirements

- Python 3.9
- NumPy 1.24
- PyTorch 2.1
- torchvision 0.16
- tqdm 4.66



## Quick start

Run the default 20-client Dirichlet non-IID experiment on GPU 0:

```bash
python main.py --data-root ./data --gpu 0
```

Run with an IID partition:

```bash
python main.py --data-root ./data --gpu 0 --iid
```

Run with 50% client participation:

```bash
python main.py --data-root ./data --gpu 0 --participation 0.5
```



## Command-line options

| Option | Default | Description |
| --- | ---: | --- |
| `--rounds` | `100` | Number of communication rounds. |
| `--num-clients` | `20` | Total number of federated clients. |
| `--participation` | `1.0` | Fraction of clients sampled per round. |
| `--local-epochs` | `5` | Number of local training epochs. |
| `--local-batch-size` | `50` | Local training batch size. |
| `--test-batch-size` | `128` | Global evaluation batch size. |
| `--learning-rate` | `0.01` | Initial local learning rate. |
| `--iid` | disabled | Use an IID split instead of Dirichlet non-IID. |
| `--dirichlet-alpha` | `0.5` | Dirichlet concentration for non-IID partitioning. |
| `--data-root` | `./data` | CIFAR-10 download and storage directory. |
| `--gpu` | `0` | CUDA device index; use `-1` for CPU. |
| `--seed` | `1` | Experiment, partition, and training seed. |



## Fixed FedRASP configuration


| Setting | Value |
| --- | ---: |
| Pruning-start layers | `4, ..., 13` |
| Candidate pruning ratios | `0.2, 0.3, ..., 0.7` |
| Continuous density range | `[0.3, 0.8]` |
| Target average density | `0.6` |
| Fastest full-model clients per round | `1` |
| Importance-guided fraction | `0.8` |
| Coverage-aware fraction | `0.2` |
| Importance EMA coefficient | `0.5` |
| Resource-profile seed | `7` |
| Communication precision | `32` bits per parameter |
| SGD momentum | `0.5` |
| Weight decay | `1e-4` |
| Round-wise learning-rate decay | `0.998` |
| FLOPs/parameter projection weights | `0.5 / 0.5` |
| Budget-overflow/time penalties | `10 / 10` |



## Synthetic resource profiles and simulated time

Clients are assigned to fast, medium, and slow resource tiers in proportions
of 40%, 30%, and 30%, respectively. Values are sampled log-uniformly once at
the start of a run using resource seed 7.

| Tier | Compute (GFLOPs/s) | Uplink | Downlink |
| --- | ---: | ---: | ---: |
| Fast | 100-500 | 100-500 Mbps | 200 Mbps-1 Gbps |
| Medium | 50-100 | 50-100 Mbps | 100-200 Mbps |
| Slow | 5-50 | 20-50 Mbps | 50-100 Mbps |


## Repository structure

```text
FedRASP/
|-- main.py                         # Entry point and reproducible seeding
|-- requirements.txt               # Python dependencies
|-- README.md
`-- fedrasp/
    |-- config.py                   # Compact command-line interface
    |-- constants.py                # Fixed FedRASP settings
    |-- data.py                     # CIFAR-10 and federated partitions
    |-- vgg.py                      # Structured VGG-16 definition
    |-- profiling.py                # FLOPs, parameters, and latency surrogate
    |-- resource_allocation.py      # Capacity allocation and projection
    |-- channel_selection.py        # Importance/coverage channel selection
    |-- model_ops.py                # Submodel slicing and indexed aggregation
    |-- local_training.py           # Local SGD and evaluation
    `-- training.py                 # End-to-end federated training loop
```



