"""Small public command-line interface for the FedRASP example."""

import argparse
from . import constants


def fedrasp_parse_args():
    parser = argparse.ArgumentParser(description="FedRASP: CIFAR-10 with VGG-16")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--participation", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--local-batch-size", type=int, default=50)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--iid", action="store_true", help="use an IID split (default: Dirichlet non-IID)")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA index; use -1 for CPU")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # Compatibility attributes consumed by the unchanged core implementation.
    args.epochs = args.rounds
    args.num_users = args.num_clients
    args.frac = args.participation
    args.local_ep = args.local_epochs
    args.local_bs = args.local_batch_size
    args.bs = args.test_batch_size
    args.lr = args.learning_rate
    args.dataset = constants.DATASET
    args.model = "vgg"
    args.num_classes = constants.NUM_CLASSES
    args.num_channels = constants.NUM_CHANNELS
    args.image_size = constants.IMAGE_SIZE
    args.r_min = constants.PRUNING_RATIO_MIN
    args.r_max = constants.PRUNING_RATIO_MAX
    args.r_step = constants.PRUNING_RATIO_STEP
    args.budget_min_density = constants.MIN_DENSITY
    args.budget_max_density = constants.MAX_DENSITY
    args.budget_target_density = constants.TARGET_DENSITY
    args.budget_full_fastest_num = constants.FULL_MODEL_FASTEST_CLIENTS
    args.budget_deadline_slack = 0.0
    args.budget_projection_slack = 0.0
    args.budget_projection_time_slack = 0.0
    args.budget_flops_weight = constants.PROJECTION_FLOPS_WEIGHT
    args.budget_params_weight = constants.PROJECTION_PARAMETERS_WEIGHT
    args.budget_over_penalty = constants.PROJECTION_OVERFLOW_PENALTY
    args.budget_time_penalty = constants.PROJECTION_TIME_PENALTY
    args.deploy_prune_rate = constants.PRUNING_RATIO_MIN
    args.deploy_l_start = constants.PRUNING_START_MIN
    args.verbose = False
    return args

