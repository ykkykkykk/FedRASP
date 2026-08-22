"""End-to-end training loop for the public FedRASP example."""

import copy
import numpy as np
import torch
from tqdm import trange

from . import constants
from .channel_selection import (
    FedRASPChannelState,
    fedrasp_floating_state,
    fedrasp_initialize_channel_units,
    fedrasp_initialize_importance_priors,
    fedrasp_update_importance_history,
)
from .local_training import FedRASPLocalTrainer, fedrasp_evaluate
from .model_ops import (
    FedRASPRoundSpec,
    fedrasp_build_compact_model,
    fedrasp_build_model_candidates,
    fedrasp_global_channel_scores,
    fedrasp_indexed_aggregation,
    fedrasp_model_bounds,
    fedrasp_slice_submodel,
)
from .profiling import FedRASPProfiler, fedrasp_simulate_client_resources
from .resource_allocation import fedrasp_allocate_resources


def fedrasp_train(args, training_dataset, testing_dataset, client_indices):
    first_prunable_layer, last_prunable_layer = fedrasp_model_bounds()
    candidates, costs = fedrasp_build_model_candidates(
        args,
        FedRASPProfiler(),
        first_prunable_layer,
        last_prunable_layer,
        constants.PRUNING_RATIO_MIN,
        constants.PRUNING_RATIO_MAX,
        constants.PRUNING_RATIO_STEP,
    )
    global_model = candidates[("0.0", last_prunable_layer)].to(args.device).train()
    resources = fedrasp_simulate_client_resources(args.num_clients, constants.RESOURCE_SEED)
    round_spec = FedRASPRoundSpec(
        args.local_epochs,
        args.local_batch_size,
        constants.BITS_PER_PARAMETER,
    )
    state = FedRASPChannelState(args.num_clients)
    fedrasp_initialize_channel_units(global_model, state)
    fedrasp_initialize_importance_priors(global_model, state, fedrasp_global_channel_scores)

    cumulative_time = 0.0
    history = []
    for round_index in trange(args.rounds, desc="FedRASP"):
        participating_count = max(1, int(args.participation * args.num_clients))
        client_generator = np.random.default_rng(args.seed + 1000 + round_index)
        selected_clients = client_generator.choice(
            np.arange(args.num_clients), size=participating_count, replace=False
        ).tolist()
        sample_counts = {client_id: len(client_indices[client_id]) for client_id in selected_clients}
        selected_resources = {client_id: resources[client_id] for client_id in selected_clients}
        allocation = fedrasp_allocate_resources(
            costs,
            first_prunable_layer,
            last_prunable_layer,
            selected_clients,
            selected_resources,
            sample_counts,
            round_spec,
            constants.PRUNING_RATIO_MIN,
            constants.PRUNING_RATIO_MAX,
            constants.PRUNING_RATIO_STEP,
            allow_no_prune=True,
            args=args,
        )

        local_states, local_sizes, local_losses, round_times = [], [], [], []
        for client_id in selected_clients:
            assignment = allocation["plan"][client_id]
            key = (assignment["r"], int(assignment["l_start"]))
            template = candidates[key]
            submodel_state = fedrasp_slice_submodel(
                global_model.state_dict(),
                template.state_dict(),
                args,
                client_id,
                state,
                round_idx=round_index,
                total_rounds=args.rounds,
            )
            state.fedrasp_update_coverage(client_id)
            client_model = copy.deepcopy(template)
            client_model.load_state_dict(submodel_state)
            trainer = FedRASPLocalTrainer(args, training_dataset, client_indices[client_id])
            local_state, local_loss = trainer.fedrasp_train(round_index, client_model.to(args.device))
            local_states.append({
                name: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for name, value in local_state.items()
            })
            local_sizes.append(sample_counts[client_id])
            local_losses.append(local_loss)
            round_times.append(float(assignment["round_time_s"]))

        before_aggregation = fedrasp_floating_state(global_model.state_dict())
        aggregated = fedrasp_indexed_aggregation(
            local_states,
            local_sizes,
            global_model.state_dict(),
            state,
            selected_clients,
        )
        global_model.load_state_dict(aggregated)
        updated_units, observed_channels = fedrasp_update_importance_history(
            state,
            local_states,
            selected_clients,
            before_aggregation,
            global_model.state_dict(),
        )

        round_time = max(round_times) if round_times else 0.0
        cumulative_time += round_time
        accuracy = fedrasp_evaluate(global_model, testing_dataset, args)
        mean_loss = float(np.mean(local_losses)) if local_losses else 0.0
        record = {
            "round": round_index,
            "accuracy": accuracy,
            "loss": mean_loss,
            "round_time_seconds": round_time,
            "cumulative_time_seconds": cumulative_time,
            "importance_units_updated": updated_units,
            "importance_channels_observed": observed_channels,
        }
        history.append(record)
        print(
            f"round={round_index:03d} accuracy={accuracy:.2f}% loss={mean_loss:.4f} "
            f"time={round_time:.3f}s density={allocation['avg_budget_density']:.3f}"
        )

    compact_model, compact_key = fedrasp_build_compact_model(
        global_model,
        candidates,
        args,
        state,
        first_prunable_layer,
        last_prunable_layer,
    )
    compact_accuracy = fedrasp_evaluate(compact_model, testing_dataset, args)
    print(f"final_compact_structure={compact_key} accuracy={compact_accuracy:.2f}%")
    return compact_model, state, history

