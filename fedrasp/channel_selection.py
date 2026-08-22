"""FedRASP importance-guided and coverage-aware channel selection.

Each layer allocates 80% of its retained channels to delayed importance scores
and 20% to the least-covered remaining channels. Channel selection itself is
fully score-based; there is no stochastic channel exploration term.
"""

import torch
from . import constants
from .model_ops import (
    FedRASPIndexState,
    fedrasp_vgg_unit_key,
    fedrasp_vgg_hidden_unit_key,
)


class FedRASPChannelState(FedRASPIndexState):
    def __init__(self, num_users):
        super().__init__(num_users)
        self.coverage_count = {}
        self.importance_scores = {}
        self._current_client = None

    def ensure_unit(self, unit_key, num_channels):
        super().ensure_unit(unit_key, num_channels)
        if unit_key not in self.coverage_count:
            self.coverage_count[unit_key] = torch.zeros(num_channels, dtype=torch.float32)

    def clear_client_round_maps(self, cid):
        self._current_client = int(cid)
        super().clear_client_round_maps(cid)

    def fedrasp_update_coverage(self, cid):
        for unit_key, indices in self.unit_idx[int(cid)].items():
            if unit_key not in self.alpha:
                continue
            self.ensure_unit(unit_key, self.alpha[unit_key].numel())
            valid = indices.detach().cpu().long().view(-1)
            valid = valid[(valid >= 0) & (valid < self.coverage_count[unit_key].numel())]
            self.coverage_count[unit_key][valid] += 1.0

    def fedrasp_update_importance(self, cid, unit_key, scores, observed):
        n = int(self.alpha[unit_key].numel())
        previous = self.importance_scores.get(int(cid), {}).get(unit_key)
        current = previous.clone() if torch.is_tensor(previous) and previous.numel() == n else torch.full((n,), 0.5)
        observed = observed.detach().cpu().bool().view(-1)[:n]
        scores = scores.detach().cpu().float().view(-1)[:n]
        current[observed] = constants.IMPORTANCE_EMA * current[observed] + (1.0 - constants.IMPORTANCE_EMA) * scores[observed]
        self.importance_scores.setdefault(int(cid), {})[unit_key] = current

    def fedrasp_topk(self, scores, pool, count):
        count = min(max(int(count), 0), int(pool.numel()))
        if count == 0:
            return torch.empty(0, dtype=torch.long)
        _, positions = torch.topk(scores[pool], count, largest=True, sorted=False)
        return pool[positions].long()

    def select_tsadj(self, unit_key, k, deterministic=False, **_kwargs):
        self.ensure_unit(unit_key, self.alpha[unit_key].numel())
        n = int(self.alpha[unit_key].numel())
        k = min(max(int(k), 1), n)
        if k >= n:
            return torch.arange(n).long()

        mean = self.alpha[unit_key] / (self.alpha[unit_key] + self.beta[unit_key]).clamp_min(1e-12)
        score = self.importance_scores.get(self._current_client, {}).get(unit_key, mean).detach().cpu().float()
        all_indices = torch.arange(n).long()
        importance_count = min(k, int(round(k * constants.IMPORTANCE_SELECTION_RATIO)))
        coverage_count = k - importance_count
        importance = self.fedrasp_topk(score, all_indices, importance_count)

        available = torch.ones(n, dtype=torch.bool)
        available[importance] = False
        pool = available.nonzero(as_tuple=False).view(-1).long()
        coverage_score = 1.0 / torch.sqrt(self.coverage_count[unit_key] + 1.0)
        coverage = self.fedrasp_topk(coverage_score, pool, coverage_count)
        selected = torch.cat([importance, coverage]).long()
        self.active_idx[(unit_key, k)] = selected.clone()
        return selected

    def mean_topk(self, unit_key, k):
        return self.select_tsadj(unit_key, k, deterministic=True)


def fedrasp_initialize_channel_units(global_model, state):
    for key, value in global_model.state_dict().items():
        if key.startswith("features.") and key.endswith(".weight") and value.dim() == 4:
            state.ensure_unit(f"vgg:{key}", int(value.shape[0]))
    hidden = global_model.state_dict()["projector.0.weight"]
    state.ensure_unit(fedrasp_vgg_hidden_unit_key(), int(hidden.shape[0]))


def fedrasp_initialize_importance_priors(global_model, state, global_score_function):
    scores = global_score_function(global_model.state_dict(), state)
    strength = 10.0
    for unit_key, score in scores.items():
        score = score.detach().cpu().float().clamp(0.0, 1.0)
        state.alpha[unit_key] = 1.0 + strength * score
        state.beta[unit_key] = 1.0 + strength * (1.0 - score)


def fedrasp_floating_state(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()
            if torch.is_tensor(value) and value.is_floating_point()}


def fedrasp_gather_like(global_tensor, local_tensor, output_indices, input_indices):
    source = global_tensor.detach().cpu()
    target_shape = tuple(local_tensor.shape)
    if source.dim() == 0:
        return source.clone()
    output_indices = output_indices.detach().cpu().long() if torch.is_tensor(output_indices) else None
    input_indices = input_indices.detach().cpu().long() if torch.is_tensor(input_indices) else None
    if source.dim() == 1:
        selected = source[output_indices[:target_shape[0]]] if output_indices is not None else source[:target_shape[0]]
    elif output_indices is not None and input_indices is not None:
        selected = source[output_indices[:target_shape[0], None], input_indices[None, :target_shape[1]]]
    elif output_indices is not None:
        selected = source[output_indices[:target_shape[0]], :target_shape[1]]
    elif input_indices is not None:
        rows = torch.arange(target_shape[0])
        selected = source[rows[:, None], input_indices[None, :target_shape[1]]]
    else:
        selected = source[tuple(slice(0, size) for size in target_shape)]
    return selected[tuple(slice(0, size) for size in target_shape)].clone()


def fedrasp_sum_except_dimension(value, dimension):
    dimensions = tuple(index for index in range(value.dim()) if index != int(dimension))
    return value if not dimensions else value.sum(dim=dimensions)


def fedrasp_channel_entries(key, local_value, output_indices, input_indices, state):
    entries = []

    def fedrasp_add(unit_key, dimension, indices):
        if unit_key not in state.alpha or dimension >= local_value.dim():
            return
        length = int(local_value.shape[dimension])
        mapped = torch.arange(length).long() if indices is None else indices.detach().cpu().long().view(-1)[:length]
        if mapped.numel() == length and bool(((mapped >= 0) & (mapped < state.alpha[unit_key].numel())).all()):
            entries.append((unit_key, dimension, mapped))

    convolution_unit = fedrasp_vgg_unit_key(key)
    if convolution_unit is not None:
        fedrasp_add(convolution_unit, 0, output_indices)
    hidden_unit = fedrasp_vgg_hidden_unit_key()
    if key == "projector.0.weight":
        fedrasp_add(hidden_unit, 0, output_indices)
    elif key == "projector.3.weight":
        fedrasp_add(hidden_unit, 0, output_indices)
        fedrasp_add(hidden_unit, 1, input_indices)
    elif key == "classifier.weight":
        fedrasp_add(hidden_unit, 1, input_indices)
    return entries


def fedrasp_rank_normalize(values):
    values = values.detach().cpu().float().view(-1)
    if values.numel() <= 1:
        return torch.ones_like(values)
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return ranks / float(values.numel() - 1)


def fedrasp_update_importance_history(state, local_states, selected_clients, before_state, after_state):
    epsilon = 1e-12
    updated_units, observed_channels = 0, 0
    for position, client_id in enumerate(selected_clients):
        dot = {unit: torch.zeros_like(alpha) for unit, alpha in state.alpha.items()}
        local_energy = {unit: torch.zeros_like(alpha) for unit, alpha in state.alpha.items()}
        global_energy = {unit: torch.zeros_like(alpha) for unit, alpha in state.alpha.items()}
        for key, local_value in local_states[position].items():
            if not (torch.is_tensor(local_value) and local_value.is_floating_point() and local_value.dim() > 0):
                continue
            if key not in before_state or key not in after_state:
                continue
            output_indices = state.key_out_idx[int(client_id)].get(key)
            input_indices = state.key_in_idx[int(client_id)].get(key)
            before = fedrasp_gather_like(before_state[key], local_value, output_indices, input_indices)
            after = fedrasp_gather_like(after_state[key], local_value, output_indices, input_indices)
            local_update = local_value.detach().cpu().float() - before.float()
            aggregate_update = after.float() - before.float()
            for unit, dimension, indices in fedrasp_channel_entries(
                    key, local_value, output_indices, input_indices, state):
                local_dot_global = fedrasp_sum_except_dimension(local_update * aggregate_update, dimension)
                local_squared = fedrasp_sum_except_dimension(local_update.square(), dimension)
                global_squared = fedrasp_sum_except_dimension(aggregate_update.square(), dimension)
                length = min(indices.numel(), local_dot_global.numel())
                destination = indices[:length]
                dot[unit][destination] += local_dot_global[:length]
                local_energy[unit][destination] += local_squared[:length]
                global_energy[unit][destination] += global_squared[:length]
                observed_channels += length
        for unit in dot:
            observed = (local_energy[unit] > epsilon) & (global_energy[unit] > epsilon)
            if not bool(observed.any()):
                continue
            alignment = dot[unit] / torch.sqrt((local_energy[unit] * global_energy[unit]).clamp_min(epsilon))
            raw = alignment.clamp(min=0.0) * torch.log1p(local_energy[unit].sqrt())
            normalized = torch.full_like(raw, 0.5)
            normalized[observed] = fedrasp_rank_normalize(raw[observed])
            state.fedrasp_update_importance(client_id, unit, normalized, observed)
            updated_units += 1
    return updated_units, observed_channels

