"""VGG submodel construction, channel-index mapping, and indexed aggregation."""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from .vgg import fedrasp_vgg16_bn


@dataclass
class FedRASPRoundSpec:
    local_epochs: int
    batch_size: int
    bits_per_parameter: int = 32


class FedRASPIndexState:
    """Per-layer channel scores and per-client scatter-back indices."""

    def __init__(self, num_users: int):
        self.alpha: Dict[str, torch.Tensor] = {}
        self.beta: Dict[str, torch.Tensor] = {}
        self.active_idx: Dict[Tuple[str, int], torch.Tensor] = {}
        self.key_out_idx = {cid: {} for cid in range(num_users)}
        self.key_in_idx = {cid: {} for cid in range(num_users)}
        self.unit_idx = {cid: {} for cid in range(num_users)}

    def ensure_unit(self, unit_key: str, num_channels: int):
        if unit_key not in self.alpha:
            self.alpha[unit_key] = torch.ones(num_channels, dtype=torch.float32)
            self.beta[unit_key] = torch.ones(num_channels, dtype=torch.float32)

    def mean_topk(self, unit_key: str, k: int) -> torch.Tensor:
        scores = self.alpha[unit_key] / (self.alpha[unit_key] + self.beta[unit_key]).clamp_min(1e-12)
        _, idx = torch.topk(scores, min(max(int(k), 1), scores.numel()), largest=True, sorted=False)
        return idx.long()

    def select_tsadj(self, unit_key: str, k: int, **_kwargs) -> torch.Tensor:
        return self.mean_topk(unit_key, k)

    def clear_client_round_maps(self, cid: int):
        self.key_out_idx[cid].clear()
        self.key_in_idx[cid].clear()
        self.unit_idx[cid].clear()


def fedrasp_prefix_slice(src: torch.Tensor, tgt_shape):
    if tuple(src.shape) == tuple(tgt_shape):
        return src
    out = src.new_zeros(tgt_shape)
    common = tuple(min(s, t) for s, t in zip(src.shape, tgt_shape))
    out[tuple(slice(0, c) for c in common)] = src[tuple(slice(0, c) for c in common)]
    return out


def fedrasp_expanded_flatten_indices(channel_idx: torch.Tensor, spatial_mult: int) -> torch.Tensor:
    if spatial_mult <= 1:
        return channel_idx
    offsets = torch.arange(spatial_mult, device=channel_idx.device)
    return (channel_idx[:, None] * spatial_mult + offsets[None, :]).reshape(-1)


def fedrasp_vgg_feature_index(k: str) -> Optional[int]:
    parts = k.split('.')
    if len(parts) >= 3 and parts[0] == 'features':
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def fedrasp_vgg_hidden_unit_key() -> str:
    return 'vgg:projector_hidden'


def fedrasp_weight_channel_score(v: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if v.dim() <= 1:
            return v.detach().abs().float().cpu()
        dims = tuple(range(1, v.dim()))
        return v.detach().abs().float().sum(dim=dims).cpu()


def fedrasp_score_along_dimension(v: torch.Tensor, dim: int) -> torch.Tensor:
    with torch.no_grad():
        if v.dim() == 0:
            return v.detach().abs().float().view(1).cpu()
        dim = int(dim)
        dims = tuple(d for d in range(v.dim()) if d != dim)
        if len(dims) == 0:
            return v.detach().abs().float().cpu()
        return v.detach().abs().float().sum(dim=dims).cpu()


def fedrasp_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float()
    if x.numel() == 0:
        return x
    return (x - x.min()) / (x.max() - x.min() + eps)


def fedrasp_model_bounds() -> Tuple[int, int]:
    return 4, 13


def fedrasp_build_model_candidates(args, profiler, l_min, l_max, r_min, r_max, r_step):
    pruning_rates = [0.0] + [round(float(r), 1) for r in np.arange(r_min, r_max + r_step, r_step)]
    example = torch.randn(2, 3, 32, 32)
    models, costs = {}, {}
    for pruning_rate in pruning_rates:
        start_layers = [l_max] if pruning_rate == 0.0 else list(range(l_min, l_max + 1))
        for start_layer in start_layers:
            model = fedrasp_vgg16_bn(
                num_classes=10,
                track_running_stats=False,
                slim_idx=start_layer,
                scale=1.0 - pruning_rate,
            )
            stats = profiler.profile(copy.deepcopy(model).cpu(), example, device="cpu")
            flops = int(sum(item.flops for item in stats))
            parameters = int(sum(parameter.nelement() for parameter in model.parameters()))
            profiled_parameters = int(sum(item.params for item in stats))
            assert profiled_parameters == parameters
            model.to(args.device).train()
            key = (f"{pruning_rate:.1f}", start_layer)
            models[key] = model
            costs[key] = (parameters, flops)
    return models, costs
def fedrasp_slice_vgg(global_param, slim_param, cid: int, state: FedRASPIndexState,
                      args=None, round_idx: int = 0, total_rounds: int = 1,
                      phase_start: Optional[int] = None, deterministic: bool = False):
    param = copy.deepcopy(slim_param)
    state.clear_client_round_maps(cid)
    bn_buffers = ('running_mean', 'running_var', 'num_batches_tracked')
    conv_keys = []
    for k, v_tgt in slim_param.items():
        v_src = global_param.get(k)
        if k.startswith('features.') and k.endswith('.weight') and torch.is_tensor(v_src) and torch.is_tensor(v_tgt) and v_src.dim() == 4 and v_tgt.dim() == 4:
            conv_keys.append(k)
    conv_keys.sort(key=lambda k: fedrasp_vgg_feature_index(k) if fedrasp_vgg_feature_index(k) is not None else -1)

    conv_out_idx, conv_in_idx = {}, {}
    prev_out_idx, last_conv_key = None, None
    for conv_key in conv_keys:
        v_src, v_tgt = global_param[conv_key], slim_param[conv_key]
        local_out, global_out = int(v_tgt.shape[0]), int(v_src.shape[0])
        unit = f'vgg:{conv_key}'
        state.ensure_unit(unit, global_out)
        out_idx = torch.arange(global_out).long() if local_out >= global_out else state.select_tsadj(
            unit, local_out, args=args, round_idx=round_idx, total_rounds=total_rounds,
            phase_start=phase_start, deterministic=deterministic).long()
        conv_out_idx[conv_key] = out_idx
        conv_in_idx[conv_key] = prev_out_idx
        if out_idx is not None:
            state.unit_idx[cid][unit] = out_idx.cpu().clone()
        prev_out_idx = out_idx
        last_conv_key = conv_key

    conv_to_bn_idx = {}
    for conv_key, out_idx in conv_out_idx.items():
        if out_idx is not None:
            conv_i = fedrasp_vgg_feature_index(conv_key)
            if conv_i is not None:
                conv_to_bn_idx[conv_i + 1] = out_idx

    final_conv_out_idx = conv_out_idx.get(last_conv_key) if last_conv_key else None
    final_conv_global_out = global_param[last_conv_key].shape[0] if last_conv_key else None

    projector_hidden_idx = None
    projector_hidden_key = 'projector.0.weight'
    if projector_hidden_key in global_param and projector_hidden_key in slim_param:
        hidden_global = int(global_param[projector_hidden_key].shape[0])
        hidden_local = int(slim_param[projector_hidden_key].shape[0])
        hidden_unit = fedrasp_vgg_hidden_unit_key()
        state.ensure_unit(hidden_unit, hidden_global)
        projector_hidden_idx = (
            torch.arange(hidden_global).long()
            if hidden_local >= hidden_global
            else state.select_tsadj(
                hidden_unit, hidden_local, args=args, round_idx=round_idx,
                total_rounds=total_rounds, phase_start=phase_start,
                deterministic=deterministic).long()
        )
        state.unit_idx[cid][hidden_unit] = projector_hidden_idx.cpu().clone()

    for k, v_tgt in param.items():
        if k not in global_param:
            continue
        v_src = global_param[k]
        if not (torch.is_tensor(v_src) and torch.is_tensor(v_tgt)) or k.endswith(bn_buffers):
            continue
        out_idx, in_idx = None, None
        if k in conv_out_idx:
            out_idx, in_idx = conv_out_idx[k], conv_in_idx[k]
        elif k.startswith('features.'):
            feature_i = fedrasp_vgg_feature_index(k)
            conv_bias_key = f'features.{feature_i}.weight' if feature_i is not None else None
            if conv_bias_key in conv_out_idx and k.endswith('.bias') and v_src.dim() == 1:
                out_idx = conv_out_idx[conv_bias_key]
            elif feature_i in conv_to_bn_idx and (k.endswith('.weight') or k.endswith('.bias')) and v_src.dim() == 1:
                out_idx = conv_to_bn_idx[feature_i]
        elif k == 'projector.0.weight':
            out_idx = projector_hidden_idx
            if final_conv_out_idx is not None and final_conv_global_out:
                spatial_mult = max(1, int(v_src.shape[1]) // int(final_conv_global_out))
                in_idx = fedrasp_expanded_flatten_indices(final_conv_out_idx.to(v_src.device), spatial_mult)[:v_tgt.shape[1]]
        elif k == 'projector.0.bias':
            out_idx = projector_hidden_idx
        elif k == 'projector.3.weight':
            out_idx = projector_hidden_idx
            in_idx = projector_hidden_idx
        elif k == 'projector.3.bias':
            out_idx = projector_hidden_idx
        elif k == 'classifier.weight':
            in_idx = projector_hidden_idx

        if v_src.dim() == 1:
            if out_idx is not None:
                o_idx = out_idx[:v_tgt.shape[0]].to(v_src.device)
                param[k] = v_src[o_idx].to(device=v_tgt.device, dtype=v_tgt.dtype)
                state.key_out_idx[cid][k] = o_idx.cpu()
                state.key_in_idx[cid][k] = None
            else:
                param[k] = fedrasp_prefix_slice(v_src, v_tgt.shape).to(device=v_tgt.device, dtype=v_tgt.dtype)
                state.key_out_idx[cid][k] = None
                state.key_in_idx[cid][k] = None
        elif v_src.dim() > 1 and 'weight' in k:
            local_out = v_tgt.shape[0]
            local_in = v_tgt.shape[1]
            o_idx = out_idx[:local_out].to(v_src.device) if out_idx is not None else None
            i_idx = in_idx[:local_in].to(v_src.device) if in_idx is not None else None
            if o_idx is not None and i_idx is not None:
                selected = v_src[o_idx[:, None], i_idx[None, :]]
            elif o_idx is not None:
                selected = v_src[o_idx, :local_in]
            elif i_idx is not None:
                rows = torch.arange(local_out, device=v_src.device)
                selected = v_src[rows[:, None], i_idx[None, :]]
            else:
                selected = fedrasp_prefix_slice(v_src, v_tgt.shape)
            param[k] = selected.to(device=v_tgt.device, dtype=v_tgt.dtype)
            state.key_out_idx[cid][k] = o_idx.cpu() if o_idx is not None else None
            state.key_in_idx[cid][k] = i_idx.cpu() if i_idx is not None else None
        else:
            param[k] = fedrasp_prefix_slice(v_src, v_tgt.shape).to(device=v_tgt.device, dtype=v_tgt.dtype)
            state.key_out_idx[cid][k] = None
            state.key_in_idx[cid][k] = None
    return param


def fedrasp_slice_submodel(global_param, slim_param, args, cid: int, state: FedRASPIndexState,
                            deterministic: bool = False, round_idx: int = 0,
                            total_rounds: int = 1, phase_start: Optional[int] = None):
    return fedrasp_slice_vgg(
        global_param, slim_param, cid, state, args=args, round_idx=round_idx,
        total_rounds=total_rounds, phase_start=phase_start, deterministic=deterministic)
def fedrasp_clone_index(idx):
    if torch.is_tensor(idx):
        return idx.detach().cpu().long().clone()
    return idx


def fedrasp_valid_indices(idx: Optional[torch.Tensor], limit: int, device) -> Optional[torch.Tensor]:
    if idx is None:
        return None
    idx = idx.to(device=device, dtype=torch.long).view(-1)
    if idx.numel() == 0:
        return idx
    return idx[(idx >= 0) & (idx < int(limit))]


def fedrasp_scatter_add_weighted(tmp_v: torch.Tensor, count_k: torch.Tensor, v_local: torch.Tensor,
                          wt: float, out_idx: Optional[torch.Tensor], in_idx: Optional[torch.Tensor]):
    local = v_local.to(device=tmp_v.device, dtype=tmp_v.dtype)
    if local.dim() == 0 or tmp_v.dim() == 0:
        tmp_v += local.reshape_as(tmp_v).to(tmp_v.dtype) * wt
        count_k += wt
        return

    out_idx = fedrasp_valid_indices(out_idx, tmp_v.shape[0], tmp_v.device)
    in_idx = fedrasp_valid_indices(in_idx, tmp_v.shape[1], tmp_v.device) if tmp_v.dim() > 1 else None

    if out_idx is None and in_idx is None:
        common = tuple(min(int(g), int(l)) for g, l in zip(tmp_v.shape, local.shape))
        dst = tuple(slice(0, c) for c in common)
        src = tuple(slice(0, c) for c in common)
        tmp_v[dst] += local[src] * wt
        count_k[dst] += wt
        return

    if local.dim() == 1:
        idx = out_idx if out_idx is not None else torch.arange(min(local.shape[0], tmp_v.shape[0]), device=tmp_v.device)
        n = min(int(idx.numel()), int(local.shape[0]))
        if n <= 0:
            return
        idx = idx[:n]
        tmp_v[idx] += local[:n] * wt
        count_k[idx] += wt
        return

    if out_idx is None:
        out_idx = torch.arange(min(local.shape[0], tmp_v.shape[0]), device=tmp_v.device)
    if in_idx is None:
        in_idx = torch.arange(min(local.shape[1], tmp_v.shape[1]), device=tmp_v.device)

    n_out = min(int(out_idx.numel()), int(local.shape[0]))
    n_in = min(int(in_idx.numel()), int(local.shape[1]))
    if n_out <= 0 or n_in <= 0:
        return
    out_idx = out_idx[:n_out]
    in_idx = in_idx[:n_in]

    rest_common = tuple(min(int(tmp_v.shape[d]), int(local.shape[d])) for d in range(2, local.dim()))
    rest_dst = tuple(slice(0, c) for c in rest_common)
    index = (out_idx[:, None], in_idx[None, :]) + rest_dst
    src = (slice(0, n_out), slice(0, n_in)) + rest_dst
    tmp_v[index] += local[src] * wt
    count_k[index] += wt


def fedrasp_indexed_aggregation(w_locals, lens, global_model_param, state: FedRASPIndexState, selected: List[int]):
    w_avg = copy.deepcopy(global_model_param)
    for k, v_global in w_avg.items():
        if not torch.is_tensor(v_global):
            continue
        if k.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
            w_avg[k] = global_model_param[k]
            continue
        tmp_v = v_global.new_zeros(v_global.size(), dtype=torch.float32)
        count_k = v_global.new_zeros(v_global.size(), dtype=torch.float32)
        for m, w in enumerate(w_locals):
            if k not in w or not torch.is_tensor(w[k]):
                continue
            v_local = w[k]
            cid = selected[m]
            out_idx = state.key_out_idx[cid].get(k)
            in_idx = state.key_in_idx[cid].get(k)
            wt = float(lens[m])
            fedrasp_scatter_add_weighted(tmp_v, count_k, v_local, wt, out_idx, in_idx)
        mask = count_k > 0
        out = tmp_v.clone()
        out[mask] = out[mask] / count_k[mask]
        out[~mask] = global_model_param[k].to(out.dtype)[~mask]
        w_avg[k] = out.to(dtype=v_global.dtype)
    return w_avg


def fedrasp_vgg_unit_key(k: str) -> Optional[str]:
    if k.startswith('features.') and k.endswith('.weight') and fedrasp_vgg_feature_index(k) is not None:
        return f'vgg:{k}'
    return None


def fedrasp_vgg_hidden_score_dimensions(k: str):
    if k == 'projector.0.weight':
        return (0,)
    if k == 'projector.3.weight':
        return (0, 1)
    if k == 'classifier.weight':
        return (1,)
    return ()


def fedrasp_vgg_hidden_score_entries(k: str, v: torch.Tensor, out_idx: Optional[torch.Tensor],
                                        in_idx: Optional[torch.Tensor], unit_size: int):
    entries = []
    if k == 'projector.0.weight' and out_idx is not None and int(v.shape[0]) == int(out_idx.numel()):
        entries.append((0, out_idx.cpu().long()))
    elif k == 'projector.3.weight':
        if out_idx is not None and int(v.shape[0]) == int(out_idx.numel()):
            entries.append((0, out_idx.cpu().long()))
        if in_idx is not None and v.dim() > 1 and int(v.shape[1]) == int(in_idx.numel()):
            entries.append((1, in_idx.cpu().long()))
    elif k == 'classifier.weight' and in_idx is not None and v.dim() > 1 and int(v.shape[1]) == int(in_idx.numel()):
        entries.append((1, in_idx.cpu().long()))
    return [(dim, idx) for dim, idx in entries if idx.numel() > 0 and int(idx.max()) < int(unit_size)]


def fedrasp_global_channel_scores(global_state: Dict[str, torch.Tensor], state: FedRASPIndexState):
    scores = {unit: torch.zeros_like(alpha.cpu()) for unit, alpha in state.alpha.items()}
    counts = {unit: torch.zeros_like(alpha.cpu()) for unit, alpha in state.alpha.items()}
    for key, value in global_state.items():
        if not torch.is_tensor(value) or value.dim() <= 1 or "weight" not in key:
            continue
        conv_unit = fedrasp_vgg_unit_key(key)
        if conv_unit is not None and conv_unit in scores:
            channel_score = fedrasp_weight_channel_score(value)
            length = min(channel_score.numel(), scores[conv_unit].numel())
            scores[conv_unit][:length] += channel_score[:length]
            counts[conv_unit][:length] += 1.0
        hidden_unit = fedrasp_vgg_hidden_unit_key()
        if hidden_unit in scores:
            for dimension in fedrasp_vgg_hidden_score_dimensions(key):
                channel_score = fedrasp_score_along_dimension(value, dimension)
                length = min(channel_score.numel(), scores[hidden_unit].numel())
                scores[hidden_unit][:length] += channel_score[:length]
                counts[hidden_unit][:length] += 1.0
    for unit in scores:
        observed = counts[unit] > 0
        if observed.any():
            scores[unit][observed] /= counts[unit][observed]
        scores[unit] = fedrasp_normalize(scores[unit])
    return scores
def fedrasp_resolve_final_template_key(net_glob_list, args, l_min: int, l_max: int):
    deploy_rate = getattr(args, 'deploy_prune_rate', None)
    deploy_rate = float(args.r_min if deploy_rate is None else deploy_rate)
    deploy_rate = float(np.clip(deploy_rate, 0.0, float(args.r_max)))
    deploy_rate = round(deploy_rate, 1)

    deploy_l_start = getattr(args, 'deploy_l_start', None)
    deploy_l_start = l_min if deploy_l_start is None else int(deploy_l_start)
    deploy_l_start = int(np.clip(deploy_l_start, l_min, l_max))

    if deploy_rate <= 0.0:
        preferred = ('0.0', l_max)
    else:
        preferred = (f'{deploy_rate:.1f}', deploy_l_start)
    if preferred in net_glob_list:
        return preferred

    def distance(key):
        rate_key, depth_key = key
        return (abs(float(rate_key) - deploy_rate), abs(int(depth_key) - deploy_l_start))

    return min(net_glob_list.keys(), key=distance)


def fedrasp_build_compact_model(global_model, net_glob_list, args, state: FedRASPIndexState,
                                      l_min: int, l_max: int):
    final_key = fedrasp_resolve_final_template_key(net_glob_list, args, l_min, l_max)
    compact_model = copy.deepcopy(net_glob_list[final_key]).to(args.device)
    compact_state = fedrasp_slice_submodel(
        global_model.state_dict(),
        compact_model.state_dict(),
        args,
        cid=0,
        state=state,
        deterministic=True,
    )
    compact_model.load_state_dict(compact_state)
    compact_model.eval()
    return compact_model, final_key
