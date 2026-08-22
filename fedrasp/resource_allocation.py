"""Continuous resource budgets and projection to deployable VGG structures."""

import numpy as np
from .profiling import fedrasp_round_time_seconds


def fedrasp_candidate_structures(net_slim_info, l_min: int, l_max: int, r_min: float, r_max: float,
                          r_step: float, allow_no_prune: bool):
    candidates = []
    rs = np.arange(float(r_min), float(r_max) + 1e-12, float(r_step), dtype=float)
    rs = np.round(np.clip(rs, 0.0, 0.999999), 1)
    for l_start in range(int(l_min), int(l_max) + 1):
        for r in rs:
            rate = f'{float(r):.1f}'
            key = (rate, int(l_start))
            if key in net_slim_info:
                W_tot, F_tot = net_slim_info[key]
                candidates.append({
                    'key': key,
                    'l_start': int(l_start),
                    'r': rate,
                    'params': float(W_tot),
                    'flops': float(F_tot),
                    'no_prune': False,
                })
    if allow_no_prune and ('0.0', int(l_max)) in net_slim_info:
        W_tot, F_tot = net_slim_info[('0.0', int(l_max))]
        candidates.append({
            'key': ('0.0', int(l_max)),
            'l_start': int(l_max),
            'r': '0.0',
            'params': float(W_tot),
            'flops': float(F_tot),
            'no_prune': True,
        })
    if not candidates:
        raise RuntimeError('No candidate structures are available for capacity-budget projection.')
    return candidates


def fedrasp_latency_table(candidates, client_ids, client_resources, client_n_samples, fl):
    lat = np.zeros((len(client_ids), len(candidates)), dtype=float)
    for i, cid in enumerate(client_ids):
        for j, cand in enumerate(candidates):
            lat[i, j] = fedrasp_round_time_seconds(
                cand['flops'], cand['params'],
                n_samples=int(client_n_samples[cid]),
                client=client_resources[cid],
                round_spec=fl,
            )
    return lat


def fedrasp_continuous_full_times(full_params: float, full_flops: float, client_ids,
                           client_resources, client_n_samples, fl):
    full_times = {}
    for cid in client_ids:
        full_times[int(cid)] = fedrasp_round_time_seconds(
            float(full_flops), float(full_params),
            n_samples=int(client_n_samples[cid]),
            client=client_resources[cid],
            round_spec=fl,
        )
    return full_times


def fedrasp_solve_continuous_budgets(full_params: float, full_flops: float,
                                       client_ids, client_resources,
                                       client_n_samples, fl, args):
    """Solve continuous capacity budgets before any (l_start, r) projection.

    The continuous macro problem is:
        minimize tau
        s.t. rho_i = 1 for the fastest full-model client(s)
             rho_min <= rho_i <= rho_max for the remaining clients
             rho_i * T_i(full) <= tau
             mean_i rho_i >= target_density

    Forced fastest clients make the best system clients train the complete
    model, while the rest still receive heterogeneous continuous budgets that
    are later projected to deployable structures.
    """
    client_ids = [int(cid) for cid in client_ids]
    if not client_ids:
        return {
            'continuous_deadline_s': 0.0,
            'budget_deadline_s': 0.0,
            'target_density': 0.0,
            'avg_budget_density': 0.0,
            'fastest_full_clients': [],
            'budgets': {},
        }

    rho_min = float(np.clip(getattr(args, 'budget_min_density', 0.1), 1e-4, 1.0))
    rho_max = float(np.clip(getattr(args, 'budget_max_density', 1.0), rho_min, 1.0))
    target_density = float(np.clip(getattr(args, 'budget_target_density', 0.5), rho_min, 1.0))
    deadline_slack = float(max(0.0, getattr(args, 'budget_deadline_slack', 0.0)))

    full_times = fedrasp_continuous_full_times(
        full_params, full_flops, client_ids, client_resources, client_n_samples, fl)
    times = np.asarray([max(float(full_times[int(cid)]), 1e-12) for cid in client_ids], dtype=float)

    force_num = int(min(max(0, getattr(args, 'budget_full_fastest_num', 1)), len(client_ids)))
    order = np.argsort(times)
    forced_pos = set(int(x) for x in order[:force_num].tolist())
    remaining_pos = [i for i in range(len(client_ids)) if i not in forced_pos]
    forced_clients = [int(client_ids[i]) for i in sorted(forced_pos)]

    target_total = target_density * float(len(client_ids))
    forced_total = float(len(forced_pos))
    remaining_target = target_total - forced_total
    if remaining_pos:
        rem_min_total = rho_min * float(len(remaining_pos))
        rem_max_total = rho_max * float(len(remaining_pos))
        remaining_target = float(np.clip(remaining_target, rem_min_total, rem_max_total))
        rem_times = times[remaining_pos]

        def remaining_capacities_for_tau(tau: float):
            return np.clip(float(tau) / rem_times, rho_min, rho_max)

        forced_deadline = float(np.max(times[list(forced_pos)])) if forced_pos else 0.0
        tau_low = max(forced_deadline, float(np.max(rem_times * rho_min)))
        tau_high = max(forced_deadline, float(np.max(rem_times * rho_max)))
        if float(remaining_capacities_for_tau(tau_low).sum()) >= remaining_target - 1e-12:
            tau_star = tau_low
        else:
            tau_star = tau_high
            for _ in range(60):
                mid = 0.5 * (tau_low + tau_high)
                if float(remaining_capacities_for_tau(mid).sum()) >= remaining_target - 1e-12:
                    tau_star = mid
                    tau_high = mid
                else:
                    tau_low = mid
    else:
        tau_star = float(np.max(times))

    tau_budget = tau_star * (1.0 + deadline_slack)
    densities = np.zeros(len(client_ids), dtype=float)
    for pos in forced_pos:
        densities[pos] = 1.0
    if remaining_pos:
        densities[np.asarray(remaining_pos, dtype=int)] = np.clip(
            tau_budget / times[remaining_pos], rho_min, rho_max)

    budgets = {}
    for pos, cid in enumerate(client_ids):
        rho = float(densities[pos])
        budgets[int(cid)] = {
            'density': rho,
            'flops': rho * float(full_flops),
            'params': rho * float(full_params),
            'full_time_s': float(times[pos]),
            'continuous_time_s': rho * float(times[pos]),
            'forced_full': bool(pos in forced_pos),
        }

    return {
        'continuous_deadline_s': float(tau_star),
        'budget_deadline_s': float(tau_budget),
        'target_density': float(target_density),
        'avg_budget_density': float(np.mean(densities)) if densities.size else 0.0,
        'min_budget_density': float(np.min(densities)) if densities.size else 0.0,
        'max_budget_density': float(np.max(densities)) if densities.size else 0.0,
        'fastest_full_clients': forced_clients,
        'budgets': budgets,
    }


def fedrasp_project_budget_to_structure(candidates, cand_lat, budget, tau, full_params: float,
                                 full_flops: float, args):
    f_weight = float(getattr(args, 'budget_flops_weight', 0.5))
    w_weight = float(getattr(args, 'budget_params_weight', 0.5))
    norm = max(f_weight + w_weight, 1e-12)
    f_weight /= norm
    w_weight /= norm
    proj_slack = float(max(0.0, getattr(args, 'budget_projection_slack', 0.0)))
    time_slack = float(max(0.0, getattr(args, 'budget_projection_time_slack', 0.0)))
    over_penalty = float(max(0.0, getattr(args, 'budget_over_penalty', 10.0)))
    time_penalty = float(max(0.0, getattr(args, 'budget_time_penalty', 10.0)))

    feasible = []
    for j, cand in enumerate(candidates):
        f_ok = cand['flops'] <= budget['flops'] * (1.0 + proj_slack) + 1e-12
        w_ok = cand['params'] <= budget['params'] * (1.0 + proj_slack) + 1e-12
        t_ok = float(cand_lat[j]) <= tau * (1.0 + time_slack) + 1e-12
        if f_ok and w_ok and t_ok:
            feasible.append(j)
    pool = feasible if feasible else list(range(len(candidates)))

    best_idx, best_score = None, None
    for j in pool:
        cand = candidates[j]
        f_gap = abs(cand['flops'] - budget['flops']) / max(float(full_flops), 1.0)
        w_gap = abs(cand['params'] - budget['params']) / max(float(full_params), 1.0)
        f_over = max(0.0, cand['flops'] - budget['flops']) / max(float(full_flops), 1.0)
        w_over = max(0.0, cand['params'] - budget['params']) / max(float(full_params), 1.0)
        t_over = max(0.0, float(cand_lat[j]) - tau) / max(float(tau), 1e-12)
        score = (f_weight * f_gap + w_weight * w_gap
                 + over_penalty * (f_over + w_over)
                 + time_penalty * t_over)
        tie = (-cand['params'] / max(float(full_params), 1.0),
               -cand['flops'] / max(float(full_flops), 1.0),
               -int(cand['l_start']),
               float(cand['r']))
        key = (float(score),) + tie
        if best_score is None or key < best_score:
            best_idx = int(j)
            best_score = key
    return int(best_idx), float(best_score[0]), bool(feasible)


def fedrasp_allocate_resources(net_slim_info, l_min: int, l_max: int, client_ids, client_resources,
                          client_n_samples, fl, r_min: float, r_max: float, r_step: float,
                          allow_no_prune: bool, args):
    full_params, full_flops = net_slim_info[('0.0', int(l_max))]
    continuous = fedrasp_solve_continuous_budgets(
        float(full_params), float(full_flops), client_ids, client_resources,
        client_n_samples, fl, args)
    tau = float(continuous['budget_deadline_s'])
    budgets = continuous['budgets']

    candidates = fedrasp_candidate_structures(
        net_slim_info, l_min, l_max, r_min, r_max, r_step, allow_no_prune)
    fixed_l_start = getattr(args, 'budget_fixed_l_start', None)
    if fixed_l_start is not None:
        fixed_l_start = int(fixed_l_start)
        if fixed_l_start < int(l_min) or fixed_l_start > int(l_max):
            raise ValueError(
                f'budget_fixed_l_start={fixed_l_start} is outside '
                f'the valid range [{int(l_min)}, {int(l_max)}].')
        candidates = [
            cand for cand in candidates
            if bool(cand.get('no_prune', False)) or int(cand['l_start']) == fixed_l_start
        ]
        if not candidates:
            raise RuntimeError(
                f'No candidate structures are available for fixed pruning start layer '
                f'{fixed_l_start}.')
    lat = fedrasp_latency_table(candidates, client_ids, client_resources, client_n_samples, fl)

    plan = {}
    round_times = []
    for i, cid in enumerate(client_ids):
        cid = int(cid)
        budget = budgets[cid]
        idx, projection_score, feasible = fedrasp_project_budget_to_structure(
            candidates, lat[i, :], budget, tau, float(full_params), float(full_flops), args)
        cand = candidates[idx]
        plan[cid] = {
            'l_start': int(cand['l_start']),
            'r': str(cand['r']),
            'round_time_s': float(lat[i, idx]),
            'kept_param_fraction': float(cand['params'] / max(float(full_params), 1.0)),
            'budget_density': float(budget['density']),
            'budget_params': float(budget['params']),
            'budget_flops': float(budget['flops']),
            'budget_params_fraction': float(budget['params'] / max(float(full_params), 1.0)),
            'budget_flops_fraction': float(budget['flops'] / max(float(full_flops), 1.0)),
            'budget_continuous_time_s': float(budget['continuous_time_s']),
            'forced_full': bool(budget.get('forced_full', False)),
            'projection_score': float(projection_score),
            'projection_feasible': bool(feasible),
        }
        round_times.append(float(lat[i, idx]))

    round_times = np.asarray(round_times, dtype=float)
    return {
        'solver': 'continuous_capacity_budget_projection',
        'objective_waiting_time_s': float(round_times.max()) if round_times.size else 0.0,
        'budget_deadline_s': float(tau),
        'continuous_deadline_s': float(continuous['continuous_deadline_s']),
        'target_density': float(continuous['target_density']),
        'avg_budget_density': float(continuous['avg_budget_density']),
        'min_budget_density': float(continuous['min_budget_density']),
        'max_budget_density': float(continuous['max_budget_density']),
        'fastest_full_clients': list(continuous.get('fastest_full_clients', [])),
        'avg_waiting_time_s': float(round_times.mean()) if round_times.size else 0.0,
        'plan': plan,
        'budgets': budgets,
    }
