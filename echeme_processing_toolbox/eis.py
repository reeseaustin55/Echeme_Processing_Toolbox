"""EIS fitting utilities mirroring the MATLAB workflow."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from .io_utils import NumericTable


@dataclass
class EISFitResult:
    times_hours: np.ndarray
    params: np.ndarray  # shape (n_curves, 7, n_cycles)


def _model_eis(params: np.ndarray, omega: np.ndarray) -> np.ndarray:
    rs, r1, q1, n1, r2, q2, n2 = params
    zc1 = 1.0 / (q1 * (1j * omega) ** n1)
    zp1 = 1.0 / (1.0 / r1 + 1.0 / zc1)
    zc2 = 1.0 / (q2 * (1j * omega) ** n2)
    zp2 = 1.0 / (1.0 / r2 + 1.0 / zc2)
    return rs + zp1 + zp2


def _assign_hf_branch(params: np.ndarray) -> np.ndarray:
    p = params.copy()
    tau1 = (p[1] * p[2]) ** (1.0 / max(p[3], 1e-6))
    tau2 = (p[4] * p[5]) ** (1.0 / max(p[6], 1e-6))
    if tau2 < tau1:
        p[[1, 2, 3, 4, 5, 6]] = p[[4, 5, 6, 1, 2, 3]]
    return p


def _initial_guess(z_obs: np.ndarray) -> np.ndarray:
    rs0 = np.nanmin(np.real(z_obs))
    rt = np.nanmax(np.real(z_obs)) - rs0
    if not np.isfinite(rt) or rt <= 0:
        rt = max(np.nanmax(np.abs(z_obs)), 1.0)
    return np.array([rs0, rt / 4, 1e-3, 0.9, 3 * rt / 4, 1e-3, 0.9], dtype=float)


def _approx_jacobian(func: Callable[[np.ndarray], np.ndarray], params: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    base = func(params)
    jac = np.empty((base.size, params.size), dtype=float)
    for i in range(params.size):
        step = eps * max(1.0, abs(params[i]))
        perturbed = params.copy()
        perturbed[i] += step
        diff = (func(perturbed) - base) / step
        jac[:, i] = np.real(diff)
    return jac


def _least_squares(initial: np.ndarray, residual_func: Callable[[np.ndarray], np.ndarray],
                   max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    params = initial.copy()
    lam = 1e-3
    best_res = residual_func(params)
    best_cost = np.linalg.norm(best_res)

    for _ in range(max_iter):
        jac = _approx_jacobian(residual_func, params)
        try:
            step = np.linalg.solve(jac.T @ jac + lam * np.eye(params.size), -jac.T @ best_res)
        except np.linalg.LinAlgError:
            lam *= 10
            continue

        trial = params + step
        trial[[0, 1, 4]] = np.clip(trial[[0, 1, 4]], 0.0, None)
        trial[[2, 5]] = np.clip(trial[[2, 5]], 1e-9, None)
        trial[[3, 6]] = np.clip(trial[[3, 6]], 0.0, 1.0)

        trial_res = residual_func(trial)
        trial_cost = np.linalg.norm(trial_res)
        if trial_cost < best_cost:
            params = trial
            best_res = trial_res
            best_cost = trial_cost
            lam = max(lam / 5, 1e-9)
            if np.linalg.norm(step) < tol:
                break
        else:
            lam = min(lam * 5, 1e9)
    return params


def fit_eis_by_cycle(table: NumericTable) -> EISFitResult:
    real_z = table.get("rez", "re_z", "re_z_ohm")
    try:
        neg_im = table.get("_imz", "_im_z", "_im_z_ohm")
        imag_component = neg_im
    except KeyError:
        imag_z = table.get("imz", "im_z", "im_z_ohm")
        imag_component = -imag_z
    freq = table.get("freq", "freq_hz")
    cycle = table.get("cycle", "cycle_number")
    time_s = table.get("time", "time_s")

    omega = 2 * math.pi * freq
    z_obs = real_z + 1j * imag_component

    order = np.argsort(time_s)
    z_obs = z_obs[order]
    omega = omega[order]
    cycle = cycle[order]
    time_s = time_s[order]

    fgap = 0.5 * (np.nanmax(freq) - np.nanmin(freq))
    df = np.abs(np.diff(freq[order]))
    dc = np.diff(cycle)
    break_points = np.unique(np.concatenate([np.where(df > fgap)[0], np.where(dc != 0)[0]]))
    starts = np.concatenate([[0], break_points + 1])
    ends = np.concatenate([break_points + 1, [z_obs.size]])

    sweeps: List[Tuple[int, np.ndarray, np.ndarray, float]] = []
    for s, e in zip(starts, ends):
        sweeps.append((int(cycle[s]), omega[s:e], z_obs[s:e], float(np.mean(time_s[s:e]))))

    cycles = sorted({cycle for cycle, *_ in sweeps})
    counts = {c: sum(1 for sweep in sweeps if sweep[0] == c) for c in cycles}
    n_curves = max(counts.values()) if counts else 0
    params = np.full((n_curves, 7, len(cycles)), np.nan, dtype=float)
    times_hours = np.full(len(cycles), np.nan, dtype=float)

    prev_cycle = [np.full(7, np.nan) for _ in range(n_curves)]

    for j, cyc in enumerate(cycles):
        subset = [s for s in sweeps if s[0] == cyc]
        times_hours[j] = np.mean([s[3] for s in subset]) / 3600.0
        for i, (_, omega_seg, z_seg, _) in enumerate(subset):
            if z_seg.size < 5:
                continue
            if j == 0:
                guess = _initial_guess(z_seg) if i == 0 else params[i - 1, :, j]
            else:
                guess = prev_cycle[i] if not np.isnan(prev_cycle[i]).any() else _initial_guess(z_seg)
            guess = np.nan_to_num(guess, nan=1.0)

            def residual(p: np.ndarray) -> np.ndarray:
                modeled = _model_eis(p, omega_seg)
                diff = np.concatenate([
                    (np.real(modeled) - np.real(z_seg)).ravel(),
                    (np.imag(modeled) - np.imag(z_seg)).ravel(),
                ])
                return diff

            fitted = _least_squares(guess, residual)
            fitted = _assign_hf_branch(fitted)
            params[i, :, j] = fitted
        prev_cycle = [params[i, :, j] for i in range(n_curves)]

    return EISFitResult(times_hours=times_hours, params=params)
