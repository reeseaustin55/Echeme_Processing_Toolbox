"""Cyclic voltammetry utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .eis import EISFitResult
from .io_utils import NumericTable


@dataclass
class CVCharges:
    time_hours: np.ndarray
    forward_charge: np.ndarray
    reverse_charge: np.ndarray


def _ir_correct_potential(time_s: np.ndarray, potential: np.ndarray, current_a: np.ndarray, eis: EISFitResult) -> np.ndarray:
    rs = eis.params[:, 0, :]
    r1 = eis.params[:, 1, :]
    rtot = np.nanmean(rs + r1, axis=0)
    t_eis_s = eis.times_hours * 3600.0
    rt_interp = np.interp(time_s, t_eis_s, rtot, left=rtot[0], right=rtot[-1])
    return potential - current_a * rt_interp


def integrate_cv_charges(
    table: NumericTable,
    eis: EISFitResult,
    vmin: float,
    vmax: float,
    off_forward: float,
    off_reverse: float,
) -> CVCharges:
    ewe = table.get("ewe", "ewe_v")
    time_s = table.get("time", "time_s")
    current = table.get("i", "i_ma", "current")
    cycle = table.columns.get("cycle")
    if cycle is None:
        cycle = table.columns.get("cycle_number")
    if cycle is None:
        raise KeyError("CV data must include a 'cycle' column.")

    order = np.argsort(time_s)
    ewe = ewe[order]
    time_s = time_s[order]
    current = current[order]
    cycle = cycle[order]

    current_a = current / 1000.0 if np.nanmax(np.abs(current)) > 1e-2 else current

    v_corr = _ir_correct_potential(time_s, ewe, current_a, eis)

    unique_cycles = np.unique(cycle.astype(int))
    time_out: List[float] = []
    q_forward: List[float] = []
    q_reverse: List[float] = []

    for cyc in unique_cycles:
        mask = cycle == cyc
        t_seg = time_s[mask]
        v_seg = v_corr[mask]
        i_seg = current_a[mask]
        if t_seg.size < 2:
            continue

        order = np.argsort(t_seg)
        t_seg = t_seg[order]
        v_seg = v_seg[order]
        i_seg = i_seg[order]

        dv_dt = np.gradient(v_seg, t_seg, edge_order=1)
        forward_mask = (dv_dt >= 0) & (v_seg >= vmin) & (v_seg <= vmax)
        reverse_mask = (dv_dt < 0) & (v_seg >= vmin) & (v_seg <= vmax)

        def integrate(mask: np.ndarray) -> float:
            if not np.any(mask):
                return 0.0
            t_sel = t_seg[mask]
            i_sel = i_seg[mask]
            return float(np.trapz(i_sel, t_sel))

        qf = integrate(forward_mask) - off_forward
        qr = integrate(reverse_mask) - off_reverse

        time_out.append(float(np.mean(t_seg) / 3600.0))
        q_forward.append(qf)
        q_reverse.append(qr)

    return CVCharges(
        time_hours=np.asarray(time_out),
        forward_charge=np.asarray(q_forward),
        reverse_charge=np.asarray(q_reverse),
    )
