"""Chronopotentiometry data processing utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

from .dependencies import np

from .eis import EISFitResult
from .io_utils import NumericTable


@dataclass
class CPProcessingResult:
    time_hours: np.ndarray
    avg_potential: np.ndarray
    std_potential: np.ndarray
    avg_current: np.ndarray


def process_cp(
    table: NumericTable,
    calibration: NumericTable,
    eis: EISFitResult,
    dt_minutes: float,
    ref_offset: float,
) -> CPProcessingResult:
    ewe_raw = table.get("ewe", "ewe_v")
    time_s = table.get("time", "time_s")
    current_ma = table.get("i", "i_ma", "current", "current_ma")

    order = np.argsort(time_s)
    ewe_raw = ewe_raw[order]
    time_s = time_s[order]
    current_ma = current_ma[order]

    cal_ewe = calibration.get("ewe", "ewe_v")
    cal_current_ma = calibration.get("i", "i_ma", "current", "current_ma")
    cal_order = np.argsort(cal_ewe)
    cal_ewe = cal_ewe[cal_order]
    cal_current_ma = cal_current_ma[cal_order]

    substrate_ma = np.interp(ewe_raw, cal_ewe, cal_current_ma, left=cal_current_ma[0], right=cal_current_ma[-1])
    net_current_ma = current_ma - substrate_ma

    ewe = ewe_raw - ref_offset

    rs = eis.params[:, 0, :]
    r1 = eis.params[:, 1, :]
    rtot = np.nanmean(rs + r1, axis=0)

    t_eis_s = eis.times_hours * 3600.0
    ir_drop = np.interp(time_s, t_eis_s, rtot, left=rtot[0], right=rtot[-1])
    current_a = net_current_ma / 1000.0
    v_corr = ewe - current_a * ir_drop

    dt = np.diff(time_s)
    if dt.size == 0:
        raise ValueError("CP data does not contain enough points to compute averages.")
    threshold = 10 * np.nanmedian(dt)
    breaks = np.where(dt > threshold)[0]
    segment_boundaries = np.concatenate([[0], breaks + 1, [time_s.size]])

    window_s = dt_minutes * 60.0
    time_out: list[float] = []
    pot_out: list[float] = []
    std_out: list[float] = []
    current_out: list[float] = []
    offset = 0.0

    for idx in range(len(segment_boundaries) - 1):
        s = segment_boundaries[idx]
        e = segment_boundaries[idx + 1]
        seg_time = time_s[s:e]
        seg_v = v_corr[s:e]
        seg_i = current_a[s:e]

        if seg_time.size == 0:
            continue

        start = seg_time[0]
        stop = seg_time[-1]
        edges = np.arange(start, stop + window_s, window_s)
        if edges.size < 2:
            edges = np.array([start, stop])

        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (seg_time >= lo) & (seg_time < hi)
            if not np.any(mask):
                continue
            t_avg = float(np.mean(seg_time[mask]) - offset)
            time_out.append(t_avg / 3600.0)
            values = seg_v[mask]
            pot_out.append(float(np.mean(values)))
            std_out.append(float(np.std(values)))
            current_out.append(float(np.mean(seg_i[mask])))

        if idx < len(segment_boundaries) - 2:
            gap = time_s[segment_boundaries[idx + 1]] - time_s[segment_boundaries[idx + 1] - 1]
            offset += gap

    return CPProcessingResult(
        time_hours=np.asarray(time_out),
        avg_potential=np.asarray(pot_out),
        std_potential=np.asarray(std_out),
        avg_current=np.asarray(current_out),
    )
