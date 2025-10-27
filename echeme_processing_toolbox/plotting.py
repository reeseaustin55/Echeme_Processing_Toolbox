"""Plotting helpers for producing publication-ready SVG figures."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
from .dependencies import np

plt.style.use("seaborn-v0_8-whitegrid")


_DEF_FONT = {"fontname": "Arial", "fontsize": 14}


def _size_from_width(width_mm: float, aspect: float = 0.75) -> tuple[float, float]:
    width_in = width_mm / 25.4
    height_in = width_in * aspect
    return width_in, height_in


def plot_cp_potential(time_h: np.ndarray, potential_v: np.ndarray, std_v: np.ndarray,
                      width_mm: float, output: Path, show_error: bool) -> None:
    fig, ax = plt.subplots(figsize=_size_from_width(width_mm))
    if show_error:
        ax.errorbar(time_h, potential_v, yerr=std_v, fmt="o-", linewidth=2, markersize=6, color=(0.2, 0.2, 0.8))
    else:
        ax.plot(time_h, potential_v, "o-", linewidth=2, markersize=6, color=(0.2, 0.2, 0.8))
    ax.set_xlabel("Time (h)", **_DEF_FONT)
    ax.set_ylabel("E$_{iR\text{-corrected}}$ vs RHE (V)", fontname="Arial", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)


def plot_cp_current(time_h: np.ndarray, current_a: np.ndarray, width_mm: float, output: Path) -> None:
    fig, ax = plt.subplots(figsize=_size_from_width(width_mm))
    ax.plot(time_h, current_a * 1000.0, "o-", linewidth=2, markersize=6, color=(0.8, 0.2, 0.2))
    ax.set_xlabel("Time (h)", **_DEF_FONT)
    ax.set_ylabel("Substrate-corrected i (mA)", **_DEF_FONT)
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)


def plot_eis_resistance(time_h: np.ndarray, mean_rt: np.ndarray, std_rt: np.ndarray,
                        width_mm: float, output: Path, show_error: bool) -> None:
    fig, ax = plt.subplots(figsize=_size_from_width(width_mm))
    if show_error:
        ax.errorbar(time_h, mean_rt, yerr=std_rt, fmt="s-", linewidth=2, markersize=8,
                    color=(0.1, 0.6, 0.1), markerfacecolor="auto")
    else:
        ax.plot(time_h, mean_rt, "s-", linewidth=2, markersize=8, color=(0.1, 0.6, 0.1))
    ax.set_xlabel("Time (h)", **_DEF_FONT)
    ax.set_ylabel("R$_s$ + R$_1$ (Ω)", fontname="Arial", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)


def plot_cv_charges(time_h: np.ndarray, forward: np.ndarray, reverse: np.ndarray,
                    width_mm: float, output: Path, title: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=_size_from_width(width_mm, aspect=0.65))
    max_q = np.nanmax([np.nanmax(np.abs(forward)), np.nanmax(np.abs(reverse)), 1.0])
    f_norm = forward / max_q
    r_norm = reverse / max_q
    ax.plot(time_h, f_norm, "o-", linewidth=1.8, markersize=6, label="Forward Scan")
    ax.plot(time_h, r_norm, "s--", linewidth=1.8, markersize=6, label="Reverse Scan")
    ax.set_xlabel("Time (h)", fontname="Arial", fontsize=12)
    ax.set_ylabel("Norm. Q (C)", fontname="Arial", fontsize=12)
    if title:
        ax.set_title(title, fontname="Arial", fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)
