"""Batch processing pipeline coordinating file discovery, analysis and plotting."""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

from .dependencies import np

from .cp import CPProcessingResult, process_cp
from .cv import CVCharges, integrate_cv_charges
from .eis import EISFitResult, fit_eis_by_cycle
from .io_utils import iter_files, load_table
from .plotting import (
    plot_cp_current,
    plot_cp_potential,
    plot_cv_charges,
    plot_eis_resistance,
)


@dataclass
class ProcessingConfig:
    folder: Path
    calibration_file: Path
    plot_width_mm: float
    dt_minutes: float
    ref_offset: float
    vmin: float
    vmax: float
    off_forward: float
    off_reverse: float
    show_error: bool


LogFunc = Callable[[str], None]


class ProcessingError(RuntimeError):
    """Raised when the batch pipeline cannot continue."""


_DEF_EXTENSIONS = (".mpr", ".txt")


def _categorise_files(files: Sequence[Path]) -> tuple[List[Path], List[Path], List[Path]]:
    eis: List[Path] = []
    cp: List[Path] = []
    cv: List[Path] = []
    for file in files:
        name = file.name.lower()
        if "geis" in name:
            eis.append(file)
        if name.count("cp") >= 2:
            cp.append(file)
        if "_cv_" in name:
            cv.append(file)
    eis.sort(key=lambda p: p.name.lower())
    cp.sort(key=lambda p: p.name.lower())
    cv.sort(key=lambda p: p.name.lower())
    return eis, cp, cv


def _ensure_files(config: ProcessingConfig) -> tuple[Path, Path, List[Path]]:
    files = sorted(iter_files(config.folder, _DEF_EXTENSIONS), key=lambda p: p.name.lower())
    if not files:
        raise ProcessingError("No .mpr or .txt files were found in the selected folder.")

    eis_files, cp_files, cv_files = _categorise_files(files)
    if not eis_files:
        raise ProcessingError("No EIS (GEIS) files detected; cannot perform Rs + R1 fitting.")
    if not cp_files:
        raise ProcessingError("No CP files detected in the folder name patterns.")
    if not cv_files:
        raise ProcessingError("No CV files detected using the '_CV_' naming convention.")

    return eis_files[0], cp_files[0], cv_files


def _prepare_output(config: ProcessingConfig) -> Path:
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = config.folder / f"processed_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _compute_resistance_stats(eis: EISFitResult) -> tuple[np.ndarray, np.ndarray]:
    rs = eis.params[:, 0, :]
    r1 = eis.params[:, 1, :]
    total = rs + r1
    mean = np.nanmean(total, axis=0)
    std = np.nanstd(total, axis=0)
    return mean, std


def run_processing(config: ProcessingConfig, log: LogFunc) -> Path:
    log("Scanning folder for electrochemistry files…")
    eis_path, cp_path, cv_paths = _ensure_files(config)
    log(f"Found EIS file: {eis_path.name}")
    log(f"Found CP file: {cp_path.name}")
    log(f"Found {len(cv_paths)} CV file(s)")

    output_dir = _prepare_output(config)
    log(f"Output directory: {output_dir}")

    log("Fitting EIS data by cycle…")
    eis_table = load_table(eis_path)
    eis_result = fit_eis_by_cycle(eis_table)
    mean_rt, std_rt = _compute_resistance_stats(eis_result)

    log("Processing CP dataset…")
    cp_table = load_table(cp_path)
    calibration_table = load_table(config.calibration_file)
    cp_result = process_cp(cp_table, calibration_table, eis_result, config.dt_minutes, config.ref_offset)

    log("Generating CP plots…")
    plot_cp_potential(
        cp_result.time_hours,
        cp_result.avg_potential,
        cp_result.std_potential,
        config.plot_width_mm,
        output_dir / "cp_ir_corrected_potential.svg",
        config.show_error,
    )
    plot_cp_current(
        cp_result.time_hours,
        cp_result.avg_current,
        config.plot_width_mm,
        output_dir / "cp_substrate_corrected_current.svg",
    )

    log("Creating Rs + R1 plot…")
    plot_eis_resistance(
        eis_result.times_hours,
        mean_rt,
        std_rt,
        config.plot_width_mm,
        output_dir / "eis_rs_plus_r1.svg",
        config.show_error,
    )

    log("Processing CV files…")
    for path in cv_paths:
        log(f"  Integrating charges for {path.name}")
        cv_table = load_table(path)
        charges = integrate_cv_charges(
            cv_table,
            eis_result,
            config.vmin,
            config.vmax,
            config.off_forward,
            config.off_reverse,
        )
        if charges.time_hours.size == 0:
            log(f"    No usable cycles found in {path.name}; skipping plot.")
            continue
        title = path.stem.replace("_", " ")
        plot_cv_charges(
            charges.time_hours,
            charges.forward_charge,
            charges.reverse_charge,
            config.plot_width_mm,
            output_dir / f"{path.stem}_charges.svg",
            title=title,
        )

    log("Processing complete.")
    return output_dir
