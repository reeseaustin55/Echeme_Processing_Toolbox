"""Tkinter GUI entry point for the electrochemistry batch processor."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

from .processor import ProcessingConfig, ProcessingError, run_processing


class ProcessingThread(threading.Thread):
    def __init__(self, config: ProcessingConfig, log_queue: "queue.Queue[str]") -> None:
        super().__init__(daemon=True)
        self.config = config
        self.log_queue = log_queue
        self.output: Path | None = None
        self.error: Exception | None = None

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def run(self) -> None:  # pragma: no cover - UI thread
        try:
            self.output = run_processing(self.config, self.log)
        except Exception as exc:  # noqa: BLE001
            self.error = exc
            self.log_queue.put(f"ERROR: {exc}")


class Application(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Echem Processing Toolbox")
        self.resizable(False, False)

        self.folder_var = tk.StringVar()
        self.calibration_var = tk.StringVar()
        self.width_var = tk.StringVar(value="120")
        self.dt_var = tk.StringVar(value="59")
        self.ref_offset_var = tk.StringVar(value="0.037")
        self.vmin_var = tk.StringVar(value="1.15")
        self.vmax_var = tk.StringVar(value="1.4")
        self.off_fwd_var = tk.StringVar(value="0.003132")
        self.off_rev_var = tk.StringVar(value="0.002525")
        self.error_var = tk.BooleanVar(value=True)

        self._build_form()
        self.log_widget = scrolledtext.ScrolledText(self, width=80, height=18, state="disabled")
        self.log_widget.grid(row=6, column=0, columnspan=3, padx=10, pady=(5, 10))

        self.thread: ProcessingThread | None = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.after(200, self._poll_log_queue)

    # UI construction -------------------------------------------------
    def _build_form(self) -> None:
        pad = {"padx": 10, "pady": 5, "sticky": "ew"}

        tk.Label(self, text="Data folder:").grid(row=0, column=0, **pad)
        tk.Entry(self, textvariable=self.folder_var, width=45).grid(row=0, column=1, **pad)
        tk.Button(self, text="Browse…", command=self._choose_folder).grid(row=0, column=2, **pad)

        tk.Label(self, text="Calibration file:").grid(row=1, column=0, **pad)
        tk.Entry(self, textvariable=self.calibration_var, width=45).grid(row=1, column=1, **pad)
        tk.Button(self, text="Browse…", command=self._choose_calibration).grid(row=1, column=2, **pad)

        tk.Label(self, text="Plot width (mm):").grid(row=2, column=0, **pad)
        tk.Entry(self, textvariable=self.width_var, width=10).grid(row=2, column=1, sticky="w", padx=10)

        tk.Label(self, text="CP window (min):").grid(row=2, column=1, sticky="e", padx=(0, 120))
        tk.Entry(self, textvariable=self.dt_var, width=10).grid(row=2, column=2, sticky="w", padx=10)

        tk.Label(self, text="Ref. offset (V):").grid(row=3, column=0, **pad)
        tk.Entry(self, textvariable=self.ref_offset_var, width=10).grid(row=3, column=1, sticky="w", padx=10)

        tk.Label(self, text="CV window Vmin/Vmax:").grid(row=3, column=1, sticky="e", padx=(0, 120))
        entry_frame = tk.Frame(self)
        entry_frame.grid(row=3, column=2, sticky="w", padx=10)
        tk.Entry(entry_frame, textvariable=self.vmin_var, width=8).pack(side=tk.LEFT)
        tk.Label(entry_frame, text="/").pack(side=tk.LEFT, padx=2)
        tk.Entry(entry_frame, textvariable=self.vmax_var, width=8).pack(side=tk.LEFT)

        tk.Label(self, text="CV offsets (forward/reverse):").grid(row=4, column=0, columnspan=2, **pad)
        offset_frame = tk.Frame(self)
        offset_frame.grid(row=4, column=2, sticky="w", padx=10)
        tk.Entry(offset_frame, textvariable=self.off_fwd_var, width=8).pack(side=tk.LEFT)
        tk.Label(offset_frame, text="/").pack(side=tk.LEFT, padx=2)
        tk.Entry(offset_frame, textvariable=self.off_rev_var, width=8).pack(side=tk.LEFT)

        tk.Checkbutton(self, text="Show error bars", variable=self.error_var).grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        tk.Button(self, text="Start Processing", command=self._start_processing).grid(row=5, column=2, padx=10, pady=5, sticky="e")

    # Actions ---------------------------------------------------------
    def _choose_folder(self) -> None:  # pragma: no cover - UI callback
        folder = filedialog.askdirectory(title="Select data folder")
        if folder:
            self.folder_var.set(folder)

    def _choose_calibration(self) -> None:  # pragma: no cover - UI callback
        file = filedialog.askopenfilename(title="Select calibration file", filetypes=[("Data files", "*.txt *.mpr"), ("All files", "*.*")])
        if file:
            self.calibration_var.set(file)

    def _validate_float(self, value: str, name: str) -> float:
        try:
            return float(value)
        except ValueError as exc:  # pragma: no cover - validation
            raise ProcessingError(f"Invalid value for {name}: '{value}'") from exc

    def _start_processing(self) -> None:  # pragma: no cover - UI callback
        if self.thread and self.thread.is_alive():
            messagebox.showinfo("Processing", "A processing job is already running.")
            return
        folder = Path(self.folder_var.get())
        calibration = Path(self.calibration_var.get())
        if not folder.exists():
            messagebox.showerror("Error", "Please select a valid data folder.")
            return
        if not calibration.exists():
            messagebox.showerror("Error", "Please select a valid calibration file.")
            return

        config = ProcessingConfig(
            folder=folder,
            calibration_file=calibration,
            plot_width_mm=self._validate_float(self.width_var.get(), "plot width"),
            dt_minutes=self._validate_float(self.dt_var.get(), "CP window"),
            ref_offset=self._validate_float(self.ref_offset_var.get(), "reference offset"),
            vmin=self._validate_float(self.vmin_var.get(), "CV Vmin"),
            vmax=self._validate_float(self.vmax_var.get(), "CV Vmax"),
            off_forward=self._validate_float(self.off_fwd_var.get(), "CV forward offset"),
            off_reverse=self._validate_float(self.off_rev_var.get(), "CV reverse offset"),
            show_error=bool(self.error_var.get()),
        )

        self._append_log("Starting processing…")
        self.thread = ProcessingThread(config, self.log_queue)
        self.thread.start()

    def _append_log(self, message: str) -> None:
        self.log_widget.configure(state="normal")
        self.log_widget.insert(tk.END, message + "\n")
        self.log_widget.configure(state="disabled")
        self.log_widget.see(tk.END)

    def _poll_log_queue(self) -> None:  # pragma: no cover - UI loop
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._append_log(message)
        self.after(200, self._poll_log_queue)


def main() -> None:  # pragma: no cover - manual execution helper
    app = Application()
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()
