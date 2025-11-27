# app/modules/hdl/resource_config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VivadoConfig:
    """
    Configuration for running Vivado resource utilization analysis.
    """

    # Path to Vivado settings script (settings64.sh)
    settings_script: Path

    # Root folder under which per-run Vivado projects will be created
    project_root: Path

    # FPGA part number (can be overridden if needed)
    device_part: str = "xc7vx485tffg1157-1"

    # Number of jobs passed to 'launch_runs'
    jobs: int = 10

    # Safety wait time after Vivado batch completes (seconds)
    # You asked for 2 minutes for now
    post_run_sleep_seconds: int = 120
