# app/modules/hdl/resource_report.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ResourceReport:
    """
    Minimal, tool-agnostic resource summary.

    tool   : e.g. "vivado"
    target : device / part number (e.g. "xc7vx485tffg1157-1")
    """
    tool: str
    target: str
    lut: Optional[int] = None
    ff: Optional[int] = None
    dsp: Optional[int] = None
    bram: Optional[int] = None
    uram: Optional[int] = None
    fmax_mhz: Optional[float] = None
