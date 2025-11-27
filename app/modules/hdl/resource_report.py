# app/modules/hdl/resource_report.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ResourceReport:
    """
    Minimal, tool-agnostic resource summary.

    tool   : e.g. "vivado"
    target : device / part number (e.g. "xc7vx485tffg1157-1")
    lut    : LUT usage (Slice LUTs*)
    dsp    : DSP usage
    bram   : Block RAM Tile usage
    """

    tool: str
    target: str

    lut: Optional[int] = None
    dsp: Optional[int] = None
    bram: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Convenience helpers for existing graph / UI
    # ------------------------------------------------------------------ #

    def to_human_summary(self) -> str:
        """
        Backwards-compatible summary used by hdl_flow.resource_init_node
        (base_summary = base_report.to_human_summary()).
        """
        lut_str = str(self.lut) if self.lut is not None else "N/A"
        dsp_str = str(self.dsp) if self.dsp is not None else "N/A"
        bram_str = str(self.bram) if self.bram is not None else "N/A"

        return (
            f"Target '{self.target}' ({self.tool}): "
            f"LUTs={lut_str}, DSPs={dsp_str}, BRAMs={bram_str}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Handy if you want to pass a JSON-serializable view
        into LangGraph / LLM state.
        """
        return {
            "tool": self.tool,
            "target": self.target,
            "lut": self.lut,
            "dsp": self.dsp,
            "bram": self.bram,
        }
