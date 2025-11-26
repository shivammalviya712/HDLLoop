from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

@dataclass
class ResourceReport:
    tool: str
    target: str
    lut: Optional[int] = None
    ff: Optional[int] = None
    dsp: Optional[int] = None
    bram: Optional[int] = None
    uram: Optional[int] = None
    fmax_mhz: Optional[float] = None
    extra: Dict[str, float] | None = None

    def to_human_summary(self) -> str:
        parts: List[str] = []

        if self.lut is not None:
            parts.append(f"LUTs: {self.lut}")
        if self.ff is not None:
            parts.append(f"FFs: {self.ff}")
        if self.dsp is not None:
            parts.append(f"DSPs: {self.dsp}")
        if self.bram is not None:
            parts.append(f"BRAMs: {self.bram}")
        if self.fmax_mhz is not None:
            parts.append(f"Fmax: {self.fmax_mhz:.1f} MHz")

        return ", ".join(parts) if parts else "No metrics available."


class ResourceUtilizationRunner:
    """
    Thin abstraction over your actual resource analysis flow.

    Replace `run()` with your real implementation:
      - MATLAB + HDL Coder
      - Vivado / Quartus synthesis TCL
      - etc.
    """

    def __init__(self, target: str = "GenericFPGA") -> None:
        self.target = target

    def run(self, file_paths: List[Path], top_name: str) -> ResourceReport:
        # TODO: integrate with your real tooling using all HDL files.
        # For now this is just a stub so the app runs end-to-end.
        dummy_metrics = {
            "lut": 1234,
            "ff": 2345,
            "dsp": 10,
            "bram": 3,
            "fmax_mhz": 180.0,
        }

        return ResourceReport(
            tool="HDL_TOOL_STUB",
            target=self.target,
            lut=dummy_metrics.get("lut"),
            ff=dummy_metrics.get("ff"),
            dsp=dummy_metrics.get("dsp"),
            bram=dummy_metrics.get("bram"),
            fmax_mhz=dummy_metrics.get("fmax_mhz"),
            extra={},
        )
