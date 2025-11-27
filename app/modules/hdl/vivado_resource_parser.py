# app/modules/hdl/vivado_resource_parser.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .resource_report import ResourceReport


class VivadoResourceReportParser:
    """
    Tiny parser for Vivado '<top>_utilization_synth.rpt'.

    We only care about:
      - tool   -> "vivado"
      - target -> Device line in header
      - lut    -> 'Slice LUTs*' Used
      - ff     -> 'Slice Registers' Used
      - dsp    -> 'DSPs' Used
      - bram   -> 'Block RAM Tile' Used
      - uram   -> 'URAM' Used (if present)
    """

    def parse(self, report_path: Path) -> ResourceReport:
        text = report_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        target = self._parse_device(lines) or "unknown"

        lut = self._extract_used(lines, "Slice LUTs*")
        ff = self._extract_used(lines, "Slice Registers")
        dsp = self._extract_used(lines, "DSPs")
        bram = self._extract_used(lines, "Block RAM Tile")
        uram = self._extract_used(lines, "URAM")  # might be None on some devices

        return ResourceReport(
            tool="vivado",
            target=target,
            lut=lut,
            ff=ff,
            dsp=dsp,
            bram=bram,
            uram=uram,
            fmax_mhz=None,  # utilization report doesn't contain fmax
        )

    # ----------------- Internals ----------------- #

    def _parse_device(self, lines: list[str]) -> Optional[str]:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("| Device"):
                # Example: "| Device       : xc7vx485tffg1157-1"
                parts = stripped.split(":", maxsplit=1)
                if len(parts) == 2:
                    return parts[1].strip()
        return None

    def _extract_used(self, lines: list[str], label: str) -> Optional[int]:
        """
        Extract the 'Used' column for a table row whose first cell matches 'label'.

        Row format is like:
          | Slice LUTs*             | 6528 | 0 | 0 | 303600 | 2.15 |
                 ^ label            ^ Used

        We just split by '|' and take 2nd cell as 'Used'.
        """
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            # quick filter, cheaper than splitting everything
            if f"| {label}" not in line:
                continue

            parts = line.split("|")[1:-1]  # drop empty first/last
            cells = [p.strip() for p in parts]
            if not cells:
                continue
            row_label = cells[0]
            if row_label != label:
                continue
            if len(cells) < 2:
                return None

            used_str = cells[1].replace("_", "").strip()
            try:
                return int(used_str)
            except ValueError:
                return None

        return None
