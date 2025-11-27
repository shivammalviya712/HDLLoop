# ResourceUtilizationTool.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any, List

from app.modules.hdl.resource_config import VivadoConfig
from app.modules.hdl.resource_report import ResourceReport
from app.modules.hdl.vivado_resource_parser import VivadoResourceReportParser
from app.modules.hdl.resource_utilization_runner import (
    ResourceUtilizationRunner as _VivadoRunner,
)

__all__ = [
    "VivadoConfig",
    "ResourceReport",
    "VivadoResourceReportParser",
    "ResourceUtilizationRunner",
]


class ResourceUtilizationRunner(_VivadoRunner):
    """
    Public runner used by the LangGraph / agent.

    Adds a .run(...) method compatible with the old stub API used in hdl_flow:
        resource_runner.run(file_paths=..., top_name=...)
    """

    def __init__(self, config: VivadoConfig) -> None:
        super().__init__(config=config)

    def run(
        self,
        file_paths: Iterable[Any],
        top_name: str,
    ) -> ResourceReport:
        """
        Backwards-compatible API for existing nodes.

        hdl_flow calls:
            resource_runner.run(file_paths=path_objs, top_name=top_name)

        We accept strings or Path objects and forward to run_for_files().
        """
        paths = [Path(p) for p in file_paths]
        return self.run_for_files(hdl_files=paths, top_module=top_name)

    # Optional helper if you want a convenience in tests / scripts
    def run_for_paths(self, hdl_paths: List[str], top_module: str) -> ResourceReport:
        paths = [Path(p) for p in hdl_paths]
        return self.run_for_files(hdl_files=paths, top_module=top_module)
