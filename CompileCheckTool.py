from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CompileCheckResult:
    ok: bool
    errors: Optional[str] = None

    def to_human_summary(self) -> str:
        if self.ok:
            return "Compilation succeeded (stub – no real checks performed yet)."
        if self.errors:
            return f"Compilation failed (stub). Errors:\n{self.errors}"
        return "Compilation failed (stub)."
    

class CompileCheckRunner:
    """
    Thin abstraction over your actual compile/simulate flow.

    Replace `run()` with your real implementation:
      - MATLAB HDL Workflow / HDL Verifier
      - Vendor synthesis / simulation tools
      - etc.
    """

    def __init__(self, name: str = "StubCompileCheck") -> None:
        self.name = name

    def run(self, file_paths: List[Path], top_name: str) -> CompileCheckResult:
        # TODO: integrate with your real compile/simulate tooling.
        # For now this always returns success so the rest of the
        # LangGraph pipeline can be wired and tested end-to-end.
        return CompileCheckResult(ok=True, errors=None)
