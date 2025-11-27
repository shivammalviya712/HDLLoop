# app/modules/hdl/resource_utilization_runner.py
from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .resource_config import VivadoConfig
from .resource_report import ResourceReport
from .vivado_resource_parser import VivadoResourceReportParser


@dataclass
class ResourceUtilizationRunner:
    """
    End-to-end runner that:
      1. Creates a Vivado project under project_root
      2. Generates run.tcl with given HDL files
      3. Runs Vivado batch mode via bash (source settings64.sh)
      4. Sleeps a fixed amount (config.post_run_sleep_seconds)
      5. Parses <top_module>_utilization_synth.rpt into ResourceReport
    """

    config: VivadoConfig

    def run_for_files(
        self,
        hdl_files: Iterable[Path],
        top_module: str,
        project_name: Optional[str] = None,
    ) -> ResourceReport:
        """
        Main API.

        :param hdl_files: Verilog/VHDL/SystemVerilog source files
        :param top_module: Top module name (used for report filename)
        :param project_name: Optional Vivado project name; if None, auto-generated.
        """
        hdl_files = [Path(f).resolve() for f in hdl_files]
        if not hdl_files:
            raise ValueError("No HDL files provided to ResourceUtilizationRunner")

        proj_name = project_name or self._default_project_name(top_module)
        project_dir = self._create_project_dir(proj_name)

        print(f"[ResourceUtilizationRunner] Creating Vivado project '{proj_name}' in {project_dir}")
        tcl_path = self._write_run_tcl(
            project_dir=project_dir,
            project_name=proj_name,
            hdl_files=hdl_files,
            top_module=top_module,
        )

        print(f"[ResourceUtilizationRunner] Running Vivado batch with script {tcl_path}")
        self._run_vivado_batch(tcl_path)

        # For now: naive wait (2 minutes by default)
        print(
            f"[ResourceUtilizationRunner] Sleeping {self.config.post_run_sleep_seconds} seconds "
            "to allow synthesis to complete..."
        )
        time.sleep(self.config.post_run_sleep_seconds)

        report_path = self._locate_report(
            project_dir=project_dir,
            project_name=proj_name,
            top_module=top_module,
        )

        print(f"[ResourceUtilizationRunner] Parsing utilization report: {report_path}")
        parser = VivadoResourceReportParser()
        report = parser.parse(report_path)

        print(f"[ResourceUtilizationRunner] Parsed ResourceReport: {report}")
        return report

    # ---------------------- Internals ---------------------- #

    def _default_project_name(self, top_module: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_top = "".join(c if c.isalnum() or c == "_" else "_" for c in top_module)
        return f"{safe_top}_proj_{ts}"

    def _create_project_dir(self, project_name: str) -> Path:
        project_dir = self.config.project_root / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ResourceUtilizationRunner] Project directory: {project_dir}")
        return project_dir

    def _write_run_tcl(
        self,
        project_dir: Path,
        project_name: str,
        hdl_files: Iterable[Path],
        top_module: str,
    ) -> Path:
        """
        Writes run.tcl:
          - create_project
          - add_files
          - launch_runs synth_1
          - wait_on_run/open_run
          - report_utilization -> <project_name>.runs/synth_1/<top_module>_utilization_synth.rpt
        """
        tcl_path = project_dir / "run.tcl"
        files_str = " ".join(str(p) for p in hdl_files)

        print(f"[ResourceUtilizationRunner] Writing run.tcl at {tcl_path}")
        print(f"[ResourceUtilizationRunner] HDL files: {files_str}")

        tcl_lines = [
            # Project variables
            f"set proj_name \"{project_name}\"",
            f"set proj_dir \"{project_dir}\"",
            "",
            # Create project
            f"create_project $proj_name $proj_dir -part {self.config.device_part}",
            "",
            # Add HDL sources
            f"add_files -norecurse {{{files_str}}}",
            "update_compile_order -fileset sources_1",
            "",
            # Launch synthesis
            f"launch_runs synth_1 -jobs {self.config.jobs}",
            "wait_on_run synth_1",
            "open_run synth_1",
            "",
            # Build run + report paths
            f"set run_dir [file join $proj_dir {project_name}.runs synth_1]",
            f"set rpt_path [file join $run_dir {top_module}_utilization_synth.rpt]",
            f"set pb_path [file join $run_dir {top_module}_utilization_synth.pb]",
            "report_utilization -file $rpt_path -pb $pb_path",
            "",
            "exit",
            "",
        ]

        tcl_path.write_text("\n".join(tcl_lines), encoding="utf-8")
        return tcl_path

    def _run_vivado_batch(self, tcl_path: Path) -> None:
        """
        Run Vivado in batch mode via bash, sourcing the settings script first.

        Equivalent shell commands:
          source <settings64.sh>
          vivado -mode batch -source run.tcl
        """
        settings_script = self.config.settings_script

        if not settings_script.exists():
            raise FileNotFoundError(
                f"Vivado settings script not found: {settings_script}"
            )

        cmd = (
            f"source {shlex.quote(str(settings_script))} && "
            f"vivado -mode batch -source {shlex.quote(str(tcl_path))}"
        )

        print(f"[ResourceUtilizationRunner] Vivado command: {cmd}")

        proc = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=str(tcl_path.parent),
            text=True,
            capture_output=True,
        )

        print(f"[ResourceUtilizationRunner] Vivado exited with code {proc.returncode}")

        if proc.stdout:
            print("[ResourceUtilizationRunner] Vivado STDOUT:")
            print(proc.stdout)

        if proc.stderr:
            print("[ResourceUtilizationRunner] Vivado STDERR:")
            print(proc.stderr)

        if proc.returncode != 0:
            raise RuntimeError(
                f"Vivado batch run failed (exit code {proc.returncode}). "
                f"Check STDOUT/STDERR above."
            )

    def _locate_report(
        self,
        project_dir: Path,
        project_name: str,
        top_module: str,
    ) -> Path:
        """
        Expected report path:
          <Project-Path>/<project_name>.runs/synth_1/<top_module>_utilization_synth.rpt
        """
        runs_dir = project_dir / f"{project_name}.runs" / "synth_1"
        report_path = runs_dir / f"{top_module}_utilization_synth.rpt"

        print(f"[ResourceUtilizationRunner] Looking for report at: {report_path}")

        if not report_path.exists():
            raise FileNotFoundError(
                f"Utilization report not found at expected path: {report_path}"
            )

        return report_path
