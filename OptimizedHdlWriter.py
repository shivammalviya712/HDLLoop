from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import re


@dataclass
class OptimizedHdlResult:
    """
    Result of writing optimized HDL to disk.

    - run_dir: folder where files were written
    - all_paths: list of all HDL paths to use for downstream steps
    - changed_files: file names that were explicitly updated by the agent
    """
    run_dir: Path
    all_paths: List[Path]
    changed_files: List[str]


@dataclass
class OptimizedHdlWriter:
    """
    Responsible for:
      1) Creating a per-run output directory.
      2) Copying original HDL files into that directory.
      3) Parsing the agent's response for per-file HDL blocks.
      4) Overwriting / adding files with the optimized HDL.

    The agent is expected to emit blocks like:

        ```vhdl
        -- File: mul_add.vhd (UPDATED)
        ...
        ```

    or

        ```verilog
        // File: my_submodule.v (UPDATED)
        ...
        ```
    """

    output_root: Path

    def __post_init__(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def write_from_agent_response(
        self,
        agent_response: Any,
        original_paths: List[Path],
        top_name: str,
    ) -> OptimizedHdlResult:
        """
        Extract optimized HDL from the agent response and write to disk.

        NOTE: If we cannot find ANY optimized file blocks, we raise an
        error instead of silently falling back to the original files.
        """
        text = self._extract_text(agent_response)
        run_dir = self._create_run_dir(top_name)

        # 1) Copy original files into run_dir
        copied_paths = self._copy_original_files(original_paths, run_dir)

        # 2) Parse optimized blocks from the agent response
        file_blocks = self._parse_file_blocks(text)
        changed_files = list(file_blocks.keys())

        if not file_blocks:
            # No more "patch" or fallback here
            raise ValueError(
                "OptimizedHdlWriter: No optimized HDL file blocks found in the "
                "agent response. Make sure each changed file is emitted as:\n"
                "  ```vhdl\n"
                "  -- File: <name>.vhd (UPDATED)\n"
                "  <code>\n"
                "  ```\n"
                "or\n"
                "  ```verilog\n"
                "  // File: <name>.v (UPDATED)\n"
                "  <code>\n"
                "  ```"
            )

        # 3) Overwrite / add files with optimized HDL
        all_paths = list(copied_paths)
        for filename, code in file_blocks.items():
            dest = run_dir / filename
            dest.write_text(code, encoding="utf-8")
            if dest not in all_paths:
                all_paths.append(dest)

        return OptimizedHdlResult(
            run_dir=run_dir,
            all_paths=all_paths,
            changed_files=changed_files,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _extract_text(self, agent_response: Any) -> str:
        """
        Normalize ChatOpenAI / Responses API style content into a plain string.
        """
        content = getattr(agent_response, "content", agent_response)

        # Simple string
        if isinstance(content, str):
            return content

        # List of parts (Responses API style)
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                # Dict-style part
                if isinstance(item, dict):
                    # New Responses API patterns
                    if item.get("type") in {"text", "output_text"}:
                        txt = (
                            item.get("text")
                            or item.get("output_text", {}).get("content", "")
                        )
                        if txt:
                            parts.append(txt)
                    continue

                # Object with .text attribute
                txt = getattr(item, "text", None)
                if txt:
                    parts.append(txt)

            if parts:
                return "".join(parts)

        # Fallback
        return str(content)

    def _create_run_dir(self, top_name: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_top = "".join(
            c if c.isalnum() or c == "_" else "_"
            for c in top_name
        )
        run_dir = self.output_root / f"{safe_top}_optimized_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _copy_original_files(
        self,
        original_paths: List[Path],
        run_dir: Path,
    ) -> List[Path]:
        results: List[Path] = []
        for src in original_paths:
            dest = run_dir / src.name
            try:
                dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                results.append(dest)
            except OSError:
                # If a file cannot be copied, skip it
                continue
        return results

    def _parse_file_blocks(self, text: str) -> Dict[str, str]:
        """
        Look for blocks of the form:

            ```vhdl
            -- File: mul_add.vhd (UPDATED)
            <code...>
            ```

        or

            ```verilog
            // File: my_submodule.v (UPDATED)
            <code...>
            ```

        Returns { "mul_add.vhd": "<full code>" }.
        """
        blocks: Dict[str, str] = {}

        # Capture content inside ```verilog / ```systemverilog / ```vhdl fences
        code_fence_pattern = re.compile(
            r"```(?:verilog|systemverilog|vhdl)?\s*(.*?)```",
            re.DOTALL | re.IGNORECASE,
        )

        # Accept both Verilog and VHDL comment headers
        file_header_pattern = re.compile(
            r"^(?://|--)\s*File:\s*(.+)$",
            re.IGNORECASE,
        )

        for block in code_fence_pattern.findall(text):
            filename = None
            lines = block.splitlines()
            cleaned_lines: List[str] = []

            for line in lines:
                stripped = line.strip()

                if filename is None:
                    m = file_header_pattern.match(stripped)
                    if m:
                        file_part = m.group(1).strip()
                        # Example: "mul_add.vhd (UPDATED)" or "my_submodule.v (UPDATED)"
                        if "(" in file_part:
                            file_part = file_part.split("(", 1)[0].strip()
                        filename = file_part

                cleaned_lines.append(line)

            if filename:
                code = "\n".join(cleaned_lines).strip() + "\n"
                blocks[filename] = code

        return blocks
