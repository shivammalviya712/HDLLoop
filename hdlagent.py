from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

from hdl_flow import build_hdl_optimization_graph
from CompileCheckTool import CompileCheckRunner
# NOTE: assuming you exposed both VivadoConfig + ResourceUtilizationRunner
# from ResourceUtilizationTool as per our earlier implementation.
from app.modules.hdl.resource_config import VivadoConfig
from ResourceUtilizationTool import ResourceUtilizationRunner

load_dotenv()

PROMPT_FILE_NAME = "hdl_optimization_prompt.txt"


class MatlabAgent:
    """
    Thin wrapper around the LangGraph HDL optimization flow.

    - Validates HDL file paths.
    - For "no files" / "missing files" cases, calls LLM directly.
    - For valid files, runs the full graph:
          resource_init -> agent -> compile_check -> resource_final
    - Streams ("think", text) and ("text", text) tuples for the UI.
    """

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=os.getenv("CONTRACT_AGENT_MODEL", "gpt-5.2"),
            use_responses_api=True,
            reasoning={"effort": "high", "summary": "auto"},
        )
        self.langfuse_handler = CallbackHandler()

        # -------- Resource Utilization Runner (Vivado-based) -------- #
        vivado_settings = Path(
            "/mathworks/hub/share/apps/HDLTools/Vivado/2024.1-mw-1/"
            "Lin/Vivado/2024.1/settings64.sh"
        )
        vivado_project_root = Path("/tmp/hdlloop_vivado_projects")

        vivado_config = VivadoConfig(
            settings_script=vivado_settings,
            project_root=vivado_project_root,
        )
        self.resource_runner = ResourceUtilizationRunner(config=vivado_config)

        # -------- Compile Check Runner (your existing tool) -------- #
        self.compile_runner = CompileCheckRunner()

        self.system_prompt_text = load_system_prompt()

        # Folder where optimized HDL files will be written per run
        optimized_root = Path("/tmp/hdlloop_optimized_hdl")
        optimized_root.mkdir(parents=True, exist_ok=True)

        # -------- Build LangGraph pipeline -------- #
        self.graph = build_hdl_optimization_graph(
            llm=self.llm,
            resource_runner=self.resource_runner,
            compile_runner=self.compile_runner,
            optimized_root=optimized_root,
        )

    # ------------------------------------------------------------------ #
    # Public entrypoint used by app_ui.py
    # ------------------------------------------------------------------ #

    async def astream_run(
        self,
        query: str,
        chat_history: List[BaseMessage],
        file_paths: Optional[List[str]] = None,
    ) -> AsyncIterator[Tuple[str, str]]:
        """Main entry point called by the UI."""

        if not file_paths:
            async for item in self._run_no_files_flow(query, chat_history):
                yield item
            return

        existing_paths = [str(p) for p in file_paths if Path(p).exists()]
        if not existing_paths:
            async for item in self._run_missing_files_flow(
                query, chat_history, file_paths
            ):
                yield item
            return

        async for item in self._run_full_graph_flow(
            query=query,
            chat_history=chat_history,
            hdl_paths=existing_paths,
        ):
            yield item

    # ------------------------------------------------------------------ #
    # Flows
    # ------------------------------------------------------------------ #

    async def _run_no_files_flow(
        self,
        query: str,
        chat_history: List[BaseMessage],
    ) -> AsyncIterator[Tuple[str, str]]:
        messages = self._build_no_files_messages(query, chat_history)
        async for kind, text in self._stream_llm(messages, flow_name="hdl_no_files"):
            yield kind, text

    async def _run_missing_files_flow(
        self,
        query: str,
        chat_history: List[BaseMessage],
        file_paths: List[str],
    ) -> AsyncIterator[Tuple[str, str]]:
        messages = self._build_missing_paths_messages(query, chat_history, file_paths)
        async for kind, text in self._stream_llm(
            messages, flow_name="hdl_missing_files"
        ):
            yield kind, text

    async def _run_full_graph_flow(
        self,
        query: str,
        chat_history: List[BaseMessage],
        hdl_paths: List[str],
    ) -> AsyncIterator[Tuple[str, str]]:
        """Run the full LangGraph: resource_init -> agent -> compile_check -> resource_final."""
        initial_state: Dict[str, Any] = {
            "hdl_paths": hdl_paths,
            "query": query,
            "chat_history": chat_history,
            "system_prompt": self.system_prompt_text,
        }

        config = {
            "callbacks": [self.langfuse_handler],
            "metadata": {"flow": "hdl_full_optimization_graph"},
        }

        current_thought_idx = 0

        async for event in self.graph.astream_events(
            initial_state,
            config=config,
            version="v1",
        ):
            ev_type = event["event"]
            name = event.get("name")

            # Node start -> "thinking" markers
            if ev_type == "on_chain_start" and name in {
                "resource_init",
                "agent",
                "apply_optimized_hdl",
                "compile_check",
                "resource_final",
            }:
                for msg in self._node_start_thinking(name):
                    yield msg

            # Node end -> summarize results we care about
            if ev_type == "on_chain_end" and name in {
                "resource_init",
                "apply_optimized_hdl",
                "compile_check",
                "resource_final",
            }:
                output = (event.get("data") or {}).get("output") or {}
                for kind, text in self._node_end_messages(name, output):
                    yield kind, text

            # LLM streaming inside the agent node
            if ev_type == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                events, current_thought_idx = self._process_llm_chunk(
                    chunk, current_thought_idx
                )
                for kind, text in events:
                    yield kind, text

    # ------------------------------------------------------------------ #
    # Node helpers
    # ------------------------------------------------------------------ #

    def _node_start_thinking(self, name: str) -> List[Tuple[str, str]]:
        if name == "resource_init":
            text = (
                "➡️ [ResourceUtilization] Running initial synthesis + "
                "resource utilization on your HDL design...\n"
            )
        elif name == "agent":
            text = (
                "➡️ [Agent] Passing HDL + utilization report to the optimization agent...\n"
            )
        elif name == "apply_optimized_hdl":
            text = (
                "➡️ [HDL Files] Saving the optimized HDL suggested by the agent "
                "and wiring it into the second synthesis run...\n"
            )
        elif name == "compile_check":
            text = (
                "➡️ [CompileCheck] Running syntax/compile check on the optimized HDL...\n"
            )
        elif name == "resource_final":
            text = (
                "➡️ [ResourceUtilization] Running final synthesis + "
                "resource utilization after optimization...\n"
            )
        else:
            return []
        return [("think", text)]


    def _format_resource_report_line(
        self,
        report: Dict[str, Any],
        prefix: str,
        top_name: str,
    ) -> str:
        """
        Pretty-print a ResourceReport-like dict:
        {tool, target, lut, dsp, bram}
        """
        tool = report.get("tool", "vivado")
        target = report.get("target", "unknown")
        lut = report.get("lut")
        dsp = report.get("dsp")
        bram = report.get("bram")

        parts = []
        if lut is not None:
            parts.append(f"LUTs={lut}")
        if dsp is not None:
            parts.append(f"DSPs={dsp}")
        if bram is not None:
            parts.append(f"BRAMs={bram}")

        metrics_str = ", ".join(parts) if parts else "no resource data available"
        return (
            f"✅ [ResourceUtilization] {prefix} for top '{top_name}' "
            f"on {tool} target '{target}': {metrics_str}\n"
        )

    def _node_end_messages(
        self,
        name: str,
        output: Dict[str, Any],
    ) -> List[Tuple[str, str]]:
        messages: List[Tuple[str, str]] = []

        if name == "resource_init":
            top_name = output.get("top_name", "")
            # Prefer structured ResourceReport if available
            report = output.get("resource_report")
            if isinstance(report, dict):
                line = self._format_resource_report_line(
                    report, prefix="Initial utilization", top_name=top_name
                )
                messages.append(("think", line))
            else:
                # Backwards-compatible fallback
                base_summary = output.get("base_summary")
                if base_summary:
                    messages.append(
                        (
                            "think",
                            f"✅ [ResourceUtilization] Initial utilization "
                            f"for top '{top_name}': {base_summary}\n\n",
                        )
                    )

        elif name == "apply_optimized_hdl":
            changed_files = output.get("optimized_files") or []
            if changed_files:
                file_list = ", ".join(changed_files)
                messages.append(
                    (
                        "think",
                        f"✅ [HDL Files] Wrote optimized HDL for: {file_list}\n",
                    )
                )
            else:
                messages.append(
                    (
                        "think",
                        "ℹ️ [HDL Files] No explicit optimized files detected in the "
                        "agent response; proceeding with original design files.\n",
                    )
                )

        elif name == "compile_check":
            compile_ok = output.get("compile_ok", True)
            errors = output.get("compile_errors")

            if compile_ok:
                messages.append(
                    (
                        "think",
                        "✅ [CompileCheck] No compilation errors detected.\n",
                    )
                )
            else:
                messages.append(
                    (
                        "think",
                        "⚠️ [CompileCheck] Compilation errors detected:\n"
                        f"{errors or ''}\n",
                    )
                )

        elif name == "resource_final":
            top_name = output.get("top_name", "")
            report = output.get("resource_report")
            if isinstance(report, dict):
                line = self._format_resource_report_line(
                    report, prefix="Final utilization", top_name=top_name
                )
                messages.append(("think", line))
            else:
                final_summary = output.get("final_summary")
                if final_summary:
                    messages.append(
                        (
                            "think",
                            f"✅ [ResourceUtilization] Final utilization "
                            f"for top '{top_name}': {final_summary}\n",
                        )
                    )

            summary_text = output.get("summary_text")
            if summary_text:
                # This is the nice human-facing wrap-up
                messages.append(("text", summary_text))

        return messages

    # ------------------------------------------------------------------ #
    # LLM helpers
    # ------------------------------------------------------------------ #

    async def _stream_llm(
        self,
        messages: List[BaseMessage],
        flow_name: str,
    ) -> AsyncIterator[Tuple[str, str]]:
        """Simple LLM streaming for non-graph flows."""
        config = {
            "callbacks": [self.langfuse_handler],
            "metadata": {"flow": flow_name},
        }

        current_thought_idx = 0

        async for chunk in self.llm.astream(messages, config=config):
            events, current_thought_idx = self._process_llm_chunk(
                chunk, current_thought_idx
            )
            for kind, text in events:
                yield kind, text

    def _process_llm_chunk(
        self,
        chunk: Any,
        current_idx: int,
    ) -> Tuple[List[Tuple[str, str]], int]:
        """
        Convert an OpenAI Responses API chunk into a list of
        ('think', text) / ('text', text) events + updated reasoning index.
        """
        events: List[Tuple[str, str]] = []
        content = getattr(chunk, "content", chunk)

        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")

                if ptype == "reasoning":
                    summary = part.get("summary", [])
                    if isinstance(summary, list):
                        for item in summary:
                            if not isinstance(item, dict) or "text" not in item:
                                continue
                            idx = item.get("index", 0)
                            if idx > current_idx:
                                events.append(("think", "\n\n"))
                                current_idx = idx
                            events.append(("think", item["text"]))

                elif ptype == "text":
                    txt = part.get("text", "")
                    if txt:
                        events.append(("text", txt))

        elif isinstance(content, str) and content:
            events.append(("text", content))

        return events, current_idx

    # ------------------------------------------------------------------ #
    # Prompt helpers
    # ------------------------------------------------------------------ #

    def _build_no_files_messages(
        self,
        query: str,
        chat_history: List[BaseMessage],
    ) -> List[BaseMessage]:
        human_text = (
            "No HDL files are attached.\n\n"
            "Ask the user to upload their HDL project files "
            "(`.v`, `.sv`, `.vhd`, including submodules and packages), "
            "and restate their optimization goal.\n\n"
            f"User request: {query}"
        )

        return [
            SystemMessage(
                content=(
                    "You are an HDL optimization assistant. "
                    "If there is no HDL attached, explain that you need the files "
                    "and ask the user to upload them."
                )
            ),
            *chat_history,
            HumanMessage(content=human_text),
        ]

    def _build_missing_paths_messages(
        self,
        query: str,
        chat_history: List[BaseMessage],
        file_paths: List[str],
    ) -> List[BaseMessage]:
        joined = "\n".join(file_paths)
        human_text = (
            "The HDL file paths provided by the UI do not exist on disk.\n\n"
            "Explain this to the user and ask them to re-upload the HDL files.\n\n"
            f"Paths from UI:\n{joined}\n\n"
            f"User request: {query}"
        )

        return [
            SystemMessage(
                content=(
                    "You are an HDL optimization assistant. "
                    "If file paths are invalid or files cannot be found, "
                    "politely tell the user and ask them to re-upload."
                )
            ),
            *chat_history,
            HumanMessage(content=human_text),
        ]


def load_system_prompt() -> str:
    """
    Load the HDL optimization prompt from a sibling text file.
    Falls back to a simple default if the file is missing/empty.
    """
    prompt_path = Path(__file__).with_name(PROMPT_FILE_NAME)

    if prompt_path.exists():
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text

    return (
        "You are an expert FPGA/ASIC hardware engineer and HDL optimization assistant. "
        "If the optimization goal or constraints are unclear, ask the user clarifying "
        "questions before changing any HDL code."
    )
