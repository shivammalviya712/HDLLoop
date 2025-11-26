from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

from ResourceUtilizationTool import ResourceUtilizationRunner

load_dotenv()

PROMPT_FILE_NAME = "hdl_optimization_prompt.txt"


class MatlabAgent:
    """
    Agent that:
      - takes multi-file HDL designs,
      - runs a resource utilization pass,
      - forwards code + report + user query to an LLM with an HDL optimization prompt,
      - streams thoughts + answer back to the UI.
    """

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=os.getenv("CONTRACT_AGENT_MODEL", "gpt-5.1"),
            use_responses_api=True,
            reasoning={"effort": "high", "summary": "auto"},
        )
        self.resource_runner = ResourceUtilizationRunner(target="GenericFPGA")
        self.system_prompt_text = load_system_prompt()

    # ----------------- Public API ----------------- #

    async def astream_run(
        self,
        query: str,
        chat_history: List[BaseMessage],
        file_paths: Optional[List[str]] = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """
        Main entrypoint called by the UI.

        - `file_paths`: list of HDL file paths (top + submodules + packages).
        - Builds messages based on what files are available.
        - Streams ("think", text) and ("text", text) tuples.
        """
        messages = self._build_messages(query, chat_history, file_paths)
        async for kind, text in self._stream_llm(messages):
            yield kind, text

    # ----------------- Message building ----------------- #

    def _build_messages(
        self,
        query: str,
        chat_history: List[BaseMessage],
        file_paths: Optional[List[str]],
    ) -> List[BaseMessage]:
        """
        Decide which prompt to send:
          - no files at all,
          - file paths invalid,
          - or full multi-file design prompt.
        """
        if not file_paths:
            return self._build_no_files_messages(query, chat_history)

        existing_paths = self._collect_existing_paths(file_paths)
        if not existing_paths:
            return self._build_missing_paths_messages(query, chat_history, file_paths)

        top_name, hdl_bundle, base_summary = self._prepare_design_prompt_inputs(
            existing_paths
        )

        full_input = self._format_design_prompt(
            query=query,
            top_name=top_name,
            base_summary=base_summary,
            hdl_bundle=hdl_bundle,
        )

        return [
            SystemMessage(content=self.system_prompt_text),
            *chat_history,
            HumanMessage(content=full_input),
        ]

    def _build_no_files_messages(
        self,
        query: str,
        chat_history: List[BaseMessage],
    ) -> List[BaseMessage]:
        """
        Ask user to upload their HDL if we literally have none.
        """
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
        """
        If UI passed file paths but nothing exists on disk (e.g. temp cleanup).
        """
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

    # ----------------- Design preparation helpers ----------------- #

    def _collect_existing_paths(self, file_paths: List[str]) -> List[Path]:
        """
        Convert raw strings from the UI into Path objects
        and filter out anything that doesn't exist.
        """
        existing: List[Path] = []
        for path_str in file_paths:
            p = Path(path_str)
            if p.exists():
                existing.append(p)
        return existing

    def _prepare_design_prompt_inputs(
        self,
        hdl_paths: List[Path],
    ) -> tuple[str, str, str]:
        """
        Given a list of HDL files:
          - choose a top module name (simple heuristic: stem of first file),
          - build a unified text bundle with file headers,
          - run resource utilization and return a human-readable summary.
        """
        top_name = hdl_paths[0].stem
        hdl_bundle = self._build_hdl_bundle(hdl_paths)
        base_report = self.resource_runner.run(file_paths=hdl_paths, top_name=top_name)
        base_summary = base_report.to_human_summary()
        return top_name, hdl_bundle, base_summary

    def _build_hdl_bundle(self, hdl_paths: List[Path]) -> str:
        """
        Concatenate all HDL files into one text block, clearly marking
        which file each section came from. This is what we feed to the LLM.
        """
        parts: List[str] = []

        for p in hdl_paths:
            try:
                code = p.read_text()
            except Exception as exc:
                parts.append(f"// File: {p.name} (FAILED TO READ: {exc})\n")
                continue

            parts.append(f"// File: {p.name}\n{code}\n")

        return "\n\n".join(parts)

    def _format_design_prompt(
        self,
        query: str,
        top_name: str,
        base_summary: str,
        hdl_bundle: str,
    ) -> str:
        """
        Final human message template sent along with the system prompt.
        """
        return f"""
            You are given an HDL design that spans multiple files (top module, submodules, packages).
            Optimize it according to the user goal while respecting the optimization rules from the system prompt.

            Top-level module name (heuristic): {top_name}

            Resource utilization BEFORE optimization:
            {base_summary}

            HDL design (all files bundled):
            ```verilog
            {hdl_bundle}
            ```
            User goal / request:
            {query}
        """

    # ----------------- LLM streaming ----------------- #

    async def _stream_llm(
        self,
        messages: List[BaseMessage],
    ) -> AsyncIterator[tuple[str, str]]:
        """
        Stream responses from the LLM, yielding:
        - ("think", text) for reasoning traces,
        - ("text", text) for user-visible answer chunks.
        """
        config = {
            "callbacks": [CallbackHandler()],
            "metadata": {"thread_id": "1"},
        }

        current_thought_idx = 0

        async for chunk in self.llm.astream(messages, config=config):
            content = chunk.content

            # Responses API: list of typed parts
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue

                    part_type = part.get("type")

                    # 1) Internal reasoning
                    if part_type == "reasoning":
                        summary = part.get("summary", [])
                        if isinstance(summary, list):
                            for item in summary:
                                if isinstance(item, dict) and "text" in item:
                                    item_idx = item.get("index", 0)

                                    # Separate blocks of thoughts with a blank line
                                    if item_idx > current_thought_idx:
                                        yield ("think", "\n\n")
                                        current_thought_idx = item_idx

                                    yield ("think", item["text"])

                    # 2) Final visible text
                    elif part_type == "text":
                        text_val = part.get("text", "")
                        if text_val:
                            yield ("text", text_val)

            # Fallback for plain-string content
            elif isinstance(content, str) and content:
                yield ("text", content)

def load_system_prompt() -> str:
    """
    Load the HDL optimization prompt from a sibling text file.

    Falls back to a minimal prompt if the file is missing or empty,
    so the agent still works in dev environments.
    """
    prompt_path = Path(__file__).with_name(PROMPT_FILE_NAME)

    try:
        text = prompt_path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("Prompt file is empty")
        return text
    except Exception as exc:
        # Fallback: short, safe prompt so nothing crashes
        return (
            "You are an expert FPGA/ASIC hardware engineer and HDL optimization assistant. "
            "If the optimization goal or constraints are unclear, ask the user clarifying "
            "questions before changing any HDL code.\n\n"
            f"[Warning: failed to load {PROMPT_FILE_NAME}: {exc}]"
        )
