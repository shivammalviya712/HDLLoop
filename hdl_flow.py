from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from ResourceUtilizationTool import ResourceUtilizationRunner
from CompileCheckTool import CompileCheckRunner


class HdlOptState(TypedDict, total=False):
    # Inputs to the flow (provided in initial state)
    hdl_paths: List[str]
    query: str
    chat_history: List[BaseMessage]
    system_prompt: str

    # Set by resource_init
    top_name: str
    hdl_bundle: str
    base_summary: str
    messages: List[BaseMessage]

    # Set by agent
    agent_response: BaseMessage

    # Set by compile_check
    compile_ok: bool
    compile_errors: Optional[str]

    # Set by resource_final
    final_summary: str
    summary_text: str


def build_hdl_optimization_graph(
    llm: ChatOpenAI,
    resource_runner: ResourceUtilizationRunner,
    compile_runner: CompileCheckRunner,
):
    """
    Build a LangGraph that does the full HDL optimization flow:

        resource_init -> agent -> compile_check -> resource_final
    """

    graph = StateGraph(HdlOptState)

    # ---------------------- Nodes ---------------------- #

    def resource_init_node(state: HdlOptState) -> Dict[str, Any]:
        path_objs = [Path(p) for p in state["hdl_paths"]]
        if not path_objs:
            raise ValueError("resource_init_node: no HDL paths provided")

        top_name = path_objs[0].stem
        hdl_bundle = _build_hdl_bundle(path_objs)
        base_report = resource_runner.run(file_paths=path_objs, top_name=top_name)
        base_summary = base_report.to_human_summary()

        human_input = _format_design_prompt(
            query=state["query"],
            top_name=top_name,
            base_summary=base_summary,
            hdl_bundle=hdl_bundle,
        )

        messages: List[BaseMessage] = [
            SystemMessage(content=state["system_prompt"]),
            *state["chat_history"],
            HumanMessage(content=human_input),
        ]

        return {
            "top_name": top_name,
            "hdl_bundle": hdl_bundle,
            "base_summary": base_summary,
            "messages": messages,
        }

    # LLM agent node: uses "messages" from state and stores the response
    agent_runnable = (
        RunnableLambda(lambda state: state["messages"])
        | llm
        | RunnableLambda(lambda msg: {"agent_response": msg})
    )

    def compile_check_node(state: HdlOptState) -> Dict[str, Any]:
        path_objs = [Path(p) for p in state["hdl_paths"]]
        if not path_objs:
            raise ValueError("compile_check_node: no HDL paths provided")
        top_name = state["top_name"]
        result = compile_runner.run(file_paths=path_objs, top_name=top_name)
        return {
            "compile_ok": result.ok,
            "compile_errors": result.errors,
        }

    def resource_final_node(state: HdlOptState) -> Dict[str, Any]:
        path_objs = [Path(p) for p in state["hdl_paths"]]
        if not path_objs:
            raise ValueError("resource_final_node: no HDL paths provided")
        top_name = state["top_name"]
        final_report = resource_runner.run(file_paths=path_objs, top_name=top_name)
        final_summary = final_report.to_human_summary()
        summary_text = (
            "\n\n---\n"
            "Resource utilization summary (stub):\n"
            f"- Before optimization: {state['base_summary']}\n"
            f"- After optimization:  {final_summary}\n"
        )
        return {
            "final_summary": final_summary,
            "summary_text": summary_text,
            "top_name": top_name,
        }

    # -------------------- Graph wiring ----------------- #

    graph.add_node("resource_init", resource_init_node)
    graph.add_node("agent", agent_runnable)
    graph.add_node("compile_check", compile_check_node)
    graph.add_node("resource_final", resource_final_node)

    graph.set_entry_point("resource_init")
    graph.add_edge("resource_init", "agent")
    graph.add_edge("agent", "compile_check")
    graph.add_edge("compile_check", "resource_final")
    graph.add_edge("resource_final", END)

    return graph.compile()


def _build_hdl_bundle(hdl_paths: List[Path]) -> str:
    """
    Concatenate all HDL files into one text block, clearly marking
    which file each section came from. This is what we feed to the LLM.
    """
    parts: List[str] = []
    for p in hdl_paths:
        try:
            code = p.read_text()
        except Exception as exc:  # defensive
            parts.append(f"// File: {p.name} (FAILED TO READ: {exc})\n")
            continue
        parts.append(f"// File: {p.name}\n{code}\n")
    return "\n\n".join(parts)


def _format_design_prompt(
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