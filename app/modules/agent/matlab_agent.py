from __future__ import annotations

import os
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langfuse.langchain import CallbackHandler

from app.modules.ingestion.matlab_ingestor import MatlabIngestor
from app.modules.memory.vector_memory import VectorMemory

# Ensure environment variables (OpenAI, Langfuse, etc.) are available.
load_dotenv()


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    query: str
    file_path: str
    context_chunks: List[str]
    full_code_context: str
    generated_code: str


class MatlabAgent:
    """LangGraph-powered agent that orchestrates ingestion, retrieval, and generation."""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.ingestor = MatlabIngestor()
        self.memory = VectorMemory()
        self.graph = self._build_graph()

    def run(self, query: str, file_path: str) -> str:
        """
        Execute the LangGraph pipeline for a given query and MATLAB file.

        Args:
            query: User request describing the refactor/change.
            file_path: Absolute path to the MATLAB file to modify.

        Returns:
            The generated MATLAB code produced by the LLM node.
        """
        initial_state: AgentState = {
            "query": query,
            "file_path": os.path.abspath(file_path),
        }
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()

        result_state: AgentState = self.graph.invoke(
            initial_state,
            config={"callbacks": [langfuse_handler]})
        return result_state.get("generated_code", "")

    def _build_graph(self):
        """Create and compile the linear LangGraph pipeline."""
        builder = StateGraph(AgentState)
        builder.add_node("_ingest_and_index", self._ingest_and_index)
        builder.add_node("_retrieve", self._retrieve)
        builder.add_node("_generate", self._generate)

        builder.set_entry_point("_ingest_and_index")
        builder.add_edge("_ingest_and_index", "_retrieve")
        builder.add_edge("_retrieve", "_generate")
        builder.add_edge("_generate", END)
        return builder.compile()

    # Node implementations -------------------------------------------------

    def _ingest_and_index(self, state: AgentState) -> AgentState:
        """Read MATLAB file, chunk content, and load it into the vector store."""
        file_path = state["file_path"]
        chunks = self.ingestor.ingest_file(file_path)
        chunk_texts = [self._chunk_to_text(chunk) for chunk in chunks]
        self.memory.index_chunks(chunk_texts)
        return {
            "full_code_context": "\n\n".join(chunk_texts),
        }

    def _retrieve(self, state: AgentState) -> AgentState:
        """Grab the top-k relevant chunks for the query."""
        query = state["query"]
        context_chunks = self.memory.similarity_search(query, k=3)
        return {"context_chunks": context_chunks}

    def _generate(self, state: AgentState) -> AgentState:
        """Call the LLM with a strong MATLAB-focused system prompt."""
        system_prompt = (
            "You are a MATLAB Expert. "
            "Use 1-based indexing. "
            "Prefer vectorization. "
            "Output ONLY the MATLAB code block."
        )
        chunks_text = "\n\n".join(state.get("context_chunks", []))
        full_context = state.get("full_code_context", "")
        user_prompt = (
            f"User request:\n{state['query']}\n\n"
            f"Relevant snippets:\n{chunks_text}\n\n"
            f"Full file context (may be truncated):\n{full_context}\n\n"
            "Return only the updated MATLAB code."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        generated_code = getattr(response, "content", "")
        return {"generated_code": generated_code}

    @staticmethod
    def _chunk_to_text(chunk: Any) -> str:
        """Normalize diverse chunk objects into plain strings."""
        if isinstance(chunk, str):
            return chunk
        if hasattr(chunk, "text"):
            return getattr(chunk, "text")
        return str(chunk)

