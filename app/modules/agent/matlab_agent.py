from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncIterator, List, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

load_dotenv()

# Where to dump raw chunk/event data for debugging
CHUNK_LOG_PATH = Path(__file__).resolve().parents[3] / "chunk_dump.log"

class MatlabAgent:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=os.getenv("CONTRACT_AGENT_MODEL", "gpt-5.1"),
            use_responses_api=True,
            reasoning={"effort": "high", "summary": "auto"},
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Senior MATLAB Expert."),
            ("user", "{input}"),
        ])

    async def astream_run(
        self, query: str, file_path: str | None = None
    ) -> AsyncIterator[tuple[str, str]]:
        """
        Stream:
        - ("think", ...) for model reasoning (summary stream)
        - ("text", ...) for final visible answer
        """
        context = (
            f"File Context: {os.path.abspath(file_path)}"
            if file_path
            else "No file context."
        )
        full_input = f"{context}\n\nUser Query: {query}"

        config = {
            "callbacks": [CallbackHandler()],
            "metadata": {"thread_id": "1"},
        }

        # Build LC messages from the prompt template
        messages = self.prompt.format_messages(input=full_input)

        # Cache per-summary-index text so we only stream new characters
        reasoning_cache: dict[int, str] = {}

        async for event in self.llm.astream_events(messages, config=config):
            if event.get("event") != "on_chat_model_stream":
                continue

            chunk = event["data"]["chunk"]
            # Dump full chunk event payload for debugging/inspection
            try:
                with CHUNK_LOG_PATH.open("a", encoding="utf-8") as f:
                    f.write(f"{event!r}\n\n")
            except Exception:
                # Don't break the stream if logging fails
                pass

            content = getattr(chunk, "content", None)

            # ----- THINKING: parse reasoning parts -----
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue

                    if part.get("type") != "reasoning":
                        continue

                    summary = part.get("summary", [])
                    if not isinstance(summary, list):
                        continue

                    for s in summary:
                        if not isinstance(s, dict):
                            continue

                        idx = s.get("index", 0)
                        text = s.get("text", "")
                        if not isinstance(text, str) or not text:
                            continue

                        prev = reasoning_cache.get(idx, "")

                        # Compute new suffix vs previous text for this index
                        common_prefix_len = 0
                        for a, b in zip(prev, text):
                            if a == b:
                                common_prefix_len += 1
                            else:
                                break

                        new_part = text[common_prefix_len:]
                        reasoning_cache[idx] = text

                        if new_part:
                            # Stream just the new characters of the reasoning
                            yield ("think", new_part)

            # ----- ANSWER TEXT: normal visible output -----
            text_piece = self._extract_text_from_chunk(chunk)
            if text_piece:
                yield ("text", text_piece)

    def _extract_text_from_chunk(self, chunk: Any) -> str:
        """
        Extract plain answer text from chunk.content.
        Skips reasoning parts entirely.
        """
        content = getattr(chunk, "content", "")

        # Sometimes it's just a string
        if isinstance(content, str):
            return content

        # responses API style: list of parts
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue

                # Skip reasoning; we already handle that above
                if part.get("type") == "reasoning":
                    continue

                text_value = part.get("text")
                if isinstance(text_value, str) and text_value:
                    parts.append(text_value)

            return "".join(parts)

        return ""