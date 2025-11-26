from __future__ import annotations

import os
from typing import AsyncIterator

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

load_dotenv()

class MatlabAgent:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=os.getenv("CONTRACT_AGENT_MODEL", "gpt-5.1"),
            use_responses_api=True, # This enables the list-based content structure
            reasoning={"effort": "high", "summary": "auto"}, # Uncomment if your specific model config needs this
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Senior MATLAB Expert."),
            ("user", "{input}"),
        ])

    async def astream_run(
        self, query: str, file_path: str | None = None
    ) -> AsyncIterator[tuple[str, str]]:
        
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

        messages = self.prompt.format_messages(input=full_input)
        
        # Track the index of the thought to know when to insert newlines
        current_thought_idx = 0

        async for chunk in self.llm.astream(messages, config=config):
            content = chunk.content

            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    
                    part_type = part.get("type")

                    # --- 1. HANDLE THOUGHTS ---
                    if part_type == "reasoning":
                        summary = part.get("summary", [])
                        if isinstance(summary, list):
                            for item in summary:
                                if isinstance(item, dict) and "text" in item:
                                    # Check if this is a new thought step
                                    item_idx = item.get("index", 0)
                                    
                                    # If index increased, insert a double newline for clean separation
                                    if item_idx > current_thought_idx:
                                        yield ("think", "\n\n")
                                        current_thought_idx = item_idx
                                    
                                    yield ("think", item["text"])

                    # --- 2. HANDLE FINAL ANSWER ---
                    elif part_type == "text":
                        text_val = part.get("text", "")
                        if text_val:
                            yield ("text", text_val)

            # Fallback for standard strings
            elif isinstance(content, str) and content:
                yield ("text", content)