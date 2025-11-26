from __future__ import annotations

import os
from typing import AsyncIterator, List

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

load_dotenv()

class MatlabAgent:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=os.getenv("CONTRACT_AGENT_MODEL", "gpt-5.1"),
            use_responses_api=True, 
            reasoning={"effort": "high", "summary": "auto"},
        )

        # Removed self.prompt - we will construct messages dynamically in astream_run

    async def astream_run(
        self, 
        query: str, 
        chat_history: List[BaseMessage], 
        file_path: str | None = None
    ) -> AsyncIterator[tuple[str, str]]:
        
        # 1. Prepare Contextualized Input
        context = (
            f"File Context: {os.path.abspath(file_path)}"
            if file_path
            else "No file context."
        )
        full_input = f"{context}\n\nUser Query: {query}"

        # 2. Construct Message List: System -> History -> Current User Input
        messages = [SystemMessage(content="You are a Senior MATLAB Expert.")]
        
        # Add previous conversation history
        messages.extend(chat_history)
        
        # Add the current query with context
        messages.append(HumanMessage(content=full_input))

        config = {
            "callbacks": [CallbackHandler()],
            "metadata": {"thread_id": "1"},
        }
        
        # Track the index of the thought to know when to insert newlines
        current_thought_idx = 0

        # Pass the constructed list of messages directly to the LLM
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
                                    item_idx = item.get("index", 0)
                                    
                                    if item_idx > current_thought_idx:
                                        yield ("think", "\n\n")
                                        current_thought_idx = item_idx
                                    
                                    yield ("think", item["text"])

                    # --- 2. HANDLE FINAL ANSWER ---
                    elif part_type == "text":
                        text_val = part.get("text", "")
                        if text_val:
                            yield ("text", text_val)

            elif isinstance(content, str) and content:
                yield ("text", content)