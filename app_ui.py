import chainlit as cl
from pathlib import Path
from langchain_community.chat_message_histories import ChatMessageHistory
import shutil

from app.modules.agent.matlab_agent import MatlabAgent # Assuming same directory for this example

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

@cl.on_chat_start
async def start():
    cl.user_session.set("agent", MatlabAgent())
    cl.user_session.set("history", ChatMessageHistory()) 
    cl.user_session.set("active_file", None)
    
    await cl.Message("Matlab Agent Ready. I will reason before answering.").send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    history = cl.user_session.get("history")
    current_file = cl.user_session.get("active_file")

    # Handle file upload
    if message.elements:
        file_el = message.elements[0]
        if file_el.path:
            dest = TEMP_DIR / file_el.name
            shutil.copy(file_el.path, dest)
            current_file = str(dest)
            cl.user_session.set("active_file", current_file)
            await cl.Message(f"Focused on {file_el.name}").send()

    final_msg = cl.Message(content="")
    await final_msg.send()

    think_step = None
    full_response_text = ""

    # Call Agent with CURRENT history (excluding the new message to avoid duplication)
    # The agent will combine: System + History + [Context + Current Query]
    async for msg_type, content in agent.astream_run(message.content, history.messages, current_file):
        
        if msg_type == "think":
            if not think_step:
                think_step = cl.Step(
                    name="Thinking Process",
                    type="llm",
                    parent_id=final_msg.id
                )
                await think_step.send()
            
            await think_step.stream_token(content)

        elif msg_type == "text":
            if think_step:
                await think_step.update()
                think_step = None
            
            full_response_text += content
            await final_msg.stream_token(content)

    if think_step:
        await think_step.update()
    
    await final_msg.update()

    # Update History AFTER the generation
    # 1. Add the clean user query (without the messy file context string)
    history.add_user_message(message.content)
    # 2. Add the AI response
    history.add_ai_message(full_response_text)