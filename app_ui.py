import chainlit as cl
from pathlib import Path
from app.modules.agent.matlab_agent import MatlabAgent

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

@cl.on_chat_start
async def start():
    cl.user_session.set("agent", MatlabAgent())
    cl.user_session.set("active_file", None)
    await cl.Message("Matlab Agent Ready. I will reason before answering.").send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    current_file = cl.user_session.get("active_file")

    # Handle file upload if present
    if message.elements:
        file_el = message.elements[0]
        # Ensure path exists before copying
        if file_el.path:
            import shutil
            dest = TEMP_DIR / file_el.name
            shutil.copy(file_el.path, dest)
            current_file = str(dest)
            cl.user_session.set("active_file", current_file)
            await cl.Message(f"Focused on {file_el.name}").send()

    # UI State
    think_step = None
    final_msg = cl.Message(content="")

    # Stream
    async for msg_type, content in agent.astream_run(message.content, current_file):
        
        # 1. Handle Thoughts
        if msg_type == "think":
            if not think_step:
                # Create the step only once when thinking starts
                think_step = cl.Step(name="Thinking Process", type="llm")
                await think_step.send()
            
            # Simply stream the delta. Chainlit handles the markdown rendering.
            await think_step.stream_token(content)

        # 2. Handle Final Answer
        elif msg_type == "text":
            # If we were thinking, close that step now
            if think_step:
                await think_step.update()
                think_step = None
            
            # Start the main message if not started
            if not final_msg.id:
                await final_msg.send()
                
            await final_msg.stream_token(content)

    # Final cleanup
    if think_step:
        await think_step.update()
    
    if final_msg.id:
        await final_msg.update()