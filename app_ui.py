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

    # [File handling logic is same as before...]
    if message.elements:
        file_el = message.elements[0]
        dest = TEMP_DIR / file_el.name
        if file_el.path:
            import shutil
            shutil.copy(file_el.path, dest)
        current_file = str(dest)
        cl.user_session.set("active_file", current_file)
        await cl.Message(f"Focused on {file_el.name}").send()

    # UI State
    think_step = None
    final_msg = cl.Message(content="")

    # Stream
    async for msg_type, content in agent.astream_run(message.content, current_file):
        if msg_type == "think":
            if not think_step:
                think_step = cl.Step(name="Thinking Process", type="llm")
                await think_step.send()  # ✅ no args here
            await think_step.stream_token(content)

        elif msg_type == "text":
            # Close thinking if open
            if think_step:
                await think_step.update()
                think_step = None
            
            # Stream tokens to main chat
            if not final_msg.id:
                await final_msg.send()
            await final_msg.stream_token(content)

    # Cleanup
    if think_step: await think_step.update()
    if final_msg: await final_msg.update()