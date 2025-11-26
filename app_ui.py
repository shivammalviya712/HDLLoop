import chainlit as cl
from pathlib import Path
from langchain_community.chat_message_histories import ChatMessageHistory
import shutil

from hdlagent import MatlabAgent

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


@cl.on_chat_start
async def start():
    cl.user_session.set("agent", MatlabAgent())
    cl.user_session.set("history", ChatMessageHistory())
    # List of absolute file paths for HDL files
    cl.user_session.set("hdl_files", [])


@cl.on_message
async def main(message: cl.Message):
    agent: MatlabAgent = cl.user_session.get("agent")
    history: ChatMessageHistory = cl.user_session.get("history")
    hdl_files: list[str] = cl.user_session.get("hdl_files") or []

    # --- 1. HANDLE HDL FILE UPLOAD(S) ---
    newly_added: list[Path] = []
    if message.elements:
        for el in message.elements:
            if not getattr(el, "path", None):
                continue
            dest = TEMP_DIR / el.name
            shutil.copy(el.path, dest)
            hdl_files.append(str(dest))
            newly_added.append(dest)

        cl.user_session.set("hdl_files", hdl_files)

        if newly_added:
            names = ", ".join(p.name for p in newly_added)
            total = ", ".join(Path(p).name for p in hdl_files)
            await cl.Message(
                f"✅ Added HDL file(s): **{names}**\n"
                f"📂 Current design files: {total}"
            ).send()

    # --- 2. REQUIRE AT LEAST ONE HDL FILE ---
    if not hdl_files:
        await cl.Message(
            "⚠️ Please upload your HDL project files first "
            "(`.v`, `.sv`, `.vhd`, packages, submodules etc.).\n"
            "You can describe what to optimize in the same message as the upload."
        ).send()
        return

    # --- 3. STREAMED RESPONSE (NO EMPTY PLACEHOLDER BUBBLE) ---
    final_msg: cl.Message | None = None
    think_step: cl.Step | None = None
    full_response_text = ""

    async for msg_type, content in agent.astream_run(
        query=message.content,
        chat_history=history.messages,
        file_paths=hdl_files,
    ):
        if msg_type == "think":
            # Create a top-level thinking step (no parent_id), only once
            if not think_step:
                think_step = cl.Step(
                    name="Thinking Process",
                    type="llm",
                )
                await think_step.send()

            await think_step.stream_token(content)

        elif msg_type == "text":
            # First visible text chunk → create and send the answer message
            if final_msg is None:
                final_msg = cl.Message(content=content)
                await final_msg.send()
                full_response_text += content
            else:
                full_response_text += content
                await final_msg.stream_token(content)

    # Finalize thinking + answer blocks
    if think_step:
        await think_step.update()

    if final_msg:
        await final_msg.update()

    # --- 4. UPDATE HISTORY ---
    history.add_user_message(message.content)
    history.add_ai_message(full_response_text)
