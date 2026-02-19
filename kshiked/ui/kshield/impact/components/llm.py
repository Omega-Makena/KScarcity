"""
Async LLM Adaptor for Policy Chat.
Handles async execution loop and calls to the core PolicyChatbot.
"""
from __future__ import annotations
import asyncio
from kshiked.pulse.llm.policy_chatbot import PolicyChatbot

def run_async(coro):
    """Run an async coroutine from synchronous Streamlit code.

    Creates a fresh event loop each time so that aiohttp sessions
    created inside the coroutine live and die within the same loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

async def async_process_bill(session, text="", title="", pdf_bytes=None, url=""):
    """Run the full bill analysis pipeline inside a fresh async context."""
    async with PolicyChatbot() as chatbot:
        return await chatbot.process_bill(
            session, text=text, title=title, pdf_bytes=pdf_bytes, url=url,
        )

async def async_ask(session, question: str):
    """Handle a follow-up question inside a fresh async context."""
    async with PolicyChatbot() as chatbot:
        return await chatbot.ask(session, question)
