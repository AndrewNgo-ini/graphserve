from langchain_core.messages import AIMessageChunk
from graphserve.adapters import chat_completion_chunks


async def _stream():
    yield (AIMessageChunk(content="Hel"), {})
    yield (AIMessageChunk(content="lo"), {})


async def test_chat_chunks_wire_format():
    lines = b"".join([c async for c in chat_completion_chunks(_stream(), completion_id="chatcmpl-1", model="m", created=1)])
    text = lines.decode()
    assert text.startswith("data: ")
    assert '"role":"assistant"' in text.replace(" ", "")
    assert '"content":"Hel"' in text.replace(" ", "")
    assert text.strip().endswith("data: [DONE]")
