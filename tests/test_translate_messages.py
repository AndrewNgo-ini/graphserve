import uuid
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from graphserve.adapters import extract_text, lc_messages_to_openai_items, messages_to_response_dict

def test_extract_text_str():
    assert extract_text("hi") == "hi"

def test_extract_text_blocks():
    blocks = [{"type": "output_text", "text": "a"}, {"type": "input_text", "text": "b"}, {"type": "image", "url": "x"}]
    assert extract_text(blocks) == "ab"

def test_ai_message_to_message_item():
    items = lc_messages_to_openai_items([AIMessage(content="hello", id="m1")])
    assert any(i["type"] == "message" and i["role"] == "assistant" for i in items)

def test_ai_message_tool_call_item():
    msg = AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "c1", "type": "tool_call"}], id="m2")
    items = lc_messages_to_openai_items([msg])
    fc = [i for i in items if i["type"] == "function_call"]
    assert fc and fc[0]["name"] == "lookup" and fc[0]["call_id"] == "c1"

def test_tool_message_to_output_item():
    items = lc_messages_to_openai_items([ToolMessage(content="result", tool_call_id="c1")])
    fco = [i for i in items if i["type"] == "function_call_output"]
    assert fco and fco[0]["output"] == "result" and fco[0]["call_id"] == "c1"

def test_messages_to_response_dict_excludes_user_items():
    cid = uuid.uuid4()
    resp = messages_to_response_dict(
        [HumanMessage(content="hi", id="h1"), AIMessage(content="yo", id="a1")],
        conversation_id=cid, model="medical", created_at=42,
    )
    assert resp["model"] == "medical" and resp["created_at"] == 42
    assert all(not (i["type"] == "message" and i["role"] in ("user", "system", "developer")) for i in resp["output"])
    assert any(i["type"] == "message" and i["role"] == "assistant" for i in resp["output"])
