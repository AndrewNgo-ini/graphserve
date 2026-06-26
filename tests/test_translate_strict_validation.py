"""Tests that messages_to_response_dict produces a dict that passes strict
OpenAI Response model validation (Response.model_validate).
"""

import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from openai.types.responses import Response

from graphserve.translate import messages_to_response_dict


def _validate(messages):
    """Helper: build the dict and run strict validation; must not raise."""
    obj = messages_to_response_dict(
        messages,
        conversation_id=uuid.uuid4(),
        model="test",
        created_at=1,
    )
    Response.model_validate(obj)


def test_single_ai_message_passes_strict_validation():
    """(a) A single AIMessage with plain text content."""
    _validate([AIMessage(content="hi", id="a1")])


def test_ai_message_with_tool_call_passes_strict_validation():
    """(b) AIMessage with a tool call, followed by a ToolMessage."""
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "c1", "type": "tool_call"}],
        id="a2",
    )
    tool_msg = ToolMessage(content="result", tool_call_id="c1")
    _validate([ai_msg, tool_msg])


def test_conversation_passes_strict_validation():
    """(c) A full conversation: HumanMessage -> AIMessage."""
    _validate([HumanMessage("hi"), AIMessage("yo", id="a3")])
