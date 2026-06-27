from types import SimpleNamespace

from graphserve.translate import request_to_context


def test_context_carries_user_and_metadata():
    req = SimpleNamespace(
        user="user-42",
        metadata={"identity_number": "VN001", "turn_number": 2},
        instructions=None,
    )
    ctx = request_to_context(req)
    assert ctx["user_id"] == "user-42"
    assert ctx["metadata"]["identity_number"] == "VN001"
    assert ctx["metadata"]["turn_number"] == 2


def test_instructions_become_custom_instructions():
    req = SimpleNamespace(user=None, metadata={}, instructions="Be concise.")
    ctx = request_to_context(req)
    assert ctx["metadata"]["custom_instructions"] == "Be concise."
    assert "user_id" not in ctx


def test_empty_request_returns_none():
    req = SimpleNamespace(user=None, metadata=None, instructions=None)
    assert request_to_context(req) is None


def test_chat_request_without_instructions_attr():
    # ChatCompletionRequest has no `instructions` attribute at all.
    req = SimpleNamespace(user="u1", metadata={"conversation_id": "c1"})
    ctx = request_to_context(req)
    assert ctx["user_id"] == "u1"
    assert ctx["metadata"]["conversation_id"] == "c1"
