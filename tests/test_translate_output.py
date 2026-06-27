from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graphserve.translate import result_to_text


def test_last_ai_message_text():
    result = {"messages": [HumanMessage(content="hi"), AIMessage(content="hello")]}
    assert result_to_text(result) == "hello"


def test_return_direct_tool_message_is_surfaced():
    # return_direct tool: last message is the ToolMessage carrying the answer.
    result = {"messages": [
        HumanMessage(content="summarize"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "summarize_medical_history", "args": {}, "type": "tool_call"}]),
        ToolMessage(content='{"khoa": "Khoa Noi"}', tool_call_id="c1", name="summarize_medical_history"),
    ]}
    assert result_to_text(result) == '{"khoa": "Khoa Noi"}'


def test_empty_messages():
    assert result_to_text({"messages": []}) == ""
    assert result_to_text({}) == ""
