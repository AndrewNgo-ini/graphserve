"""Minimal real LangGraph graphs for testing the OpenAI surface — no domain code."""
from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Fake streaming chat model that emits tool_call_chunks
# ---------------------------------------------------------------------------

class FakeToolThenTextModel(BaseChatModel):
    """Deterministic fake model that streams a tool call then text content.

    First invocation yields one tool_call_chunk; subsequent invocations yield
    a plain text chunk.  Both paths flow through on_chat_model_stream events.
    """

    _call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-then-text"

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        # Non-streaming fallback (not used in tests, but required by BaseChatModel)
        msg = AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        # Determine turn: if last message is a human message, it's the first turn
        last = messages[-1] if messages else None
        from langchain_core.messages import HumanMessage
        if isinstance(last, (HumanMessage, type(None))):
            # First turn: emit a tool call chunk
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "lookup",
                            "args": '{"q":"x"}',
                            "id": "call_1",
                            "index": 0,
                        }
                    ],
                )
            )
        else:
            # Subsequent turn: emit plain text
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="done")
            )

    async def _astream(
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._stream(messages, **kwargs):
            yield chunk


class FakePlainTextModel(BaseChatModel):
    """Deterministic fake model that streams a single plain-text assistant message."""

    @property
    def _llm_type(self) -> str:
        return "fake-plain-text"

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        msg = AIMessage(content="hello world")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="hello world"))

    async def _astream(
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._stream(messages, **kwargs):
            yield chunk


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def echo_graph():
    def respond(state: State) -> State:
        last = state["messages"][-1]
        text = last.content if isinstance(last.content, str) else str(last.content)
        return {"messages": [AIMessage(content=f"echo: {text}")]}
    g = StateGraph(State)
    g.add_node("respond", respond)
    g.add_edge(START, "respond")
    g.add_edge("respond", END)
    return g.compile()


def echo_graph_with_checkpointer():
    """Same as echo_graph but compiled with a MemorySaver checkpointer.

    Use this in tests that need GET /responses/{id} to return non-empty output,
    since checkpoint replay requires a checkpointer bound at compile time.
    """
    def respond(state: State) -> State:
        last = state["messages"][-1]
        text = last.content if isinstance(last.content, str) else str(last.content)
        return {"messages": [AIMessage(content=f"echo: {text}")]}
    g = StateGraph(State)
    g.add_node("respond", respond)
    g.add_edge(START, "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=MemorySaver())


def tool_then_text_graph():
    """Graph that calls a fake streaming model producing tool_call_chunks.

    The fake model routes on the last message type:
    - HumanMessage -> emits a tool call chunk (on_chat_model_stream with tool_call_chunks)
    - ToolMessage  -> emits plain text "done"

    This exercises the real on_chat_model_stream path in emit_response_sse.
    """
    model = FakeToolThenTextModel()

    def call_llm(state: State) -> State:
        # Invoke synchronously to get the AIMessage with tool_calls populated
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    g = StateGraph(State)
    g.add_node("call_llm", call_llm)
    g.add_edge(START, "call_llm")
    g.add_edge("call_llm", END)
    return g.compile()


def plain_text_llm_graph():
    """Graph whose node calls a fake streaming LLM returning a plain text message.

    Used for double-emission regression tests.
    """
    model = FakePlainTextModel()

    def call_llm(state: State) -> State:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    g = StateGraph(State)
    g.add_node("call_llm", call_llm)
    g.add_edge(START, "call_llm")
    g.add_edge("call_llm", END)
    return g.compile()


def tool_call_graph():
    """Graph whose node returns an AIMessage carrying a tool call (non-streaming).

    Exercises the Chat Completions non-streaming tool-call path: the assistant
    message must surface ``tool_calls`` with finish_reason ``tool_calls``.
    """
    def respond(state: State) -> State:
        return {"messages": [AIMessage(
            content="",
            tool_calls=[{
                "name": "get_weather",
                "args": {"city": "Hanoi"},
                "id": "call_abc",
                "type": "tool_call",
            }],
        )]}
    g = StateGraph(State)
    g.add_node("respond", respond)
    g.add_edge(START, "respond")
    g.add_edge("respond", END)
    return g.compile()


def streaming_tool_call_graph():
    """Graph whose node *streams* a tool-calling model so tool_call_chunks flow
    through stream_mode='messages' — exercises Chat Completions streaming tool calls."""
    model = FakeToolThenTextModel()

    async def call_llm(state: State) -> State:
        final: AIMessageChunk | None = None
        async for chunk in model.astream(state["messages"]):
            final = chunk if final is None else final + chunk
        return {"messages": [final]}

    g = StateGraph(State)
    g.add_node("call_llm", call_llm)
    g.add_edge(START, "call_llm")
    g.add_edge("call_llm", END)
    return g.compile()


def recording_graph():
    """Graph that records the messages it received, then echoes the last one.

    Returns ``(graph, received)``. After any invoke, ``received["messages"]`` is
    the full list of LangChain messages the node saw — used to assert that the
    server forwards the ENTIRE OpenAI ``messages`` array each turn (Open WebUI is
    stateless toward the backend) with roles mapped to the right message types.
    """
    received: dict = {}

    def respond(state: State) -> State:
        received["messages"] = list(state["messages"])
        last = state["messages"][-1]
        text = last.content if isinstance(last.content, str) else str(last.content)
        return {"messages": [AIMessage(content=f"echo: {text}")]}

    g = StateGraph(State)
    g.add_node("respond", respond)
    g.add_edge(START, "respond")
    g.add_edge("respond", END)
    return g.compile(), received


def streaming_text_graph():
    """Graph whose node *streams* the model so stream_mode='messages' surfaces tokens.

    Unlike plain_text_llm_graph (which uses model.invoke and yields the message
    whole), this node calls model.astream, so token chunks flow through
    LangGraph's "messages" stream mode — required to exercise real token-level
    streaming on the Chat Completions path.
    """
    model = FakePlainTextModel()

    async def call_llm(state: State) -> State:
        final: AIMessageChunk | None = None
        async for chunk in model.astream(state["messages"]):
            final = chunk if final is None else final + chunk
        return {"messages": [final]}

    g = StateGraph(State)
    g.add_node("call_llm", call_llm)
    g.add_edge(START, "call_llm")
    g.add_edge("call_llm", END)
    return g.compile()
