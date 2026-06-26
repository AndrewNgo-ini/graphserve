import uuid
import pytest
from graphserve.persistence import (
    InMemoryConversationStore, Conversation, ConversationNotFoundError,
)

async def test_create_then_get():
    store = InMemoryConversationStore()
    conv = await store.create(model="medical", user="u1", created_at=123)
    assert isinstance(conv, Conversation)
    assert conv.model == "medical" and conv.user == "u1" and conv.created_at == 123
    assert await store.get(conv.id) == conv

async def test_get_missing_raises():
    store = InMemoryConversationStore()
    with pytest.raises(ConversationNotFoundError):
        await store.get(uuid.uuid4())

async def test_delete():
    store = InMemoryConversationStore()
    conv = await store.create(model="m", user=None, created_at=1)
    await store.delete(conv.id)
    with pytest.raises(ConversationNotFoundError):
        await store.get(conv.id)

async def test_resolve_previous_by_response_id():
    store = InMemoryConversationStore()
    conv = await store.create(model="m", user=None, created_at=1)
    from graphserve._ids import format_conv_id
    assert (await store.resolve_previous(format_conv_id(conv.id))) == conv
    assert (await store.resolve_previous("conv_" + "0"*32)) is None
