import uuid
import pytest
from graphserve._ids import (
    format_conv_id,
    parse_conv_id,
    format_resp_id,
    parse_resp_id,
    thread_uuid_from_anchor,
)

def test_format_then_parse_roundtrips():
    u = uuid.uuid4()
    assert parse_conv_id(format_conv_id(u)) == u

def test_parse_accepts_bare_uuid():
    u = uuid.uuid4()
    assert parse_conv_id(str(u)) == u

def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_conv_id("not-a-uuid")

def test_resp_id_roundtrips_model_with_dots():
    u = uuid.uuid4()
    model = "gpt-4.1-mini"
    rid = format_resp_id(model, u)
    assert rid.startswith("resp_")
    assert parse_resp_id(rid) == (model, u)

def test_parse_resp_id_rejects_garbage():
    with pytest.raises(ValueError):
        parse_resp_id("resp_not-a-real-id")

def test_thread_uuid_from_anchor_handles_all_shapes():
    u = uuid.uuid4()
    assert thread_uuid_from_anchor(format_conv_id(u)) == u
    assert thread_uuid_from_anchor(format_resp_id("m", u)) == u
    assert thread_uuid_from_anchor(str(u)) == u
