import uuid
import pytest
from graphserve._ids import format_conv_id, parse_conv_id

def test_format_then_parse_roundtrips():
    u = uuid.uuid4()
    assert parse_conv_id(format_conv_id(u)) == u

def test_parse_accepts_bare_uuid():
    u = uuid.uuid4()
    assert parse_conv_id(str(u)) == u

def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_conv_id("not-a-uuid")
