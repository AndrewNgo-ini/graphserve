from graphserve.errors import openai_error_body

def test_error_body_shape():
    body = openai_error_body("nope", type="invalid_request_error", code="model_not_found")
    assert body == {"error": {"message": "nope", "type": "invalid_request_error", "code": "model_not_found"}}

def test_error_body_optional_code():
    body = openai_error_body("boom", type="server_error")
    assert body["error"]["code"] is None
