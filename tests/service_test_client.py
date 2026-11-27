import os

from fastapi.testclient import TestClient


class OperatorClient(TestClient):
    """Authenticate fixtures that exercise installation operator workflows."""

    def __init__(self, app, **kwargs):
        headers = {"x-api-key": os.environ["BALOOT_API_KEY"]}
        headers.update(kwargs.pop("headers", None) or {})
        super().__init__(app, headers=headers, **kwargs)
