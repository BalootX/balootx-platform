import asyncio
import os

from fastapi.testclient import TestClient


class OperatorClient(TestClient):
    """Authenticate fixtures that exercise installation operator workflows."""

    def __init__(self, app, **kwargs):
        headers = {"x-api-key": os.environ["BALOOT_API_KEY"]}
        headers.update(kwargs.pop("headers", None) or {})
        super().__init__(app, headers=headers, **kwargs)


def grpc_probe(tmp_path, monkeypatch, probe, *, stub_type):
    import grpc

    from baloot_platform.service.grpc_service import serve

    monkeypatch.setenv("BALOOT_RUNTIME", str(tmp_path))
    monkeypatch.delenv("BALOOT_GRPC_TLS_CERT", raising=False)
    monkeypatch.delenv("BALOOT_GRPC_TLS_KEY", raising=False)
    make_server = grpc.aio.server

    async def run():
        server = None

        def create_server(**options):
            nonlocal server
            server = make_server(**options)
            port = server.add_insecure_port("127.0.0.1:0")

            async def after_start():
                async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                    await probe(grpc, stub_type(channel))

            monkeypatch.setattr(server, "add_insecure_port", lambda _address: port)
            monkeypatch.setattr(server, "wait_for_termination", after_start)
            return server

        monkeypatch.setattr(grpc.aio, "server", create_server)
        try:
            await serve()
        finally:
            if server is not None:
                await server.stop(0)

    asyncio.run(run())
