from fastapi import FastAPI


def security_probe_app() -> FastAPI:
    from baloot_platform.service import api

    app = FastAPI()
    app.middleware("http")(api.platform_security)
    app.add_middleware(api.RequestLoggingMiddleware)
    return app
