from fastapi import FastAPI

from faithful_edge_rag.api.routes import health, research
from faithful_edge_rag.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings()
    app = FastAPI(
        title=app_settings.app_name,
        version="0.1.0",
        description="Open-source faithful edge-cloud RAG research prototype.",
    )
    app.include_router(health.router)
    app.include_router(research.router)
    return app


app = create_app()

