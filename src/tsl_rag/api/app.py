from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from tsl_rag.api.routers.query import router as query_router
from tsl_rag.core.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="TSL-RAG API",
        description="EU Transport & Logistics compliance RAG system",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(query_router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": "tsl-rag", "status": "running"}

    logger.info(f"App created | env={settings.app_env} | llm={settings.active_llm_model}")
    return app


app = create_app()
