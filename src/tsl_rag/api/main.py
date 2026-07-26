import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "tsl_rag.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,  # 1 worker — BM25 index in-memory per process
    )
