from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import health
from app.api.routes import analysis


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI-assisted multimodal deepfake "
        "video analysis API."
    ),
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================

app.include_router(
    health.router
)

app.include_router(
    analysis.router
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():
    return {
        "name": "DeepScan API",
        "status": "online",
        "version": settings.APP_VERSION,
    }