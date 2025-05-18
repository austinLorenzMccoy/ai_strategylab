from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from app.core.config import get_settings
from app.api.routes import router as api_router
from app.api.websocket_routes import router as websocket_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI Strategy Lab for OKX crypto trading",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Include WebSocket routes
app.include_router(websocket_router)


@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"status": "healthy", "message": "AI Strategy Lab API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )