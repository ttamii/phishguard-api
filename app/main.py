"""
PhishGuard API - FastAPI Backend
Дипломная работа: Обнаружение фишинговых атак с использованием ИИ
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import scan, models
from app.core.config import settings

app = FastAPI(
    title="PhishGuard API",
    description="API для обнаружения фишинговых атак с использованием ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware для мобильного приложения
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене ограничить!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(scan.router, prefix="/api/v1/scan", tags=["Сканирование"])
app.include_router(models.router, prefix="/api/v1/models", tags=["ML модели"])


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "name": "PhishGuard API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "healthy"}
