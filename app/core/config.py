"""
Конфигурация приложения
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Настройки приложения"""
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # ML модели
    MODEL_PATH: str = "app/ml/models"
    DEFAULT_MODEL: str = "xgboost"
    
    # Пороги классификации
    THRESHOLD_SAFE: float = 0.3
    THRESHOLD_SUSPICIOUS: float = 0.7
    
    class Config:
        env_file = ".env"


settings = Settings()
