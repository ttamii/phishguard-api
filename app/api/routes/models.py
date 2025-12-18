"""
API роуты для информации о моделях
"""

from fastapi import APIRouter

from app.ml.predictor import predictor

router = APIRouter()


@router.get("")
async def get_models():
    """
    Получение информации о доступных ML моделях
    """
    models = predictor.get_model_info()
    return {
        "models": models,
        "defaultModel": "xgboost",
    }


@router.get("/{model_id}")
async def get_model(model_id: str):
    """
    Получение информации о конкретной модели
    """
    models = predictor.get_model_info()
    for model in models:
        if model["id"] == model_id:
            return model
    
    return {"error": "Модель не найдена"}
