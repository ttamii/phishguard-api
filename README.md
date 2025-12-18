# PhishGuard API

API для обнаружения фишинговых атак с использованием машинного обучения.

## Установка

```bash
# Создание виртуального окружения
python -m venv venv

# Активация (Windows)
venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

## Запуск

```bash
# Разработка
uvicorn app.main:app --reload --port 8000

# Продакшен
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Документация

После запуска доступна по адресам:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Эндпоинты

### POST /api/v1/scan/url
Сканирование URL на фишинг

### POST /api/v1/scan/message
Сканирование текстового сообщения

### GET /api/v1/models
Информация о ML моделях

## ML Модели

- **Logistic Regression** - быстрая, интерпретируемая
- **Random Forest** - хорошая точность
- **XGBoost** - лучшая точность (рекомендован)
