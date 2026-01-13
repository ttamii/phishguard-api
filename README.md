# PhishGuard API

> API для обнаружения фишинговых атак с использованием машинного обучения

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange.svg)](https://xgboost.readthedocs.io/)

## О проекте

PhishGuard — это система обнаружения фишинговых URL с использованием градиентного бустинга (XGBoost). Проект включает:

- **Backend API** на FastAPI
- **Мобильное приложение** на React Native (Expo)
- **ML модель** с точностью 94.5%
- **AI чат-бот** на базе Google Gemini

## Быстрый старт

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/ttamii/phishguard-api.git
cd phishguard-api

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### Запуск

```bash
# Локальный запуск
uvicorn app.main:app --reload --port 8000

# Или с помощью Python
python -m uvicorn app.main:app --reload
```

API будет доступен по адресу: http://localhost:8000

## Документация API

После запуска доступна интерактивная документация:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Основные эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/v1/scan/url` | Проверка URL на фишинг |
| GET | `/api/v1/models` | Информация о моделях |
| GET | `/health` | Проверка здоровья сервиса |

### Пример запроса

```bash
curl -X POST http://localhost:8000/api/v1/scan/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://kaspi.kz"}'
```

### Пример ответа

```json
{
  "scanId": "uuid",
  "url": "https://kaspi.kz",
  "isPhishing": false,
  "probability": 0.05,
  "confidence": 0.99,
  "classification": "safe",
  "modelUsed": "xgboost",
  "explanation": {
    "interpretationTextRu": "Домен в белом списке"
  }
}
```

## ML модель

### Характеристики

| Метрика | Значение |
|---------|----------|
| Модель | XGBoost |
| Accuracy | 94.5% |
| Precision | 94.4% |
| Recall | 95.1% |
| F1 Score | 94.8% |
| AUC-ROC | 0.98 |

### Датасет

- **Основной**: [GregaVrbancic/Phishing-Dataset](https://github.com/GregaVrbancic/Phishing-Dataset) — 58,645 URL, 111 признаков
- **Дополнительный**: Казахстанские URL — 988 URL (kaspi.kz, halyk.kz, egov.kz и др.)

### Признаки

Модель анализирует 111 признаков URL, включая:
- Наличие HTTPS
- Длина URL и домена
- Количество поддоменов
- Подозрительные ключевые слова
- IP вместо домена
- Typosquatting известных брендов

## Архитектура

```
phishing-detector-api/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── scan.py      # Эндпоинты сканирования
│   │       └── models.py    # Информация о моделях
│   ├── core/
│   │   └── config.py        # Настройки
│   ├── ml/
│   │   ├── predictor.py     # Предсказатель
│   │   ├── features.py      # Извлечение признаков
│   │   └── kz_dataset.py    # Казахстанский датасет
│   └── schemas/
│       └── scan.py          # Pydantic схемы
├── notebooks/
│   └── PhishGuard_Analysis.ipynb  # Jupyter анализ
├── requirements.txt
└── README.md
```

## Деплой

Проект развёрнут на [Render](https://render.com/):

**Production URL**: https://phishguard-api-b6un.onrender.com

### Деплой на Render

1. Подключите GitHub репозиторий
2. Выберите Python окружение
3. Build Command: `pip install -r requirements.txt && python train_on_deploy.py`
4. Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## Мобильное приложение

Мобильное приложение находится в отдельном репозитории:
- React Native + Expo
- Поддержка iOS и Android
- QR-сканер для ссылок
- AI чат-бот (Gemini)

## Автор

**Тамирис** — Дипломная работа 2025

---

## Лицензия

MIT License
