# PhishGuard - Полный контекст проекта для ИИ-ассистента

## Общая информация

**Тема дипломной работы:** Разработка мобильного приложения для обнаружения фишинговых атак с использованием методов машинного обучения

**Студент:** Тамирис  
**Университет:** Astana International University (AIU)  
**Год:** 2025

---

## Репозитории на GitHub

### 1. Backend API
- **URL:** https://github.com/ttamii/phishguard-api
- **Деплой:** https://phishguard-api-b6un.onrender.com
- **Технологии:** Python 3.11, FastAPI, XGBoost, Scikit-learn

### 2. Мобильное приложение  
- **URL:** https://github.com/ttamii/phishguard-mobile
- **Технологии:** React Native, Expo, TypeScript

---

## Как склонировать и запустить

### Backend API:
```bash
git clone https://github.com/ttamii/phishguard-api.git
cd phishguard-api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Мобильное приложение:
```bash
git clone https://github.com/ttamii/phishguard-mobile.git
cd phishguard-mobile
npm install
npx expo start
```

---

## Архитектура системы

```
[Мобильное приложение (React Native)]
            ↓ HTTPS/JSON
[REST API (FastAPI)] → https://phishguard-api-b6un.onrender.com
            ↓
[ML модель (XGBoost)] → Извлечение 111 признаков → Классификация
            ↓
[Результат + Объяснение XAI]
```

---

## Детали Backend API

### Структура файлов:
```
phishguard-api/
├── app/
│   ├── api/routes/
│   │   ├── scan.py          # POST /api/v1/scan/url - проверка URL
│   │   └── models.py        # GET /api/v1/models - инфо о моделях
│   ├── core/
│   │   └── config.py        # Настройки (пороги, пути)
│   ├── ml/
│   │   ├── predictor.py     # Класс PhishingPredictor - основная логика
│   │   ├── features.py      # Извлечение 111 признаков из URL
│   │   ├── kz_dataset.py    # Казахстанский датасет (988 URL)
│   │   └── models/          # Папка для .pkl моделей
│   ├── schemas/
│   │   └── scan.py          # Pydantic схемы запросов/ответов
│   └── main.py              # FastAPI приложение, CORS, роуты
├── notebooks/
│   └── PhishGuard_Analysis.ipynb  # Jupyter с EDA и обучением
├── docs/
│   └── presentation_formal.html   # Презентация для защиты
├── train_on_deploy.py       # Скрипт обучения модели при деплое
├── requirements.txt
├── render.yaml              # Конфигурация для Render.com
└── README.md
```

### Ключевой класс PhishingPredictor (app/ml/predictor.py):
- Загружает модель XGBoost и StandardScaler
- Реализует whitelist казахстанских доменов (kaspi.kz, halyk.kz, egov.kz и др.)
- Обнаруживает typosquatting через расстояние Левенштейна
- Генерирует текстовые объяснения решений

### API эндпоинты:
- `POST /api/v1/scan/url` - проверка URL, принимает `{"url": "https://example.com"}`
- `GET /api/v1/models` - информация о модели
- `GET /health` - проверка здоровья
- `GET /docs` - Swagger UI документация

### ML модель:
- **Алгоритм:** XGBoost (Gradient Boosting)
- **Датасет:** 58,645 URL (GregaVrbancic/Phishing-Dataset) + 988 казахстанских URL
- **Признаков:** 111 (структурные, лексические, доменные)
- **Метрики:**
  - Accuracy: 94.5%
  - Precision: 94.4%
  - Recall: 95.1%
  - F1 Score: 94.8%
  - AUC-ROC: 0.98

### Обучение модели:
- `train_on_deploy.py` - автоматически обучает модель при деплое на Render
- `notebooks/train_models.py` - локальное обучение
- Модели сохраняются в `app/ml/models/` как .pkl файлы

---

## Детали мобильного приложения

### Структура файлов:
```
phishguard-mobile/
├── app/
│   ├── (tabs)/
│   │   ├── _layout.tsx      # Layout с TabBar навигацией
│   │   ├── index.tsx        # Главный экран (ввод URL)
│   │   ├── scanner.tsx      # Экран с полем ввода
│   │   ├── qrscanner.tsx    # QR-сканер камеры
│   │   ├── history.tsx      # История проверок
│   │   └── about.tsx        # О приложении
│   └── result/
│       └── [id].tsx         # Экран результата проверки
├── components/
│   ├── chat/
│   │   └── FloatingChatButton.tsx  # AI чат-бот (Gemini)
│   ├── results/
│   │   └── ThreatGauge.tsx  # Круговой индикатор риска
│   └── ui/
│       ├── Button.tsx
│       ├── Card.tsx
│       └── Input.tsx
├── services/
│   ├── api.ts               # Вызовы к backend API
│   ├── storage.ts           # AsyncStorage для истории
│   └── types.ts             # TypeScript типы
├── constants/
│   ├── Config.ts            # API URL и настройки
│   ├── Colors.ts            # Цветовая схема
│   └── Typography.ts        # Шрифты
├── store/
│   └── scanStore.ts         # Zustand store для состояния
├── app.json                 # Expo конфигурация
└── package.json
```

### Функционал:
1. **Ручной ввод URL** - проверка через текстовое поле
2. **QR-сканер** - сканирование QR-кодов с URL
3. **История** - сохранение проверок в AsyncStorage
4. **AI чат-бот** - интеграция с Google Gemini API
5. **Результат** - отображение риска, объяснения, рекомендаций

### API URL в приложении:
Файл `constants/Config.ts`:
```typescript
export const API_URL = 'https://phishguard-api-b6un.onrender.com';
```

### Gemini API:
Файл `components/chat/FloatingChatButton.tsx` содержит GEMINI_API_KEY

---

## Документы дипломной работы

### Готовые файлы в папке docs/:
1. **ДИПЛОМНАЯ_РАБОТА.md** - полный текст дипломной работы
2. **diploma_thesis.md** - Введение + Глава 1
3. **diploma_chapter2.md** - Глава 2 (ML модель)
4. **diploma_chapter3.md** - Глава 3 + Заключение + Литература
5. **presentation_formal.html** - HTML презентация (12 слайдов)

### Требования к оформлению (AIU):
- Объём: 40-60 страниц
- Шрифт: Times New Roman, 14 кегль
- Интервал: 1.0 (одинарный)
- Поля: левое 30мм, верхнее 20мм, правое 10мм, нижнее 25мм
- Отступ: 1.25 см
- Источников: 30-50

---

## Известные проблемы и решения

### 1. Модель не загружается на Render
**Причина:** Несовместимость версий sklearn между локальной машиной и сервером
**Решение:** Используется `train_on_deploy.py` - модель обучается прямо на сервере при деплое

### 2. Gemini API ошибки (429/404)
**Причина:** Лимит бесплатного API или неверное имя модели
**Решение:** Использовать модель `gemini-2.0-flash`, проверить API key

### 3. API возвращает 500 ошибку
**Причина:** Несоответствие признаков между моделью и features.py
**Решение:** Убедиться что features.py извлекает ровно 111 признаков

---

## Что ещё может понадобиться

### Для защиты:
- [ ] Конвертировать дипломку из .md в .docx (Word)
- [ ] Добавить скриншоты приложения в презентацию
- [ ] Подготовить демо на реальном устройстве

### Улучшения (опционально):
- [ ] Добавить SHAP визуализацию в приложение
- [ ] Добавить push-уведомления
- [ ] Браузерное расширение

---

## Быстрые команды

```bash
# Запуск API локально
cd phishguard-api
uvicorn app.main:app --reload

# Запуск мобильного приложения
cd phishguard-mobile
npx expo start

# Тест API
curl -X POST https://phishguard-api-b6un.onrender.com/api/v1/scan/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://kaspi.kz"}'
```

---

Этот документ содержит полный контекст проекта PhishGuard. Используй его для продолжения работы над дипломом.
