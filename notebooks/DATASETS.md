# Датасеты для обучения моделей обнаружения фишинга

## Рекомендуемые датасеты (для реального обучения)

### 1. PhishTank
- URL: https://phishtank.org/developer_info.php
- Описание: База данных фишинговых URL с верифицированными метками
- Формат: JSON/CSV
- Обновление: Ежедневно
- Использование: Бесплатно с регистрацией

### 2. UCI Phishing Websites Dataset
- URL: https://archive.ics.uci.edu/ml/datasets/phishing+websites
- Описание: 11,055 URL с 30 признаками
- Формат: ARFF
- Признаки: Уже извлечены

### 3. Kaggle Phishing Datasets
- Phishing Website Dataset: https://www.kaggle.com/datasets/akashkr/phishing-website-dataset
- Web Page Phishing Detection: https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset

### 4. OpenPhish
- URL: https://openphish.com/
- Описание: Фиды фишинговых URL
- Обновление: Каждые 12 часов

## Структура датасета

Для обучения необходим CSV файл со следующими колонками:

```
url,label
https://legitimate-site.com,0
http://phishing-site.xyz/login,1
...
```

## Загрузка реального датасета

```python
import pandas as pd

# Загрузка из CSV
df = pd.read_csv('phishing_dataset.csv')

# Или из PhishTank API
import requests
response = requests.get('http://data.phishtank.com/data/API_KEY/online-valid.json')
phishing_urls = response.json()
```

## Баланс классов

Рекомендуется использовать сбалансированный датасет или применять техники:
- Oversampling (SMOTE)
- Undersampling
- Class weights

## Для дипломной работы

1. Скачайте датасет с Kaggle или UCI
2. Замените функцию `generate_synthetic_dataset` на загрузку реальных данных
3. Запустите `train_models.py`
4. Модели сохранятся в `app/ml/models/`
