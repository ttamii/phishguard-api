"""
Скрипт обучения ML моделей для обнаружения фишинга
С поддержкой казахстанского датасета

Автор: Tamiris
Дата: 2025
"""

import os
import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    roc_auc_score,
)
import xgboost as xgb

# Добавляем путь к app
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.ml.features import extract_all_features, get_feature_vector, get_feature_names
from app.ml.kz_dataset import get_kz_dataset, get_kz_dataset_stats

warnings.filterwarnings('ignore')

# Путь для сохранения моделей
MODEL_DIR = Path(__file__).parent.parent / 'app' / 'ml' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_main_dataset() -> pd.DataFrame:
    """Загружает основной датасет GregaVrbancic/Phishing-Dataset"""
    import urllib.request
    import io
    
    print("=" * 60)
    print("ЗАГРУЗКА ОСНОВНОГО ДАТАСЕТА")
    print("=" * 60)
    
    DATASET_URL = "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_small.csv"
    
    print(f"Источник: GregaVrbancic/Phishing-Dataset")
    print("Загрузка...")
    
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
            data = response.read().decode('utf-8')
        
        df = pd.read_csv(io.StringIO(data))
        
        # Переименовываем целевую переменную
        if 'phishing' in df.columns:
            df = df.rename(columns={'phishing': 'label'})
        
        # Приводим все к числовым
        for col in df.columns:
            if col != 'label':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        print(f"Загружено: {len(df)} образцов")
        print(f"Признаков: {len(df.columns) - 1}")
        print(f"Легитимных: {(df['label'] == 0).sum()}")
        print(f"Фишинговых: {(df['label'] == 1).sum()}")
        
        return df
        
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None


def create_kz_dataset() -> pd.DataFrame:
    """Создаёт датасет из казахстанских URL"""
    print("\n" + "=" * 60)
    print("СОЗДАНИЕ КАЗАХСТАНСКОГО ДАТАСЕТА")
    print("=" * 60)
    
    stats = get_kz_dataset_stats()
    print(f"Всего KZ URL: {stats['total']}")
    print(f"Легитимных: {stats['legitimate']}")
    print(f"Фишинговых: {stats['phishing']}")
    
    kz_data = get_kz_dataset()
    
    print("\nИзвлечение признаков из KZ URL...")
    
    rows = []
    feature_names = get_feature_names()
    
    for i, (url, label) in enumerate(kz_data):
        if i % 100 == 0:
            print(f"  Обработано: {i}/{len(kz_data)}")
        
        try:
            features = extract_all_features(url)
            vector = get_feature_vector(features)
            
            row = {name: value for name, value in zip(feature_names, vector)}
            row['label'] = label
            rows.append(row)
        except Exception as e:
            print(f"  Ошибка для {url}: {e}")
            continue
    
    df = pd.DataFrame(rows)
    print(f"\nСоздано KZ образцов: {len(df)}")
    
    return df


def train_models(df: pd.DataFrame):
    """Обучает модели на датасете"""
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Разделяем признаки и метки
    X = df.drop('label', axis=1)
    y = df['label']
    
    print(f"Размер датасета: {len(df)}")
    print(f"Признаков: {X.shape[1]}")
    print(f"Классов: {y.nunique()}")
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Сохраняем scaler
    with open(MODEL_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler сохранён")
    
    # Модели
    models = {
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),
        'xgboost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name.upper()} ---")
        
        # Обучаем
        model.fit(X_train_scaled, y_train)
        
        # Предсказываем
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
        }
        
        # Сохраняем модель
        model_path = MODEL_DIR / f'{name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Модель сохранена: {model_path}")
    
    return results


def main():
    """Главная функция обучения"""
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ ДЛЯ ОБНАРУЖЕНИЯ ФИШИНГА")
    print("PhishGuard - Дипломная работа")
    print("=" * 60)
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Загружаем основной датасет
    main_df = load_main_dataset()
    
    if main_df is None:
        print("Не удалось загрузить основной датасет!")
        return
    
    # Создаём KZ датасет
    kz_df = create_kz_dataset()
    
    # Объединяем датасеты
    print("\n" + "=" * 60)
    print("ОБЪЕДИНЕНИЕ ДАТАСЕТОВ")
    print("=" * 60)
    
    # Убедимся что колонки совпадают
    main_cols = set(main_df.columns)
    kz_cols = set(kz_df.columns)
    
    common_cols = main_cols.intersection(kz_cols)
    print(f"Общих колонок: {len(common_cols)}")
    
    # Используем только общие колонки
    main_df = main_df[list(common_cols)]
    kz_df = kz_df[list(common_cols)]
    
    # Объединяем
    combined_df = pd.concat([main_df, kz_df], ignore_index=True)
    combined_df = combined_df.dropna()
    
    print(f"\nФинальный датасет:")
    print(f"  Всего: {len(combined_df)}")
    print(f"  Легитимных: {(combined_df['label'] == 0).sum()}")
    print(f"  Фишинговых: {(combined_df['label'] == 1).sum()}")
    print(f"  Из них KZ: {len(kz_df)}")
    
    # Обучаем модели
    results = train_models(combined_df)
    
    # Итоговый отчёт
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy:  {metrics['accuracy']*100:.1f}%")
        print(f"  F1-Score:  {metrics['f1']*100:.1f}%")
        print(f"  ROC-AUC:   {metrics['roc_auc']*100:.1f}%")
    
    # Лучшая модель
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\nЛучшая модель: {best_model[0]} (F1: {best_model[1]['f1']*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"Модели сохранены в: {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
