"""
Скрипт обучения ML моделей для обнаружения фишинга
Дипломная работа: Разработка методов обнаружения фишинговых атак с использованием ИИ

Автор: Tamiris
Дата: 2025

Этот скрипт:
1. Загружает и подготавливает датасет
2. Извлекает признаки из URL
3. Обучает три модели: Logistic Regression, Random Forest, XGBoost
4. Оценивает и сравнивает модели
5. Сохраняет обученные модели
"""

import os
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import xgboost as xgb

# Подавляем предупреждения для чистого вывода
warnings.filterwarnings('ignore')

# Путь для сохранения моделей
MODEL_DIR = Path(__file__).parent.parent / 'app' / 'ml' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_real_dataset() -> pd.DataFrame:
    """
    Загружает реальный датасет фишинговых URL
    
    Источник: GregaVrbancic/Phishing-Dataset (GitHub)
    Содержит 88,647 URL с 111 признаками
    
    Returns:
        DataFrame с признаками и метками
    """
    import urllib.request
    import io
    
    print("=" * 60)
    print("ЗАГРУЗКА РЕАЛЬНОГО ДАТАСЕТА")
    print("=" * 60)
    
    # URL датасета на GitHub (dataset_small.csv - 58,645 образцов)
    DATASET_URL = "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_small.csv"
    
    print(f"\nИсточник: GregaVrbancic/Phishing-Dataset")
    print(f"URL: {DATASET_URL}")
    print("\nЗагрузка датасета...")
    
    try:
        # Загружаем датасет
        with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
            data = response.read().decode('utf-8')
        
        # Читаем в DataFrame
        df = pd.read_csv(io.StringIO(data))
        
        print(f"Датасет загружен успешно!")
        print(f"\nРазмер датасета: {len(df)} образцов")
        print(f"Количество признаков: {len(df.columns) - 1}")
        
        # Проверяем структуру
        if 'phishing' in df.columns:
            # Переименовываем целевую переменную
            df = df.rename(columns={'phishing': 'label'})
        elif 'class' in df.columns:
            df = df.rename(columns={'class': 'label'})
        elif 'result' in df.columns:
            df = df.rename(columns={'result': 'label'})
        
        # Убедимся что label бинарный (0/1)
        if df['label'].dtype == object:
            df['label'] = df['label'].map({'legitimate': 0, 'phishing': 1, 'good': 0, 'bad': 1})
        
        # Приводим все значения к числовым
        for col in df.columns:
            if col != 'label':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Удаляем строки с пропусками
        df = df.dropna()
        
        # Статистика по классам
        legitimate_count = (df['label'] == 0).sum()
        phishing_count = (df['label'] == 1).sum()
        
        print(f"\nРаспределение классов:")
        print(f"   Легитимных: {legitimate_count} ({legitimate_count/len(df)*100:.1f}%)")
        print(f"   Фишинговых: {phishing_count} ({phishing_count/len(df)*100:.1f}%)")
        
        # Используем ВСЕ признаки из датасета (кроме label)
        print(f"\nИспользуем все {len(df.columns) - 1} признаков из датасета")
        
        return df
        
    except urllib.error.URLError as e:
        print(f"\nОшибка загрузки: {e}")
        print("Используем синтетический датасет как fallback...")
        return generate_synthetic_fallback()
    except Exception as e:
        print(f"\nОшибка обработки данных: {e}")
        print("Используем синтетический датасет как fallback...")
        return generate_synthetic_fallback()


def generate_synthetic_fallback(n_samples: int = 10000) -> pd.DataFrame:
    """
    Генерирует синтетический датасет как fallback если загрузка не удалась
    """
    print("\nГенерация синтетического датасета (fallback)...")
    
    np.random.seed(42)
    
    data = {
        'url_length': np.concatenate([
            np.random.normal(45, 15, n_samples // 2).astype(int),
            np.random.normal(85, 25, n_samples // 2).astype(int),
        ]),
        'domain_length': np.concatenate([
            np.random.normal(12, 4, n_samples // 2).astype(int),
            np.random.normal(22, 8, n_samples // 2).astype(int),
        ]),
        'path_length': np.concatenate([
            np.random.normal(15, 10, n_samples // 2).astype(int),
            np.random.normal(35, 15, n_samples // 2).astype(int),
        ]),
        'has_https': np.concatenate([
            np.random.choice([0, 1], n_samples // 2, p=[0.1, 0.9]),
            np.random.choice([0, 1], n_samples // 2, p=[0.6, 0.4]),
        ]),
        'has_ip_address': np.concatenate([
            np.random.choice([0, 1], n_samples // 2, p=[0.99, 0.01]),
            np.random.choice([0, 1], n_samples // 2, p=[0.8, 0.2]),
        ]),
        'subdomain_count': np.concatenate([
            np.random.poisson(0.5, n_samples // 2),
            np.random.poisson(2, n_samples // 2),
        ]),
        'special_char_count': np.concatenate([
            np.random.poisson(3, n_samples // 2),
            np.random.poisson(8, n_samples // 2),
        ]),
        'suspicious_keyword_count': np.concatenate([
            np.random.poisson(0.2, n_samples // 2),
            np.random.poisson(3, n_samples // 2),
        ]),
        'entropy_score': np.concatenate([
            np.random.normal(3.5, 0.5, n_samples // 2),
            np.random.normal(4.5, 0.7, n_samples // 2),
        ]),
    }
    
    labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    df = pd.DataFrame(data)
    df['label'] = labels
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Синтетический датасет создан: {len(df)} образцов")
    
    return df


def train_and_evaluate_models(df: pd.DataFrame):
    """
    Обучает и оценивает ML модели
    """
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ ML МОДЕЛЕЙ")
    print("="*60)
    
    # Разделение на признаки и метки
    X = df.drop('label', axis=1)
    y = df['label']
    
    feature_names = X.columns.tolist()
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📈 Размер обучающей выборки: {len(X_train)}")
    print(f"📊 Размер тестовой выборки: {len(X_test)}")
    
    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Определение моделей
    models = {
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            solver='lbfgs'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'xgboost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        ),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'─'*40}")
        print(f"📦 Обучение: {name}")
        print('─'*40)
        
        # Обучение
        if name == 'logistic_regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Кросс-валидация
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if name == 'logistic_regression':
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        print(f"  CV Score:  {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"  TP: {cm[1,1]:4d}  FP: {cm[0,1]:4d}")
        print(f"  FN: {cm[1,0]:4d}  TN: {cm[0,0]:4d}")
        
        # Feature Importance (для RF и XGBoost)
        if name in ['random_forest', 'xgboost']:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:5]
            print(f"\n  Top 5 важных признаков:")
            for i, idx in enumerate(indices):
                print(f"    {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    return results, scaler


def save_models(results: dict, scaler):
    """
    Сохраняет обученные модели
    """
    print("\n" + "="*60)
    print("💾 СОХРАНЕНИЕ МОДЕЛЕЙ")
    print("="*60)
    
    for name, data in results.items():
        model_path = MODEL_DIR / f"{name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(data['model'], f)
        print(f"✅ Сохранено: {model_path}")
    
    # Сохраняем scaler
    scaler_path = MODEL_DIR / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Сохранено: {scaler_path}")


def print_summary(results: dict):
    """
    Выводит итоговое сравнение моделей
    """
    print("\n" + "="*60)
    print("📊 ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*60)
    
    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "Модель", "Accuracy", "Precision", "Recall", "F1"
    ))
    print("-"*60)
    
    best_model = None
    best_f1 = 0
    
    for name, data in results.items():
        print("{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            name, data['accuracy'], data['precision'], data['recall'], data['f1']
        ))
        if data['f1'] > best_f1:
            best_f1 = data['f1']
            best_model = name
    
    print("-"*60)
    print(f"\n🏆 Лучшая модель: {best_model} (F1: {best_f1:.4f})")
    
    return best_model


def main():
    """
    Главная функция обучения
    """
    print("\n" + "="*60)
    print("🛡️  PHISHGUARD - ОБУЧЕНИЕ ML МОДЕЛЕЙ")
    print("    Дипломная работа: Обнаружение фишинга с ИИ")
    print("="*60)
    print(f"    Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 1. Загрузка реального датасета
    df = load_real_dataset()
    
    # 2. Обучение и оценка моделей
    results, scaler = train_and_evaluate_models(df)
    
    # 3. Сохранение моделей
    save_models(results, scaler)
    
    # 4. Итоговое сравнение
    best_model = print_summary(results)
    
    print("\n" + "="*60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*60)
    print(f"\n📁 Модели сохранены в: {MODEL_DIR}")
    print("\n🚀 Для запуска API выполните:")
    print("   cd phishing-detector-api")
    print("   uvicorn app.main:app --reload")
    print()


if __name__ == "__main__":
    main()
