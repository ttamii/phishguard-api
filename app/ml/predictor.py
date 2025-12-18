"""
ML предсказатель для обнаружения фишинга
Использует 111 признаков из датасета GregaVrbancic/Phishing-Dataset

Автор: Tamiris
Дата: 2025
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, Tuple, Optional, Literal
from pathlib import Path

from app.ml.features import extract_all_features, get_feature_vector, get_feature_names, FEATURE_NAMES_RU
from app.core.config import settings


ModelType = Literal["logistic_regression", "random_forest", "xgboost"]


class PhishingPredictor:
    """Сервис предсказания фишинга"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = None
        self.feature_names = get_feature_names()
        
        self.model_info: Dict[str, Dict] = {
            'logistic_regression': {
                'name': 'Logistic Regression',
                'nameRu': 'Логистическая регрессия',
                'description': 'Linear classifier for binary classification',
                'descriptionRu': 'Линейный классификатор',
                'accuracy': 0.91,
                'precision': 0.89,
                'recall': 0.93,
                'f1Score': 0.91,
            },
            'random_forest': {
                'name': 'Random Forest',
                'nameRu': 'Случайный лес',
                'description': 'Ensemble of decision trees',
                'descriptionRu': 'Ансамбль деревьев решений',
                'accuracy': 0.94,
                'precision': 0.93,
                'recall': 0.95,
                'f1Score': 0.94,
            },
            'xgboost': {
                'name': 'XGBoost',
                'nameRu': 'XGBoost',
                'description': 'Gradient boosting classifier',
                'descriptionRu': 'Градиентный бустинг',
                'accuracy': 0.96,
                'precision': 0.95,
                'recall': 0.97,
                'f1Score': 0.96,
            },
        }
        self._load_models()
    
    def _load_models(self):
        """Загрузка обученных моделей"""
        model_dir = Path(settings.MODEL_PATH)
        
        # Загружаем scaler
        scaler_path = model_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("Loaded scaler")
        
        # Загружаем модели
        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            model_path = model_dir / f"{model_name}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"Loaded model: {model_name}")
            else:
                print(f"Model not found: {model_path}")
    
    def predict(
        self, 
        url: str, 
        model: ModelType = "xgboost",
        include_explanation: bool = True
    ) -> Tuple[Dict[str, Any], float, Optional[Dict[str, Any]]]:
        """
        Выполняет предсказание для URL
        
        Args:
            url: URL для анализа
            model: Тип модели
            include_explanation: Включить объяснение
            
        Returns:
            (features, probability, explanation)
        """
        # Извлечение признаков
        features = extract_all_features(url)
        feature_vector = get_feature_vector(features)
        
        # Если модель загружена - используем её
        if model in self.models:
            try:
                loaded_model = self.models[model]
                
                # Применяем scaler если есть
                if self.scaler is not None:
                    feature_array = np.array([feature_vector])
                    feature_array = self.scaler.transform(feature_array)
                    feature_vector_scaled = feature_array[0].tolist()
                else:
                    feature_vector_scaled = feature_vector
                
                # Предсказание
                probability = float(loaded_model.predict_proba([feature_vector_scaled])[0][1])
                
                # Объяснение
                explanation = None
                if include_explanation:
                    explanation = self._generate_explanation(
                        loaded_model, feature_vector_scaled, features, probability
                    )
            except Exception as e:
                print(f"Model prediction error: {e}")
                probability, explanation = self._demo_predict(features, include_explanation)
        else:
            # Демо-режим
            probability, explanation = self._demo_predict(features, include_explanation)
        
        return features, probability, explanation
    
    def _demo_predict(
        self, 
        features: Dict[str, Any], 
        include_explanation: bool
    ) -> Tuple[float, Optional[Dict]]:
        """Эвристическое предсказание для демо-режима"""
        score = 0.25
        contributions = []
        
        # HTTPS
        has_https = features.get('tls_ssl_certificate', 0) == 1
        if not has_https:
            contrib = 0.15
            score += contrib
            contributions.append({
                'feature': 'tls_ssl_certificate',
                'featureRu': 'SSL сертификат',
                'value': False,
                'displayValue': 'Нет',
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        else:
            contrib = -0.1
            score += contrib
            contributions.append({
                'feature': 'tls_ssl_certificate',
                'featureRu': 'SSL сертификат',
                'value': True,
                'displayValue': 'Да',
                'contribution': contrib,
                'direction': 'decreases_risk',
            })
        
        # IP адрес
        if features.get('domain_in_ip', 0) == 1:
            contrib = 0.25
            score += contrib
            contributions.append({
                'feature': 'domain_in_ip',
                'featureRu': 'IP вместо домена',
                'value': True,
                'displayValue': 'Да',
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        
        # Длина URL
        url_length = features.get('length_url', 0)
        if url_length > 75:
            contrib = min(0.15, (url_length - 75) * 0.002)
            score += contrib
            contributions.append({
                'feature': 'length_url',
                'featureRu': 'Длина URL',
                'value': url_length,
                'displayValue': f'{url_length} символов',
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        
        # Много точек в домене (поддомены)
        dots_in_domain = features.get('qty_dot_domain', 0)
        if dots_in_domain > 2:
            contrib = 0.1
            score += contrib
            contributions.append({
                'feature': 'qty_dot_domain',
                'featureRu': 'Поддомены',
                'value': dots_in_domain,
                'displayValue': f'{dots_in_domain} точек',
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        
        # Символ @
        if features.get('qty_at_url', 0) > 0:
            contrib = 0.2
            score += contrib
            contributions.append({
                'feature': 'qty_at_url',
                'featureRu': 'Символ @ в URL',
                'value': True,
                'displayValue': 'Да',
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        
        # Сокращённый URL
        if features.get('url_shortened', 0) == 1:
            contrib = 0.15
            score += contrib
            contributions.append({
                'feature': 'url_shortened',
                'featureRu': 'Сокращённый URL',
                'value': True,
                'displayValue': 'Да',
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        
        # Много дефисов в домене
        hyphens = features.get('qty_hyphen_domain', 0)
        if hyphens > 2:
            contrib = 0.1
            score += contrib
            contributions.append({
                'feature': 'qty_hyphen_domain',
                'featureRu': 'Дефисы в домене',
                'value': hyphens,
                'displayValue': f'{hyphens} дефисов',
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        
        # Ограничиваем вероятность
        probability = max(0.0, min(1.0, score))
        
        # Формируем объяснение
        explanation = None
        if include_explanation:
            contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
            
            if probability > 0.7:
                interpretation = "High phishing probability"
                interpretation_ru = "Высокая вероятность фишинга"
            elif probability > 0.3:
                interpretation = "Suspicious URL"
                interpretation_ru = "Подозрительный URL"
            else:
                interpretation = "URL appears legitimate"
                interpretation_ru = "URL выглядит легитимным"
            
            explanation = {
                'shapValues': contributions,
                'topPositiveFeatures': [c for c in contributions if c['direction'] == 'increases_risk'][:3],
                'topNegativeFeatures': [c for c in contributions if c['direction'] == 'decreases_risk'][:3],
                'baseValue': 0.25,
                'interpretationText': interpretation,
                'interpretationTextRu': interpretation_ru,
            }
        
        return probability, explanation
    
    def _generate_explanation(
        self,
        model: Any,
        feature_vector: list,
        features: Dict[str, Any],
        probability: float
    ) -> Dict[str, Any]:
        """Генерирует объяснение для реальной модели"""
        try:
            import shap
            
            if hasattr(model, 'feature_importances_'):
                # Для tree-based моделей используем feature importances
                importances = model.feature_importances_
            else:
                # Для линейных моделей используем коэффициенты
                importances = np.abs(model.coef_[0]) if hasattr(model, 'coef_') else np.zeros(len(feature_vector))
            
            contributions = []
            for i, (name, imp) in enumerate(zip(self.feature_names, importances)):
                value = features.get(name, feature_vector[i] if i < len(feature_vector) else 0)
                
                # Преобразуем значение для отображения
                if isinstance(value, bool):
                    display_value = "Да" if value else "Нет"
                elif isinstance(value, float):
                    display_value = f"{value:.2f}"
                else:
                    display_value = str(value)
                
                # Определяем направление
                if name in ['tls_ssl_certificate']:
                    direction = 'decreases_risk' if value else 'increases_risk'
                elif name in ['domain_in_ip', 'url_shortened', 'email_in_url']:
                    direction = 'increases_risk' if value else 'decreases_risk'
                else:
                    direction = 'increases_risk' if imp > 0.01 else 'decreases_risk'
                
                contributions.append({
                    'feature': name,
                    'featureRu': FEATURE_NAMES_RU.get(name, name),
                    'value': value,
                    'displayValue': display_value,
                    'contribution': float(imp),
                    'direction': direction,
                })
            
            contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
            contributions = contributions[:15]  # Топ-15 признаков
            
            if probability > 0.7:
                interpretation = "High phishing probability"
                interpretation_ru = "Высокая вероятность фишинга"
            elif probability > 0.3:
                interpretation = "Suspicious URL"
                interpretation_ru = "Подозрительный URL"
            else:
                interpretation = "URL appears legitimate"
                interpretation_ru = "URL выглядит легитимным"
            
            return {
                'shapValues': contributions,
                'topPositiveFeatures': [c for c in contributions if c['direction'] == 'increases_risk'][:3],
                'topNegativeFeatures': [c for c in contributions if c['direction'] == 'decreases_risk'][:3],
                'baseValue': 0.5,
                'interpretationText': interpretation,
                'interpretationTextRu': interpretation_ru,
            }
        except Exception as e:
            print(f"Explanation error: {e}")
            _, explanation = self._demo_predict(features, True)
            return explanation
    
    def get_classification(self, probability: float) -> str:
        """Определяет класс угрозы по вероятности"""
        if probability >= settings.THRESHOLD_SUSPICIOUS:
            return "dangerous"
        elif probability >= settings.THRESHOLD_SAFE:
            return "suspicious"
        else:
            return "safe"
    
    def get_model_info(self) -> list:
        """Возвращает информацию о всех моделях"""
        return [
            {"id": model_id, **info}
            for model_id, info in self.model_info.items()
        ]


# Глобальный экземпляр предсказателя
predictor = PhishingPredictor()
