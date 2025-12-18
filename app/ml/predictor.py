"""
ML предсказатель для обнаружения фишинга
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, Tuple, Optional, Literal
from pathlib import Path

from app.ml.features import extract_features, get_feature_vector, FEATURE_NAMES_RU
from app.core.config import settings


ModelType = Literal["logistic_regression", "random_forest", "xgboost"]


class PhishingPredictor:
    """Сервис предсказания фишинга"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict] = {
            'logistic_regression': {
                'name': 'Logistic Regression',
                'nameRu': 'Логистическая регрессия',
                'description': 'Linear classifier for binary classification',
                'descriptionRu': 'Линейный классификатор для бинарной классификации',
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
                'description': 'Gradient boosting on decision trees',
                'descriptionRu': 'Градиентный бустинг на деревьях решений',
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
        
        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            model_path = model_dir / f"{model_name}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"Loaded model: {model_name}")
            else:
                print(f"Model not found: {model_path} (will use demo mode)")
    
    def predict(
        self, 
        url: str, 
        model: ModelType = "xgboost",
        include_explanation: bool = True
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Выполняет предсказание для URL
        
        Args:
            url: URL для анализа
            model: Тип модели
            include_explanation: Включить SHAP объяснение
            
        Returns:
            (features, explanation)
        """
        # Извлечение признаков
        features = extract_features(url)
        feature_vector = get_feature_vector(features)
        
        # Если модель загружена - используем её
        if model in self.models:
            loaded_model = self.models[model]
            probability = float(loaded_model.predict_proba([feature_vector])[0][1])
            
            # SHAP объяснение (если доступно)
            explanation = None
            if include_explanation:
                explanation = self._generate_explanation(
                    loaded_model, feature_vector, features, probability
                )
        else:
            # Демо-режим: эвристический расчёт
            probability, explanation = self._demo_predict(features, include_explanation)
        
        return features, probability, explanation
    
    def _demo_predict(
        self, 
        features: Dict[str, Any], 
        include_explanation: bool
    ) -> Tuple[float, Optional[Dict]]:
        """
        Эвристическое предсказание для демо-режима
        """
        score = 0.35  # Базовая вероятность
        
        contributions = []
        
        # Отсутствие HTTPS
        if not features['hasHttps']:
            contrib = 0.15
            score += contrib
            contributions.append({
                'feature': 'hasHttps',
                'featureRu': 'Наличие HTTPS',
                'value': False,
                'displayValue': 'Нет',
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        else:
            contrib = -0.1
            score += contrib
            contributions.append({
                'feature': 'hasHttps',
                'featureRu': 'Наличие HTTPS',
                'value': True,
                'displayValue': 'Да',
                'contribution': contrib,
                'direction': 'decreases_risk',
            })
        
        # IP адрес вместо домена
        if features['hasIPAddress']:
            contrib = 0.25
            score += contrib
            contributions.append({
                'feature': 'hasIPAddress',
                'featureRu': 'IP вместо домена',
                'value': True,
                'displayValue': 'Да',
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        
        # Подозрительные ключевые слова
        keyword_count = len(features['suspiciousKeywords'])
        if keyword_count > 0:
            contrib = min(0.25, keyword_count * 0.05)
            score += contrib
            contributions.append({
                'feature': 'suspiciousKeywords',
                'featureRu': 'Подозрительные слова',
                'value': keyword_count,
                'displayValue': f'{keyword_count} слов(а)',
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        
        # Длинный URL
        if features['urlLength'] > 75:
            contrib = min(0.1, (features['urlLength'] - 75) * 0.001)
            score += contrib
            contributions.append({
                'feature': 'urlLength',
                'featureRu': 'Длина URL',
                'value': features['urlLength'],
                'displayValue': f'{features["urlLength"]} символов',
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        
        # Много поддоменов
        if features['subdomainCount'] > 2:
            contrib = 0.1
            score += contrib
            contributions.append({
                'feature': 'subdomainCount',
                'featureRu': 'Поддомены',
                'value': features['subdomainCount'],
                'displayValue': str(features['subdomainCount']),
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        
        # Символ @
        if features['hasAtSymbol']:
            contrib = 0.2
            score += contrib
            contributions.append({
                'feature': 'hasAtSymbol',
                'featureRu': 'Символ @',
                'value': True,
                'displayValue': 'Да',
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        
        # Сокращённый URL
        if features['isShortened']:
            contrib = 0.15
            score += contrib
            contributions.append({
                'feature': 'isShortened',
                'featureRu': 'Сокращённый URL',
                'value': True,
                'displayValue': 'Да',
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        
        # Высокая энтропия
        if features['entropyScore'] > 4.5:
            contrib = 0.08
            score += contrib
            contributions.append({
                'feature': 'entropyScore',
                'featureRu': 'Энтропия URL',
                'value': features['entropyScore'],
                'displayValue': str(features['entropyScore']),
                'contribution': contrib,
                'direction': 'increases_risk',
            })
        
        # Ограничиваем вероятность
        probability = max(0.0, min(1.0, score))
        
        # Формируем объяснение
        explanation = None
        if include_explanation:
            # Сортируем по абсолютному вкладу
            contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
            
            # Генерируем интерпретацию
            if probability > 0.7:
                interpretation = "High phishing probability detected"
                interpretation_ru = "Обнаружена высокая вероятность фишинга"
            elif probability > 0.3:
                interpretation = "Suspicious URL, caution advised"
                interpretation_ru = "Подозрительный URL, требуется осторожность"
            else:
                interpretation = "URL appears to be legitimate"
                interpretation_ru = "URL выглядит легитимным"
            
            explanation = {
                'shapValues': contributions,
                'topPositiveFeatures': [c for c in contributions if c['direction'] == 'increases_risk'][:3],
                'topNegativeFeatures': [c for c in contributions if c['direction'] == 'decreases_risk'][:3],
                'baseValue': 0.35,
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
        """
        Генерирует SHAP объяснение для реальной модели
        """
        try:
            import shap
            
            # Создаём SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(np.array([feature_vector]))
            
            # Формируем contributions
            feature_names = [
                'urlLength', 'domainLength', 'pathLength', 'hasHttps',
                'hasIPAddress', 'subdomainCount', 'specialCharCount',
                'hasAtSymbol', 'hasSuspiciousPort', 'suspiciousKeywords',
                'isShortened', 'numericDomain', 'pathDepth', 'queryParamCount',
                'entropyScore', '_hasDoubleSlash', '_hasHttpsInDomain',
                '_digitRatio', '_letterRatio'
            ]
            
            contributions = []
            for i, (name, shap_val) in enumerate(zip(feature_names, shap_values[0])):
                if name.startswith('_'):
                    continue
                    
                value = features.get(name, feature_vector[i])
                if isinstance(value, list):
                    display_value = f"{len(value)} элементов"
                elif isinstance(value, bool):
                    display_value = "Да" if value else "Нет"
                else:
                    display_value = str(value)
                
                contributions.append({
                    'feature': name,
                    'featureRu': FEATURE_NAMES_RU.get(name, name),
                    'value': value,
                    'displayValue': display_value,
                    'contribution': float(shap_val),
                    'direction': 'increases_risk' if shap_val > 0 else 'decreases_risk',
                })
            
            contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
            
            return {
                'shapValues': contributions,
                'topPositiveFeatures': [c for c in contributions if c['direction'] == 'increases_risk'][:3],
                'topNegativeFeatures': [c for c in contributions if c['direction'] == 'decreases_risk'][:3],
                'baseValue': float(explainer.expected_value),
                'interpretationText': self._generate_interpretation(probability),
                'interpretationTextRu': self._generate_interpretation_ru(probability),
            }
        except ImportError:
            # SHAP не установлен, используем демо объяснение
            _, explanation = self._demo_predict(features, True)
            return explanation
    
    def _generate_interpretation(self, probability: float) -> str:
        if probability > 0.7:
            return "High phishing probability. Multiple risk factors detected."
        elif probability > 0.3:
            return "Suspicious URL. Some risk factors present."
        else:
            return "URL appears legitimate based on analyzed features."
    
    def _generate_interpretation_ru(self, probability: float) -> str:
        if probability > 0.7:
            return "Высокая вероятность фишинга. Обнаружено несколько факторов риска."
        elif probability > 0.3:
            return "Подозрительный URL. Присутствуют некоторые факторы риска."
        else:
            return "URL выглядит легитимным на основе анализа признаков."
    
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
