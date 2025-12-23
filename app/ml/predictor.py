"""
ML предсказатель для обнаружения фишинга
Улучшенный эвристический анализатор с whitelist/blacklist и typosquatting

Автор: Tamiris
Дата: 2025
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, Tuple, Optional, Literal, List
from pathlib import Path
from urllib.parse import urlparse
import re

from app.ml.features import extract_all_features, get_feature_vector, get_feature_names, FEATURE_NAMES_RU
from app.core.config import settings


ModelType = Literal["logistic_regression", "random_forest", "xgboost"]


# Белый список - гарантированно безопасные домены
WHITELIST_DOMAINS = {
    # Казахстанские банки
    'kaspi.kz', 'my.kaspi.kz',
    'halykbank.kz', 'homebank.kz',
    'fortebank.com', 'forte.kz', 'online.forte.kz',
    'jusan.kz', 'jysanbank.kz',
    'berekebank.kz', 'eubank.kz', 'bankrbk.kz',
    'bcc.kz', 'centercredit.kz', 'sberbank.kz',
    
    # Госсервисы
    'egov.kz', 'elicense.kz', 'adilet.zan.kz',
    'gov.kz', 'stat.gov.kz', 'data.egov.kz',
    
    # Телеком
    'beeline.kz', 'kcell.kz', 'tele2.kz', 'altel.kz',
    
    # E-commerce
    'wildberries.kz', 'arbuz.kz', 'technodom.kz',
    'sulpak.kz', 'mechta.kz', 'kolesa.kz', 'krisha.kz',
    
    # Медиа
    'tengrinews.kz', 'nur.kz', 'informburo.kz', 'zakon.kz',
    
    # Международные
    'google.com', 'google.kz', 'youtube.com',
    'facebook.com', 'instagram.com', 'twitter.com',
    'linkedin.com', 'github.com', 'microsoft.com',
    'apple.com', 'amazon.com', 'netflix.com',
}

# Известные бренды для typosquatting детекции
KNOWN_BRANDS = [
    'kaspi', 'halyk', 'forte', 'jusan', 'egov', 'beeline', 'kcell',
    'tele2', 'wildberries', 'google', 'facebook', 'instagram',
    'amazon', 'netflix', 'microsoft', 'apple', 'paypal', 'ebay',
]

# Подозрительные TLD
SUSPICIOUS_TLDS = {
    'xyz', 'top', 'click', 'link', 'online', 'site', 'fun',
    'club', 'work', 'live', 'store', 'tech', 'space', 'icu',
}

# Подозрительные ключевые слова
PHISHING_KEYWORDS = [
    'login', 'signin', 'verify', 'secure', 'account', 'update',
    'confirm', 'password', 'credential', 'suspend', 'locked',
    'expired', 'urgent', 'immediately', 'winner', 'prize', 'free',
    'gift', 'bonus', 'congratulations', 'selected', 'claim',
]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Расстояние Левенштейна между двумя строками"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def check_typosquatting(domain: str) -> Tuple[bool, Optional[str]]:
    """
    Проверяет является ли домен typosquatting известного бренда
    Возвращает (is_typosquatting, matched_brand)
    """
    # Извлекаем основную часть домена
    domain_parts = domain.lower().split('.')
    if not domain_parts:
        return False, None
    
    main_part = domain_parts[0]
    
    for brand in KNOWN_BRANDS:
        # Точное совпадение - не typosquatting
        if main_part == brand:
            return False, None
        
        # Проверяем расстояние Левенштейна
        distance = levenshtein_distance(main_part, brand)
        
        # Если очень похоже (1-2 символа разницы) - подозрительно
        if 1 <= distance <= 2 and len(main_part) >= 4:
            return True, brand
        
        # Проверяем содержит ли домен бренд с добавками
        if brand in main_part and main_part != brand:
            # kaspi-secure, kaspi-login, kaspii и т.д.
            if any(kw in main_part for kw in ['secure', 'login', 'verify', 'update', 'bank', 'online']):
                return True, brand
    
    return False, None


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
                'description': 'Linear classifier',
                'descriptionRu': 'Линейный классификатор',
                'accuracy': 0.901,
                'precision': 0.894,
                'recall': 0.919,
                'f1Score': 0.906,
            },
            'random_forest': {
                'name': 'Random Forest',
                'nameRu': 'Случайный лес',
                'description': 'Ensemble of decision trees',
                'descriptionRu': 'Ансамбль деревьев',
                'accuracy': 0.946,
                'precision': 0.943,
                'recall': 0.954,
                'f1Score': 0.948,
            },
            'xgboost': {
                'name': 'XGBoost',
                'nameRu': 'XGBoost',
                'description': 'Gradient boosting',
                'descriptionRu': 'Градиентный бустинг',
                'accuracy': 0.945,
                'precision': 0.944,
                'recall': 0.951,
                'f1Score': 0.947,
            },
        }
        self._load_models()
    
    def _load_models(self):
        """Загрузка обученных моделей"""
        model_dir = Path(settings.MODEL_PATH)
        
        scaler_path = model_dir / "scaler.pkl"
        if scaler_path.exists():
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("Loaded scaler")
            except Exception as e:
                print(f"Failed to load scaler: {e}")
        
        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            model_path = model_dir / f"{model_name}.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"Loaded model: {model_name}")
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
    
    def predict(
        self, 
        url: str, 
        model: ModelType = "xgboost",
        include_explanation: bool = True
    ) -> Tuple[Dict[str, Any], float, Optional[Dict[str, Any]]]:
        """Выполняет предсказание для URL"""
        
        # Нормализация URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Извлечение признаков
        features = extract_all_features(url)
        
        # Парсим домен
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
        except:
            domain = ""
        
        # Проверяем whitelist
        for safe_domain in WHITELIST_DOMAINS:
            if domain == safe_domain or domain.endswith('.' + safe_domain):
                probability = 0.05
                explanation = self._create_explanation(
                    features, probability,
                    main_reason="Домен в белом списке",
                    main_reason_en="Domain is whitelisted"
                )
                return features, probability, explanation
        
        # Проверяем typosquatting
        is_typo, matched_brand = check_typosquatting(domain)
        if is_typo:
            probability = 0.92
            explanation = self._create_explanation(
                features, probability,
                main_reason=f"Похож на {matched_brand} (typosquatting)",
                main_reason_en=f"Looks like {matched_brand} (typosquatting)"
            )
            return features, probability, explanation
        
        # Пробуем ML модель
        if model in self.models:
            try:
                feature_vector = get_feature_vector(features)
                loaded_model = self.models[model]
                
                if self.scaler is not None:
                    feature_array = np.array([feature_vector])
                    feature_array = self.scaler.transform(feature_array)
                    feature_vector = feature_array[0].tolist()
                
                probability = float(loaded_model.predict_proba([feature_vector])[0][1])
                explanation = self._generate_ml_explanation(features, probability)
                return features, probability, explanation
            except Exception as e:
                print(f"Model prediction error: {e}")
        
        # Эвристический анализ
        probability, explanation = self._heuristic_predict(features, url, domain)
        return features, probability, explanation
    
    def _heuristic_predict(
        self, 
        features: Dict[str, Any],
        url: str,
        domain: str
    ) -> Tuple[float, Dict]:
        """Улучшенный эвристический анализ"""
        score = 0.0
        contributions = []
        
        # 1. HTTPS (-0.15 если есть, +0.2 если нет)
        has_https = url.startswith('https://')
        if has_https:
            score -= 0.1
            contributions.append({
                'feature': 'https',
                'featureRu': 'HTTPS протокол',
                'value': True,
                'displayValue': 'Да',
                'contribution': -0.1,
                'direction': 'decreases_risk',
            })
        else:
            score += 0.25
            contributions.append({
                'feature': 'https',
                'featureRu': 'HTTPS протокол',
                'value': False,
                'displayValue': 'Нет',
                'contribution': 0.25,
                'direction': 'increases_risk',
            })
        
        # 2. IP адрес вместо домена (+0.35)
        if features.get('domain_in_ip', 0) == 1:
            score += 0.35
            contributions.append({
                'feature': 'ip_address',
                'featureRu': 'IP вместо домена',
                'value': True,
                'displayValue': 'Да',
                'contribution': 0.35,
                'direction': 'increases_risk',
            })
        
        # 3. Подозрительный TLD (+0.2)
        tld = domain.split('.')[-1] if '.' in domain else ''
        if tld in SUSPICIOUS_TLDS:
            score += 0.2
            contributions.append({
                'feature': 'suspicious_tld',
                'featureRu': 'Подозрительный TLD',
                'value': tld,
                'displayValue': f'.{tld}',
                'contribution': 0.2,
                'direction': 'increases_risk',
            })
        
        # 4. Длина URL (+0.1 если > 75)
        url_length = len(url)
        if url_length > 75:
            contrib = min(0.15, (url_length - 75) * 0.002)
            score += contrib
            contributions.append({
                'feature': 'url_length',
                'featureRu': 'Длина URL',
                'value': url_length,
                'displayValue': f'{url_length} символов',
                'contribution': round(contrib, 3),
                'direction': 'increases_risk',
            })
        
        # 5. Много поддоменов (+0.1 за каждый после 2)
        subdomain_count = domain.count('.')
        if subdomain_count > 2:
            contrib = min(0.2, (subdomain_count - 2) * 0.1)
            score += contrib
            contributions.append({
                'feature': 'subdomains',
                'featureRu': 'Поддомены',
                'value': subdomain_count,
                'displayValue': f'{subdomain_count} уровней',
                'contribution': round(contrib, 3),
                'direction': 'increases_risk',
            })
        
        # 6. Символ @ в URL (+0.3)
        if '@' in url:
            score += 0.3
            contributions.append({
                'feature': 'at_symbol',
                'featureRu': 'Символ @ в URL',
                'value': True,
                'displayValue': 'Да',
                'contribution': 0.3,
                'direction': 'increases_risk',
            })
        
        # 7. Фишинговые ключевые слова (+0.05 за каждое)
        url_lower = url.lower()
        found_keywords = [kw for kw in PHISHING_KEYWORDS if kw in url_lower]
        if found_keywords:
            contrib = min(0.25, len(found_keywords) * 0.05)
            score += contrib
            contributions.append({
                'feature': 'phishing_keywords',
                'featureRu': 'Подозрительные слова',
                'value': found_keywords,
                'displayValue': ', '.join(found_keywords[:3]),
                'contribution': round(contrib, 3),
                'direction': 'increases_risk',
            })
        
        # 8. Много дефисов в домене (+0.1)
        hyphen_count = domain.count('-')
        if hyphen_count > 2:
            contrib = min(0.15, hyphen_count * 0.05)
            score += contrib
            contributions.append({
                'feature': 'hyphens',
                'featureRu': 'Дефисы в домене',
                'value': hyphen_count,
                'displayValue': f'{hyphen_count} дефисов',
                'contribution': round(contrib, 3),
                'direction': 'increases_risk',
            })
        
        # 9. Цифры в домене (+0.1)
        digit_count = sum(c.isdigit() for c in domain.split('.')[0])
        if digit_count > 2:
            contrib = min(0.1, digit_count * 0.02)
            score += contrib
            contributions.append({
                'feature': 'digits_in_domain',
                'featureRu': 'Цифры в домене',
                'value': digit_count,
                'displayValue': f'{digit_count} цифр',
                'contribution': round(contrib, 3),
                'direction': 'increases_risk',
            })
        
        # 10. Известный домен второго уровня (-0.15)
        for safe in ['kz', 'ru', 'com', 'org', 'net', 'edu', 'gov']:
            if domain.endswith(f'.{safe}') and subdomain_count <= 1:
                score -= 0.05
                break
        
        # Нормализация вероятности
        probability = max(0.02, min(0.98, score + 0.35))
        
        # Сортируем по вкладу
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        # Интерпретация
        if probability > 0.7:
            interpretation = "Высокая вероятность фишинга"
            interpretation_en = "High phishing probability"
        elif probability > 0.4:
            interpretation = "Подозрительный URL"
            interpretation_en = "Suspicious URL"
        else:
            interpretation = "URL выглядит безопасным"
            interpretation_en = "URL appears safe"
        
        explanation = {
            'shapValues': contributions,
            'topPositiveFeatures': [c for c in contributions if c['direction'] == 'increases_risk'][:3],
            'topNegativeFeatures': [c for c in contributions if c['direction'] == 'decreases_risk'][:3],
            'baseValue': 0.35,
            'interpretationText': interpretation_en,
            'interpretationTextRu': interpretation,
        }
        
        return probability, explanation
    
    def _create_explanation(
        self,
        features: Dict,
        probability: float,
        main_reason: str,
        main_reason_en: str
    ) -> Dict:
        """Создаёт объяснение для whitelist/blacklist"""
        return {
            'shapValues': [{
                'feature': 'domain_check',
                'featureRu': main_reason,
                'value': True,
                'displayValue': 'Да',
                'contribution': probability - 0.5,
                'direction': 'decreases_risk' if probability < 0.5 else 'increases_risk',
            }],
            'topPositiveFeatures': [],
            'topNegativeFeatures': [],
            'baseValue': 0.5,
            'interpretationText': main_reason_en,
            'interpretationTextRu': main_reason,
        }
    
    def _generate_ml_explanation(
        self,
        features: Dict,
        probability: float
    ) -> Dict:
        """Объяснение для ML модели"""
        if probability > 0.7:
            interpretation_ru = "Модель определила высокий риск фишинга"
            interpretation_en = "Model detected high phishing risk"
        elif probability > 0.4:
            interpretation_ru = "Модель обнаружила подозрительные признаки"
            interpretation_en = "Model found suspicious features"
        else:
            interpretation_ru = "Модель не обнаружила угроз"
            interpretation_en = "Model found no threats"
        
        return {
            'shapValues': [],
            'topPositiveFeatures': [],
            'topNegativeFeatures': [],
            'baseValue': 0.5,
            'interpretationText': interpretation_en,
            'interpretationTextRu': interpretation_ru,
        }
    
    def get_classification(self, probability: float) -> str:
        """Определяет класс угрозы"""
        if probability >= 0.7:
            return "dangerous"
        elif probability >= 0.4:
            return "suspicious"
        else:
            return "safe"
    
    def get_model_info(self) -> list:
        """Информация о моделях"""
        return [
            {"id": model_id, **info}
            for model_id, info in self.model_info.items()
        ]


# Глобальный экземпляр
predictor = PhishingPredictor()
