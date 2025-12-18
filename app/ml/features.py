"""
Извлечение признаков из URL для ML моделей
"""

import re
import math
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List
from collections import Counter


# Подозрительные ключевые слова
SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'verify', 'secure', 'account', 'update', 'confirm',
    'banking', 'password', 'credential', 'suspend', 'unusual', 'activity',
    'locked', 'expired', 'urgent', 'immediately', 'click', 'here', 'now',
    'free', 'winner', 'prize', 'congratulations', 'selected', 'offer',
    'limited', 'act', 'fast', 'paypal', 'ebay', 'amazon', 'apple', 'microsoft',
    'google', 'facebook', 'netflix', 'support', 'service', 'customer', 'help',
    'security', 'alert', 'warning', 'notification', 'invoice', 'payment',
    'refund', 'transaction', 'authorize', 'validate', 'restore', 'recover',
]

# Сокращатели URL
URL_SHORTENERS = [
    'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'is.gd', 'buff.ly',
    'adf.ly', 'shorte.st', 'bc.vc', 'j.mp', 'clicky.me', 'cutt.ly', 'rb.gy',
]


def extract_features(url: str) -> Dict[str, Any]:
    """
    Извлекает признаки из URL для ML классификации
    
    Args:
        url: URL для анализа
        
    Returns:
        Словарь признаков
    """
    # Парсинг URL
    try:
        parsed = urlparse(url)
    except Exception:
        parsed = urlparse('')
    
    domain = parsed.netloc.lower()
    path = parsed.path
    query = parsed.query
    
    # Базовые признаки длины
    url_length = len(url)
    domain_length = len(domain)
    path_length = len(path)
    
    # Проверка на HTTPS
    has_https = parsed.scheme.lower() == 'https'
    
    # Проверка на IP адрес вместо домена
    ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
    has_ip_address = bool(re.match(ip_pattern, domain.split(':')[0]))
    
    # Подсчёт поддоменов
    domain_parts = domain.split('.')
    subdomain_count = max(0, len(domain_parts) - 2)
    
    # Специальные символы в URL
    special_chars = ['@', '-', '_', '~', '.', '!', '$', '&', "'", '(', ')', '*', '+', ',', ';', '=']
    special_char_count = sum(url.count(char) for char in special_chars)
    
    # Символ @ в URL (часто используется в фишинге)
    has_at_symbol = '@' in url
    
    # Подозрительный порт
    suspicious_ports = ['8080', '8443', '8888', '81', '82', '8000', '8001']
    has_suspicious_port = any(f':{port}' in url for port in suspicious_ports)
    
    # Поиск подозрительных ключевых слов
    url_lower = url.lower()
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url_lower]
    
    # Проверка на сокращённый URL
    is_shortened = any(shortener in domain for shortener in URL_SHORTENERS)
    
    # Числовой домен
    domain_without_tld = domain_parts[0] if domain_parts else ''
    numeric_domain = domain_without_tld.isdigit()
    
    # Глубина пути
    path_depth = len([p for p in path.split('/') if p])
    
    # Количество параметров запроса
    query_params = parse_qs(query)
    query_param_count = len(query_params)
    
    # Энтропия URL (мера случайности)
    entropy_score = calculate_entropy(url)
    
    # Дополнительные признаки
    has_double_slash = '//' in path
    has_https_in_domain = 'https' in domain or 'http' in domain
    digit_ratio = sum(c.isdigit() for c in url) / max(len(url), 1)
    letter_ratio = sum(c.isalpha() for c in url) / max(len(url), 1)
    
    return {
        'urlLength': url_length,
        'domainLength': domain_length,
        'pathLength': path_length,
        'hasHttps': has_https,
        'hasIPAddress': has_ip_address,
        'subdomainCount': subdomain_count,
        'specialCharCount': special_char_count,
        'hasAtSymbol': has_at_symbol,
        'hasSuspiciousPort': has_suspicious_port,
        'suspiciousKeywords': found_keywords,
        'isShortened': is_shortened,
        'numericDomain': numeric_domain,
        'pathDepth': path_depth,
        'queryParamCount': query_param_count,
        'entropyScore': round(entropy_score, 2),
        # Дополнительные признаки для модели
        '_hasDoubleSlash': has_double_slash,
        '_hasHttpsInDomain': has_https_in_domain,
        '_digitRatio': round(digit_ratio, 4),
        '_letterRatio': round(letter_ratio, 4),
        '_keywordCount': len(found_keywords),
    }


def calculate_entropy(text: str) -> float:
    """
    Рассчитывает энтропию Шеннона для строки
    Высокая энтропия может указывать на случайно сгенерированный домен
    """
    if not text:
        return 0.0
    
    # Подсчёт частоты символов
    freq = Counter(text)
    total = len(text)
    
    # Расчёт энтропии
    entropy = 0.0
    for count in freq.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)
    
    return entropy


def get_feature_vector(features: Dict[str, Any]) -> List[float]:
    """
    Преобразует словарь признаков в вектор для ML модели
    """
    return [
        features['urlLength'],
        features['domainLength'],
        features['pathLength'],
        1 if features['hasHttps'] else 0,
        1 if features['hasIPAddress'] else 0,
        features['subdomainCount'],
        features['specialCharCount'],
        1 if features['hasAtSymbol'] else 0,
        1 if features['hasSuspiciousPort'] else 0,
        len(features['suspiciousKeywords']),
        1 if features['isShortened'] else 0,
        1 if features['numericDomain'] else 0,
        features['pathDepth'],
        features['queryParamCount'],
        features['entropyScore'],
        1 if features.get('_hasDoubleSlash') else 0,
        1 if features.get('_hasHttpsInDomain') else 0,
        features.get('_digitRatio', 0),
        features.get('_letterRatio', 0),
    ]


# Названия признаков на русском для объяснений
FEATURE_NAMES_RU = {
    'urlLength': 'Длина URL',
    'domainLength': 'Длина домена',
    'pathLength': 'Длина пути',
    'hasHttps': 'Наличие HTTPS',
    'hasIPAddress': 'IP вместо домена',
    'subdomainCount': 'Количество поддоменов',
    'specialCharCount': 'Спецсимволы',
    'hasAtSymbol': 'Символ @',
    'hasSuspiciousPort': 'Подозрительный порт',
    'suspiciousKeywords': 'Подозрительные слова',
    'isShortened': 'Сокращённый URL',
    'numericDomain': 'Цифровой домен',
    'pathDepth': 'Глубина пути',
    'queryParamCount': 'Параметры запроса',
    'entropyScore': 'Энтропия URL',
}
