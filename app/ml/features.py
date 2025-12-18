"""
Извлечение признаков из URL для ML моделей
Полная реализация 111 признаков из датасета GregaVrbancic/Phishing-Dataset
Адаптировано для Казахстана

Автор: Tamiris
Дата: 2025
"""

import re
import math
import socket
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List, Tuple
from collections import Counter


# Список популярных TLD
TLDS = [
    'com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'eu', 'asia', 'mobi',
    'info', 'biz', 'name', 'pro', 'aero', 'coop', 'museum', 'jobs', 'travel',
    'kz', 'ru', 'uk', 'de', 'fr', 'cn', 'jp', 'br', 'in', 'au', 'us', 'ca',
    'io', 'co', 'me', 'tv', 'cc', 'ws', 'xyz', 'top', 'site', 'online', 'store',
]

# Сокращатели URL
URL_SHORTENERS = [
    'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'is.gd', 'buff.ly',
    'adf.ly', 'shorte.st', 'bc.vc', 'j.mp', 'clicky.me', 'cutt.ly', 'rb.gy',
    'clck.ru', 'qps.ru',  # Казахстанские/русские
]


def count_char(text: str, char: str) -> int:
    """Подсчёт количества символа в строке"""
    return text.count(char)


def count_vowels(text: str) -> int:
    """Подсчёт гласных"""
    vowels = 'aeiouаеёиоуыэюя'
    return sum(1 for c in text.lower() if c in vowels)


def is_ip_address(domain: str) -> int:
    """Проверка является ли домен IP-адресом"""
    ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
    return 1 if re.match(ip_pattern, domain.split(':')[0]) else 0


def has_server_client(domain: str) -> int:
    """Проверка наличия server/client в домене"""
    keywords = ['server', 'client', 'cliente', 'servidor']
    domain_lower = domain.lower()
    return 1 if any(kw in domain_lower for kw in keywords) else 0


def extract_url_features(url: str) -> Dict[str, int]:
    """Извлекает признаки из всего URL"""
    return {
        'qty_dot_url': count_char(url, '.'),
        'qty_hyphen_url': count_char(url, '-'),
        'qty_underline_url': count_char(url, '_'),
        'qty_slash_url': count_char(url, '/'),
        'qty_questionmark_url': count_char(url, '?'),
        'qty_equal_url': count_char(url, '='),
        'qty_at_url': count_char(url, '@'),
        'qty_and_url': count_char(url, '&'),
        'qty_exclamation_url': count_char(url, '!'),
        'qty_space_url': count_char(url, ' '),
        'qty_tilde_url': count_char(url, '~'),
        'qty_comma_url': count_char(url, ','),
        'qty_plus_url': count_char(url, '+'),
        'qty_asterisk_url': count_char(url, '*'),
        'qty_hashtag_url': count_char(url, '#'),
        'qty_dollar_url': count_char(url, '$'),
        'qty_percent_url': count_char(url, '%'),
        'qty_tld_url': sum(1 for tld in TLDS if f'.{tld}' in url.lower()),
        'length_url': len(url),
    }


def extract_domain_features(domain: str) -> Dict[str, int]:
    """Извлекает признаки из домена"""
    return {
        'qty_dot_domain': count_char(domain, '.'),
        'qty_hyphen_domain': count_char(domain, '-'),
        'qty_underline_domain': count_char(domain, '_'),
        'qty_slash_domain': count_char(domain, '/'),
        'qty_questionmark_domain': count_char(domain, '?'),
        'qty_equal_domain': count_char(domain, '='),
        'qty_at_domain': count_char(domain, '@'),
        'qty_and_domain': count_char(domain, '&'),
        'qty_exclamation_domain': count_char(domain, '!'),
        'qty_space_domain': count_char(domain, ' '),
        'qty_tilde_domain': count_char(domain, '~'),
        'qty_comma_domain': count_char(domain, ','),
        'qty_plus_domain': count_char(domain, '+'),
        'qty_asterisk_domain': count_char(domain, '*'),
        'qty_hashtag_domain': count_char(domain, '#'),
        'qty_dollar_domain': count_char(domain, '$'),
        'qty_percent_domain': count_char(domain, '%'),
        'qty_vowels_domain': count_vowels(domain),
        'domain_length': len(domain),
        'domain_in_ip': is_ip_address(domain),
        'server_client_domain': has_server_client(domain),
    }


def extract_directory_features(path: str) -> Dict[str, int]:
    """Извлекает признаки из директории (путь без файла)"""
    # Отделяем директорию от файла
    parts = path.rsplit('/', 1)
    directory = parts[0] if len(parts) > 1 else ''
    
    if not directory:
        # Нет директории - возвращаем -1 для всех признаков
        return {
            'qty_dot_directory': -1,
            'qty_hyphen_directory': -1,
            'qty_underline_directory': -1,
            'qty_slash_directory': -1,
            'qty_questionmark_directory': -1,
            'qty_equal_directory': -1,
            'qty_at_directory': -1,
            'qty_and_directory': -1,
            'qty_exclamation_directory': -1,
            'qty_space_directory': -1,
            'qty_tilde_directory': -1,
            'qty_comma_directory': -1,
            'qty_plus_directory': -1,
            'qty_asterisk_directory': -1,
            'qty_hashtag_directory': -1,
            'qty_dollar_directory': -1,
            'qty_percent_directory': -1,
            'directory_length': -1,
        }
    
    return {
        'qty_dot_directory': count_char(directory, '.'),
        'qty_hyphen_directory': count_char(directory, '-'),
        'qty_underline_directory': count_char(directory, '_'),
        'qty_slash_directory': count_char(directory, '/'),
        'qty_questionmark_directory': count_char(directory, '?'),
        'qty_equal_directory': count_char(directory, '='),
        'qty_at_directory': count_char(directory, '@'),
        'qty_and_directory': count_char(directory, '&'),
        'qty_exclamation_directory': count_char(directory, '!'),
        'qty_space_directory': count_char(directory, ' '),
        'qty_tilde_directory': count_char(directory, '~'),
        'qty_comma_directory': count_char(directory, ','),
        'qty_plus_directory': count_char(directory, '+'),
        'qty_asterisk_directory': count_char(directory, '*'),
        'qty_hashtag_directory': count_char(directory, '#'),
        'qty_dollar_directory': count_char(directory, '$'),
        'qty_percent_directory': count_char(directory, '%'),
        'directory_length': len(directory),
    }


def extract_file_features(path: str) -> Dict[str, int]:
    """Извлекает признаки из имени файла"""
    # Отделяем файл от директории
    parts = path.rsplit('/', 1)
    filename = parts[-1] if parts else ''
    
    # Убираем query parameters
    if '?' in filename:
        filename = filename.split('?')[0]
    
    if not filename or filename == path:
        return {
            'qty_dot_file': -1,
            'qty_hyphen_file': -1,
            'qty_underline_file': -1,
            'qty_slash_file': -1,
            'qty_questionmark_file': -1,
            'qty_equal_file': -1,
            'qty_at_file': -1,
            'qty_and_file': -1,
            'qty_exclamation_file': -1,
            'qty_space_file': -1,
            'qty_tilde_file': -1,
            'qty_comma_file': -1,
            'qty_plus_file': -1,
            'qty_asterisk_file': -1,
            'qty_hashtag_file': -1,
            'qty_dollar_file': -1,
            'qty_percent_file': -1,
            'file_length': -1,
        }
    
    return {
        'qty_dot_file': count_char(filename, '.'),
        'qty_hyphen_file': count_char(filename, '-'),
        'qty_underline_file': count_char(filename, '_'),
        'qty_slash_file': 0,  # В имени файла не может быть /
        'qty_questionmark_file': count_char(filename, '?'),
        'qty_equal_file': count_char(filename, '='),
        'qty_at_file': count_char(filename, '@'),
        'qty_and_file': count_char(filename, '&'),
        'qty_exclamation_file': count_char(filename, '!'),
        'qty_space_file': count_char(filename, ' '),
        'qty_tilde_file': count_char(filename, '~'),
        'qty_comma_file': count_char(filename, ','),
        'qty_plus_file': count_char(filename, '+'),
        'qty_asterisk_file': count_char(filename, '*'),
        'qty_hashtag_file': count_char(filename, '#'),
        'qty_dollar_file': count_char(filename, '$'),
        'qty_percent_file': count_char(filename, '%'),
        'file_length': len(filename),
    }


def extract_params_features(query: str) -> Dict[str, int]:
    """Извлекает признаки из параметров запроса"""
    if not query:
        return {
            'qty_dot_params': -1,
            'qty_hyphen_params': -1,
            'qty_underline_params': -1,
            'qty_slash_params': -1,
            'qty_questionmark_params': -1,
            'qty_equal_params': -1,
            'qty_at_params': -1,
            'qty_and_params': -1,
            'qty_exclamation_params': -1,
            'qty_space_params': -1,
            'qty_tilde_params': -1,
            'qty_comma_params': -1,
            'qty_plus_params': -1,
            'qty_asterisk_params': -1,
            'qty_hashtag_params': -1,
            'qty_dollar_params': -1,
            'qty_percent_params': -1,
            'params_length': -1,
            'tld_present_params': -1,
            'qty_params': -1,
        }
    
    params = parse_qs(query)
    tld_in_params = 1 if any(f'.{tld}' in query.lower() for tld in TLDS) else 0
    
    return {
        'qty_dot_params': count_char(query, '.'),
        'qty_hyphen_params': count_char(query, '-'),
        'qty_underline_params': count_char(query, '_'),
        'qty_slash_params': count_char(query, '/'),
        'qty_questionmark_params': count_char(query, '?'),
        'qty_equal_params': count_char(query, '='),
        'qty_at_params': count_char(query, '@'),
        'qty_and_params': count_char(query, '&'),
        'qty_exclamation_params': count_char(query, '!'),
        'qty_space_params': count_char(query, ' '),
        'qty_tilde_params': count_char(query, '~'),
        'qty_comma_params': count_char(query, ','),
        'qty_plus_params': count_char(query, '+'),
        'qty_asterisk_params': count_char(query, '*'),
        'qty_hashtag_params': count_char(query, '#'),
        'qty_dollar_params': count_char(query, '$'),
        'qty_percent_params': count_char(query, '%'),
        'params_length': len(query),
        'tld_present_params': tld_in_params,
        'qty_params': len(params),
    }


def extract_external_features(url: str, domain: str) -> Dict[str, Any]:
    """
    Извлекает внешние признаки (требуют сетевых запросов)
    Для production используем упрощённые значения
    """
    # Проверка на email в URL
    email_pattern = r'[\w\.-]+@[\w\.-]+'
    email_in_url = 1 if re.search(email_pattern, url) else 0
    
    # Проверка на сокращённый URL
    url_shortened = 1 if any(shortener in domain.lower() for shortener in URL_SHORTENERS) else 0
    
    # Для остальных признаков используем значения по умолчанию
    # В реальном production эти данные получали бы из DNS/WHOIS/SSL
    return {
        'email_in_url': email_in_url,
        'time_response': 0.5,  # Средний ответ
        'domain_spf': 1,  # Предполагаем SPF есть
        'asn_ip': 0,  # Неизвестно
        'time_domain_activation': -1,  # Неизвестно
        'time_domain_expiration': -1,  # Неизвестно  
        'qty_ip_resolved': 1,  # Предполагаем 1 IP
        'qty_nameservers': 2,  # Стандартно 2
        'qty_mx_servers': 1,  # Стандартно 1
        'ttl_hostname': 3600,  # Стандартный TTL
        'tls_ssl_certificate': 1 if url.startswith('https') else 0,
        'qty_redirects': 0,  # Предполагаем нет редиректов
        'url_google_index': 0,  # Неизвестно
        'domain_google_index': 0,  # Неизвестно
        'url_shortened': url_shortened,
    }


def extract_all_features(url: str) -> Dict[str, Any]:
    """
    Извлекает все 111 признаков из URL
    Совместимо с датасетом GregaVrbancic/Phishing-Dataset
    """
    # Нормализация URL
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        parsed = urlparse(url)
    except Exception:
        parsed = urlparse('https://invalid.url')
    
    domain = parsed.netloc.lower()
    path = parsed.path
    query = parsed.query
    
    # Собираем все признаки
    features = {}
    
    # URL признаки (19)
    features.update(extract_url_features(url))
    
    # Domain признаки (21)
    features.update(extract_domain_features(domain))
    
    # Directory признаки (18)
    features.update(extract_directory_features(path))
    
    # File признаки (18)
    features.update(extract_file_features(path))
    
    # Params признаки (20)
    features.update(extract_params_features(query))
    
    # External признаки (15)
    features.update(extract_external_features(url, domain))
    
    return features


def get_feature_vector(features: Dict[str, Any]) -> List[float]:
    """
    Преобразует словарь признаков в вектор для ML модели
    Порядок соответствует датасету GregaVrbancic/Phishing-Dataset
    """
    # Порядок признаков как в датасете
    feature_order = [
        # URL features (19)
        'qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url',
        'qty_questionmark_url', 'qty_equal_url', 'qty_at_url', 'qty_and_url',
        'qty_exclamation_url', 'qty_space_url', 'qty_tilde_url', 'qty_comma_url',
        'qty_plus_url', 'qty_asterisk_url', 'qty_hashtag_url', 'qty_dollar_url',
        'qty_percent_url', 'qty_tld_url', 'length_url',
        
        # Domain features (21)
        'qty_dot_domain', 'qty_hyphen_domain', 'qty_underline_domain', 'qty_slash_domain',
        'qty_questionmark_domain', 'qty_equal_domain', 'qty_at_domain', 'qty_and_domain',
        'qty_exclamation_domain', 'qty_space_domain', 'qty_tilde_domain', 'qty_comma_domain',
        'qty_plus_domain', 'qty_asterisk_domain', 'qty_hashtag_domain', 'qty_dollar_domain',
        'qty_percent_domain', 'qty_vowels_domain', 'domain_length', 'domain_in_ip',
        'server_client_domain',
        
        # Directory features (18)
        'qty_dot_directory', 'qty_hyphen_directory', 'qty_underline_directory', 'qty_slash_directory',
        'qty_questionmark_directory', 'qty_equal_directory', 'qty_at_directory', 'qty_and_directory',
        'qty_exclamation_directory', 'qty_space_directory', 'qty_tilde_directory', 'qty_comma_directory',
        'qty_plus_directory', 'qty_asterisk_directory', 'qty_hashtag_directory', 'qty_dollar_directory',
        'qty_percent_directory', 'directory_length',
        
        # File features (18)
        'qty_dot_file', 'qty_hyphen_file', 'qty_underline_file', 'qty_slash_file',
        'qty_questionmark_file', 'qty_equal_file', 'qty_at_file', 'qty_and_file',
        'qty_exclamation_file', 'qty_space_file', 'qty_tilde_file', 'qty_comma_file',
        'qty_plus_file', 'qty_asterisk_file', 'qty_hashtag_file', 'qty_dollar_file',
        'qty_percent_file', 'file_length',
        
        # Params features (20)
        'qty_dot_params', 'qty_hyphen_params', 'qty_underline_params', 'qty_slash_params',
        'qty_questionmark_params', 'qty_equal_params', 'qty_at_params', 'qty_and_params',
        'qty_exclamation_params', 'qty_space_params', 'qty_tilde_params', 'qty_comma_params',
        'qty_plus_params', 'qty_asterisk_params', 'qty_hashtag_params', 'qty_dollar_params',
        'qty_percent_params', 'params_length', 'tld_present_params', 'qty_params',
        
        # External features (15)
        'email_in_url', 'time_response', 'domain_spf', 'asn_ip',
        'time_domain_activation', 'time_domain_expiration', 'qty_ip_resolved',
        'qty_nameservers', 'qty_mx_servers', 'ttl_hostname', 'tls_ssl_certificate',
        'qty_redirects', 'url_google_index', 'domain_google_index', 'url_shortened',
    ]
    
    return [float(features.get(name, 0)) for name in feature_order]


def get_feature_names() -> List[str]:
    """Возвращает названия всех признаков в правильном порядке"""
    return [
        'qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url',
        'qty_questionmark_url', 'qty_equal_url', 'qty_at_url', 'qty_and_url',
        'qty_exclamation_url', 'qty_space_url', 'qty_tilde_url', 'qty_comma_url',
        'qty_plus_url', 'qty_asterisk_url', 'qty_hashtag_url', 'qty_dollar_url',
        'qty_percent_url', 'qty_tld_url', 'length_url',
        'qty_dot_domain', 'qty_hyphen_domain', 'qty_underline_domain', 'qty_slash_domain',
        'qty_questionmark_domain', 'qty_equal_domain', 'qty_at_domain', 'qty_and_domain',
        'qty_exclamation_domain', 'qty_space_domain', 'qty_tilde_domain', 'qty_comma_domain',
        'qty_plus_domain', 'qty_asterisk_domain', 'qty_hashtag_domain', 'qty_dollar_domain',
        'qty_percent_domain', 'qty_vowels_domain', 'domain_length', 'domain_in_ip',
        'server_client_domain',
        'qty_dot_directory', 'qty_hyphen_directory', 'qty_underline_directory', 'qty_slash_directory',
        'qty_questionmark_directory', 'qty_equal_directory', 'qty_at_directory', 'qty_and_directory',
        'qty_exclamation_directory', 'qty_space_directory', 'qty_tilde_directory', 'qty_comma_directory',
        'qty_plus_directory', 'qty_asterisk_directory', 'qty_hashtag_directory', 'qty_dollar_directory',
        'qty_percent_directory', 'directory_length',
        'qty_dot_file', 'qty_hyphen_file', 'qty_underline_file', 'qty_slash_file',
        'qty_questionmark_file', 'qty_equal_file', 'qty_at_file', 'qty_and_file',
        'qty_exclamation_file', 'qty_space_file', 'qty_tilde_file', 'qty_comma_file',
        'qty_plus_file', 'qty_asterisk_file', 'qty_hashtag_file', 'qty_dollar_file',
        'qty_percent_file', 'file_length',
        'qty_dot_params', 'qty_hyphen_params', 'qty_underline_params', 'qty_slash_params',
        'qty_questionmark_params', 'qty_equal_params', 'qty_at_params', 'qty_and_params',
        'qty_exclamation_params', 'qty_space_params', 'qty_tilde_params', 'qty_comma_params',
        'qty_plus_params', 'qty_asterisk_params', 'qty_hashtag_params', 'qty_dollar_params',
        'qty_percent_params', 'params_length', 'tld_present_params', 'qty_params',
        'email_in_url', 'time_response', 'domain_spf', 'asn_ip',
        'time_domain_activation', 'time_domain_expiration', 'qty_ip_resolved',
        'qty_nameservers', 'qty_mx_servers', 'ttl_hostname', 'tls_ssl_certificate',
        'qty_redirects', 'url_google_index', 'domain_google_index', 'url_shortened',
    ]


# Названия признаков на русском для объяснений
FEATURE_NAMES_RU = {
    'qty_dot_url': 'Точки в URL',
    'qty_hyphen_url': 'Дефисы в URL',
    'qty_underline_url': 'Подчёркивания в URL',
    'qty_slash_url': 'Слэши в URL',
    'qty_at_url': 'Символы @ в URL',
    'length_url': 'Длина URL',
    'qty_dot_domain': 'Точки в домене',
    'qty_hyphen_domain': 'Дефисы в домене',
    'domain_length': 'Длина домена',
    'domain_in_ip': 'IP вместо домена',
    'tls_ssl_certificate': 'SSL сертификат',
    'url_shortened': 'Сокращённый URL',
    'qty_params': 'Параметры запроса',
    'email_in_url': 'Email в URL',
    'directory_length': 'Длина директории',
    'file_length': 'Длина файла',
    'params_length': 'Длина параметров',
    'qty_vowels_domain': 'Гласные в домене',
    'server_client_domain': 'Server/Client в домене',
}


# Для обратной совместимости со старым кодом
def extract_features(url: str) -> Dict[str, Any]:
    """Обёртка для совместимости со старым API"""
    all_features = extract_all_features(url)
    
    # Добавляем человеко-читаемые признаки для UI
    parsed = urlparse(url if url.startswith('http') else f'https://{url}')
    domain = parsed.netloc.lower()
    
    # Подозрительные ключевые слова
    suspicious_keywords = [
        'login', 'signin', 'verify', 'secure', 'account', 'update', 'confirm',
        'banking', 'password', 'credential', 'suspend', 'unusual', 'activity',
        'locked', 'expired', 'urgent', 'immediately', 'click', 'free', 'winner',
        'kaspi', 'halyk', 'forte', 'jusan', 'egov', 'bank',  # Казахстанские
    ]
    url_lower = url.lower()
    found_keywords = [kw for kw in suspicious_keywords if kw in url_lower]
    
    all_features.update({
        # Совместимость со старым UI
        'urlLength': all_features['length_url'],
        'domainLength': all_features['domain_length'],
        'pathLength': all_features.get('directory_length', 0) if all_features.get('directory_length', -1) >= 0 else 0,
        'hasHttps': 1 if url.startswith('https') else 0,
        'hasIPAddress': all_features['domain_in_ip'],
        'subdomainCount': max(0, all_features['qty_dot_domain'] - 1),
        'specialCharCount': all_features['qty_at_url'] + all_features['qty_hyphen_url'] + all_features['qty_underline_url'],
        'hasAtSymbol': all_features['qty_at_url'] > 0,
        'hasSuspiciousPort': ':8080' in url or ':8443' in url,
        'suspiciousKeywords': found_keywords,
        'isShortened': all_features['url_shortened'] == 1,
        'numericDomain': domain.replace('.', '').isdigit() if domain else False,
        'pathDepth': all_features['qty_slash_url'] - 2,  # Минус протокол
        'queryParamCount': all_features.get('qty_params', 0) if all_features.get('qty_params', -1) >= 0 else 0,
        'entropyScore': round(calculate_entropy(url), 2),
    })
    
    return all_features


def calculate_entropy(text: str) -> float:
    """Рассчитывает энтропию Шеннона для строки"""
    if not text:
        return 0.0
    
    freq = Counter(text)
    total = len(text)
    entropy = 0.0
    
    for count in freq.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)
    
    return entropy
