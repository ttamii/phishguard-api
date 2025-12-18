"""
API роуты для сканирования
"""

import uuid
import re
import time
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException

from app.schemas.scan import (
    ScanURLRequest,
    ScanMessageRequest,
    ScanResult,
    MessageScanResult,
    Explanation,
)
from app.ml.predictor import predictor

router = APIRouter()


@router.post("/url", response_model=ScanResult)
async def scan_url(request: ScanURLRequest):
    """
    Сканирование URL на фишинг
    
    - **url**: URL для проверки
    - **model**: ML модель (logistic_regression, random_forest, xgboost)
    - **include_explanation**: Включить объяснение
    """
    start_time = time.time()
    
    # Нормализация URL
    url = request.url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        # Получаем предсказание
        features, probability, explanation = predictor.predict(
            url=url,
            model=request.model,
            include_explanation=request.include_explanation
        )
        
        # Определяем классификацию
        classification = predictor.get_classification(probability)
        
        # Время выполнения
        scan_duration = int((time.time() - start_time) * 1000)
        
        # Фильтруем features для UI (только основные)
        ui_features = {
            'urlLength': features.get('length_url', features.get('urlLength', 0)),
            'domainLength': features.get('domain_length', features.get('domainLength', 0)),
            'pathLength': features.get('directory_length', features.get('pathLength', 0)),
            'hasHttps': features.get('tls_ssl_certificate', 0) == 1,
            'hasIPAddress': features.get('domain_in_ip', 0) == 1,
            'subdomainCount': features.get('qty_dot_domain', 0),
            'specialCharCount': features.get('qty_at_url', 0) + features.get('qty_hyphen_url', 0),
            'hasAtSymbol': features.get('qty_at_url', 0) > 0,
            'hasSuspiciousPort': False,
            'suspiciousKeywords': features.get('suspiciousKeywords', []),
            'isShortened': features.get('url_shortened', 0) == 1,
            'numericDomain': False,
            'pathDepth': features.get('qty_slash_url', 0),
            'queryParamCount': features.get('qty_params', 0) if features.get('qty_params', -1) >= 0 else 0,
            'entropyScore': features.get('entropyScore', 0),
        }
        
        # Формируем ответ
        return ScanResult(
            scanId=str(uuid.uuid4()),
            url=url,
            isPhishing=probability >= 0.5,
            probability=round(probability, 4),
            confidence=round(min(probability * 1.1, 0.99) if probability > 0.5 else min((1 - probability) * 1.1, 0.99), 4),
            classification=classification,
            modelUsed=request.model,
            features=ui_features,
            explanation=Explanation(**explanation) if explanation else None,
            timestamp=datetime.now(),
            scanDuration=scan_duration,
        )
    except Exception as e:
        print(f"Scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message", response_model=MessageScanResult)
async def scan_message(request: ScanMessageRequest):
    """
    Сканирование текстового сообщения на фишинговые ссылки
    """
    start_time = time.time()
    
    # Извлечение URL из текста
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    extracted_urls = re.findall(url_pattern, request.text) if request.extract_urls else []
    
    # Сканируем каждый URL
    url_results = []
    for url in extracted_urls:
        try:
            features, probability, explanation = predictor.predict(
                url=url,
                model="xgboost",
                include_explanation=True
            )
            
            classification = predictor.get_classification(probability)
            scan_duration = int((time.time() - start_time) * 1000)
            
            ui_features = {
                'urlLength': features.get('length_url', 0),
                'domainLength': features.get('domain_length', 0),
                'hasHttps': features.get('tls_ssl_certificate', 0) == 1,
            }
            
            url_results.append(ScanResult(
                scanId=str(uuid.uuid4()),
                url=url,
                isPhishing=probability >= 0.5,
                probability=round(probability, 4),
                confidence=round(min(probability * 1.1, 0.99), 4),
                classification=classification,
                modelUsed="xgboost",
                features=ui_features,
                explanation=Explanation(**explanation) if explanation else None,
                timestamp=datetime.now(),
                scanDuration=scan_duration,
            ))
        except Exception:
            continue
    
    # Определяем общий уровень риска
    if not url_results:
        overall_risk = "safe"
    elif any(r.classification == "dangerous" for r in url_results):
        overall_risk = "dangerous"
    elif any(r.classification == "suspicious" for r in url_results):
        overall_risk = "suspicious"
    else:
        overall_risk = "safe"
    
    return MessageScanResult(
        scanId=str(uuid.uuid4()),
        originalText=request.text[:500],
        extractedUrls=extracted_urls,
        urlResults=url_results,
        overallRisk=overall_risk,
        timestamp=datetime.now(),
    )


@router.get("/{scan_id}/explanation")
async def get_explanation(scan_id: str):
    """Получение объяснения для предыдущего сканирования"""
    raise HTTPException(
        status_code=404,
        detail="Используйте include_explanation=true при сканировании."
    )
