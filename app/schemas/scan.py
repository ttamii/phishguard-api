"""
Pydantic схемы для API запросов и ответов
"""

from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Literal
from datetime import datetime
import uuid


class ScanURLRequest(BaseModel):
    """Запрос на сканирование URL"""
    url: str = Field(..., description="URL для проверки")
    model: Optional[Literal["logistic_regression", "random_forest", "xgboost"]] = Field(
        default="xgboost",
        description="ML модель для анализа"
    )
    include_explanation: bool = Field(
        default=True,
        description="Включить SHAP объяснение"
    )


class ScanMessageRequest(BaseModel):
    """Запрос на сканирование сообщения"""
    text: str = Field(..., description="Текст сообщения")
    extract_urls: bool = Field(default=True, description="Извлекать URL из текста")


class FeatureSet(BaseModel):
    """Набор признаков URL"""
    url_length: int = Field(..., alias="urlLength")
    domain_length: int = Field(..., alias="domainLength")
    path_length: int = Field(..., alias="pathLength")
    has_https: bool = Field(..., alias="hasHttps")
    has_ip_address: bool = Field(..., alias="hasIPAddress")
    subdomain_count: int = Field(..., alias="subdomainCount")
    special_char_count: int = Field(..., alias="specialCharCount")
    has_at_symbol: bool = Field(..., alias="hasAtSymbol")
    has_suspicious_port: bool = Field(..., alias="hasSuspiciousPort")
    suspicious_keywords: List[str] = Field(..., alias="suspiciousKeywords")
    is_shortened: bool = Field(..., alias="isShortened")
    numeric_domain: bool = Field(..., alias="numericDomain")
    path_depth: int = Field(..., alias="pathDepth")
    query_param_count: int = Field(..., alias="queryParamCount")
    entropy_score: float = Field(..., alias="entropyScore")
    
    class Config:
        populate_by_name = True


class FeatureContribution(BaseModel):
    """Вклад признака в предсказание"""
    feature: str
    feature_ru: str = Field(..., alias="featureRu")
    value: str | int | float | bool
    display_value: str = Field(..., alias="displayValue")
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]
    
    class Config:
        populate_by_name = True


class Explanation(BaseModel):
    """Объяснение решения модели"""
    shap_values: List[FeatureContribution] = Field(..., alias="shapValues")
    top_positive_features: List[FeatureContribution] = Field(..., alias="topPositiveFeatures")
    top_negative_features: List[FeatureContribution] = Field(..., alias="topNegativeFeatures")
    base_value: float = Field(..., alias="baseValue")
    interpretation_text: str = Field(..., alias="interpretationText")
    interpretation_text_ru: str = Field(..., alias="interpretationTextRu")
    
    class Config:
        populate_by_name = True


class ScanResult(BaseModel):
    """Результат сканирования URL"""
    scan_id: str = Field(..., alias="scanId")
    url: str
    is_phishing: bool = Field(..., alias="isPhishing")
    probability: float
    confidence: float
    classification: Literal["safe", "suspicious", "dangerous"]
    model_used: str = Field(..., alias="modelUsed")
    features: FeatureSet
    explanation: Optional[Explanation] = None
    timestamp: datetime
    scan_duration: int = Field(..., alias="scanDuration", description="мс")
    
    class Config:
        populate_by_name = True


class MessageScanResult(BaseModel):
    """Результат сканирования сообщения"""
    scan_id: str = Field(..., alias="scanId")
    original_text: str = Field(..., alias="originalText")
    extracted_urls: List[str] = Field(..., alias="extractedUrls")
    url_results: List[ScanResult] = Field(..., alias="urlResults")
    overall_risk: Literal["safe", "suspicious", "dangerous"] = Field(..., alias="overallRisk")
    timestamp: datetime
    
    class Config:
        populate_by_name = True


class ModelInfo(BaseModel):
    """Информация о ML модели"""
    id: str
    name: str
    name_ru: str = Field(..., alias="nameRu")
    description: str
    description_ru: str = Field(..., alias="descriptionRu")
    accuracy: float
    precision: float
    recall: float
    f1_score: float = Field(..., alias="f1Score")
    
    class Config:
        populate_by_name = True
