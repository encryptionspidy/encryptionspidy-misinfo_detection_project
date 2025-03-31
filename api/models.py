# api/models.py
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class QueryModel(BaseModel):
    question: str

class AnalysisResponse(BaseModel):
    result: Dict[str, Any]
    analysis_type: str
    status: str
    message: Optional[str] = None

class FactCheckResponse(BaseModel):
    verdict: Optional[str] = None
    explanation: Optional[str] = None
    source: Optional[str] = None
    confidence: Optional[float] = None
