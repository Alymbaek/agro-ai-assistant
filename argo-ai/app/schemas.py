from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    region: str
    month: int
    avg_temp: float
    max_temp: float
    precipitation: float


class AnalyzeResponse(BaseModel):
    yield_index: float
    risk_level: str




