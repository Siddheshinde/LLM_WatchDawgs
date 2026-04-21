from enum import Enum
from pydantic import BaseModel

class RiskZone(str, Enum):
    RELIABLE = "RELIABLE"
    OVERCONFIDENT = "OVERCONFIDENT"
    UNSTABLE = "UNSTABLE"
    AMBIGUOUS = "AMBIGUOUS"

class RawMetrics(BaseModel):
    question: str
    uncertainty: float
    consistency: float

class RiskReport(BaseModel):
    question: str
    uncertainty: float
    consistency: float
    risk_zone: RiskZone
    risk_score: float
    calibration_score: float
    recommendation: str

class RiskEngine:
    def __init__(self, threshold_u: float = 0.4, threshold_c: float = 0.6):
        self.threshold_u = threshold_u
        self.threshold_c = threshold_c

    def determine_zone(self, u: float, c: float) -> RiskZone:
        if u < self.threshold_u and c >= self.threshold_c:
            return RiskZone.RELIABLE
        elif u < self.threshold_u and c < self.threshold_c:
            return RiskZone.OVERCONFIDENT
        elif u >= self.threshold_u and c < self.threshold_c:
            return RiskZone.UNSTABLE
        else:  
            return RiskZone.AMBIGUOUS

    def calculate_calibration(self, u: float, c: float) -> float:
        return round(c / (1.0 + u), 4)

    def calculate_risk_score(self, u: float, c: float) -> float:
        return round((0.6 * u) + (0.4 * (1.0 - c)), 4)

    def get_recommendation(self, zone: RiskZone) -> str:
        recommendations = {
            RiskZone.RELIABLE: "Safe: The answer is stable and confident.",
            RiskZone.OVERCONFIDENT: "Dangerous: Do not trust this answer; high risk of hallucination.",
            RiskZone.UNSTABLE: "Review: The model is highly uncertain and output varies.",
            RiskZone.AMBIGUOUS: "Clarify: The model is confident but outputs are contradictory."
        }
        return recommendations[zone]