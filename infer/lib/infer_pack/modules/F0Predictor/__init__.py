from .base import F0Predictor
from .dio import DioF0Predictor
from .harvest import HarvestF0Predictor
from .pm import PMF0Predictor

class F0PredictorFactory:
    @staticmethod
    def create(f0_method: str, hop_length: int, sample_rate: int) -> F0Predictor:
        if f0_method == "dio":
            return DioF0Predictor(hop_length=hop_length, sampling_rate=sample_rate)
        elif f0_method == "harvest":
            return HarvestF0Predictor(hop_length=hop_length, sampling_rate=sample_rate)
        elif f0_method == "pm":
            return PMF0Predictor(hop_length=hop_length, sampling_rate=sample_rate)
        else:
            raise ValueError("Invalid predictor type")

__all__ = ["F0PredictorFactory", "DioF0Predictor", "HarvestF0Predictor", "PMF0Predictor"]
