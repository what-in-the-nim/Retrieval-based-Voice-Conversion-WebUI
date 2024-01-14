from .base import F0Predictor
from .dio import DioF0Predictor
from .harvest import HarvestF0Predictor
from .pm import PMF0Predictor


class F0PredictorFactory:
    @staticmethod
    def create(f0_method: str, hop_length: int, sample_rate: int) -> F0Predictor:
        if f0_method == "dio":
            cls = DioF0Predictor
        elif f0_method == "harvest":
            cls = HarvestF0Predictor
        elif f0_method == "pm":
            cls =  PMF0Predictor
        else:
            raise ValueError("Invalid predictor type")
        return cls(hop_length=hop_length, sampling_rate=sample_rate)


__all__ = [
    "F0PredictorFactory",
    "DioF0Predictor",
    "HarvestF0Predictor",
    "PMF0Predictor",
]
