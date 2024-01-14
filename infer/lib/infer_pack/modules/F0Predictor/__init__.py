from .base import F0Predictor
from .dio import DioF0Predictor
from .harvest import HarvestF0Predictor
from .pm import PMF0Predictor

class F0PredictorFactory:
    @staticmethod
    def create(predictor_type) -> F0Predictor:
        if predictor_type == "dio":
            return DioF0Predictor()
        elif predictor_type == "harvest":
            return HarvestF0Predictor()
        elif predictor_type == "pm":
            return PMF0Predictor()
        else:
            raise ValueError("Invalid predictor type")

__all__ = ["F0PredictorFactory", "DioF0Predictor", "HarvestF0Predictor", "PMF0Predictor"]
