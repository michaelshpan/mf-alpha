"""
Prediction Service for Mutual Fund Alpha Prediction

A standalone service for feeding external fund data into trained ML models
to predict alpha based on the DeMiguel et al. methodology.
"""

from .predictor import FundAlphaPredictor
from .config import PredictionConfig, DEFAULT_CONFIG
from .data_preprocessor import ETLDataPreprocessor
from .model_loader import ModelLoader, ModelEnsemble

__version__ = "1.0.0"
__all__ = [
    "FundAlphaPredictor",
    "PredictionConfig", 
    "DEFAULT_CONFIG",
    "ETLDataPreprocessor",
    "ModelLoader",
    "ModelEnsemble"
]