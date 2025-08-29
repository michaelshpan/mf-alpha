"""
Model loader with caching capabilities for trained ML models
"""
import pickle
import joblib
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from .config import PredictionConfig, DEFAULT_CONFIG
from .training_bridge import TrainingBridge

log = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for model wrappers"""
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on input data"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities if available"""
        pass

class SklearnModel(BaseModel):
    """Wrapper for sklearn-compatible models"""
    
    def __init__(self, model: Any, model_name: str):
        self.model = model
        self.model_name = model_name
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        try:
            # Suppress sklearn feature name warnings for RandomForest models
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*RandomForestRegressor was fitted without feature names")
                return self.model.predict(X)
        except Exception as e:
            log.error(f"Prediction failed for {self.model_name}: {e}")
            raise
            
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

class StatsModel(BaseModel):
    """Wrapper for statsmodels models (OLS)"""
    
    def __init__(self, model: Any, model_name: str):
        self.model = model
        self.model_name = model_name
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        try:
            # Add constant if needed for OLS
            if hasattr(self.model, 'fittedvalues') and 'const' in self.model.params.index:
                import statsmodels.api as sm
                X_with_const = sm.add_constant(X)
                return self.model.predict(X_with_const)
            else:
                return self.model.predict(X)
        except Exception as e:
            log.error(f"Prediction failed for {self.model_name}: {e}")
            raise
            
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Statsmodels typically don't have predict_proba"""
        return None

class ModelLoader:
    """
    Loads and caches trained ML models for prediction
    """
    
    def __init__(self, config: PredictionConfig = DEFAULT_CONFIG):
        self.config = config
        self.model_cache: Dict[str, BaseModel] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.training_bridge = TrainingBridge()
        
    def load_model(self, model_name: str, force_reload: bool = False) -> BaseModel:
        """
        Load a specific model by name
        """
        if not force_reload and model_name in self.model_cache:
            log.debug(f"Using cached model: {model_name}")
            return self.model_cache[model_name]
            
        log.info(f"Loading model: {model_name}")
        
        # Find model file
        model_path = self._find_model_path(model_name)
        if not model_path:
            raise FileNotFoundError(f"Model file not found for: {model_name}")
            
        # Load the model
        model = self._load_model_file(model_path, model_name)
        
        # Wrap in appropriate wrapper
        wrapped_model = self._wrap_model(model, model_name)
        
        # Cache if enabled
        if self.config.enable_model_caching:
            self.model_cache[model_name] = wrapped_model
            
        log.info(f"Successfully loaded model: {model_name}")
        return wrapped_model
    
    def load_models(self, model_names: Optional[List[str]] = None) -> Dict[str, BaseModel]:
        """
        Load multiple models
        """
        if model_names is None:
            model_names = self.config.default_models
            
        models = {}
        for name in model_names:
            try:
                models[name] = self.load_model(name)
            except Exception as e:
                log.error(f"Failed to load model {name}: {e}")
                
        if not models:
            raise RuntimeError("No models could be loaded")
            
        log.info(f"Loaded {len(models)} models: {list(models.keys())}")
        return models
    
    def _find_model_path(self, model_name: str) -> Optional[Path]:
        """
        Find the model file path for a given model name
        """
        model_dir = Path(self.config.model_directory)
        
        # Common file extensions for different model types
        extensions = ['.pkl', '.joblib', '.pickle', '.model', '.json']
        
        # Common naming patterns
        patterns = [
            f"{model_name}",
            f"{model_name}_model",
            f"model_{model_name}",
            f"{model_name.replace('_', '')}", 
            f"{model_name.replace('_', '-')}",
        ]
        
        for pattern in patterns:
            for ext in extensions:
                candidate = model_dir / f"{pattern}{ext}"
                if candidate.exists():
                    log.debug(f"Found model file: {candidate}")
                    return candidate
                    
        # Look for subdirectories
        for subdir in model_dir.iterdir():
            if subdir.is_dir() and model_name.lower() in subdir.name.lower():
                for pattern in patterns:
                    for ext in extensions:
                        candidate = subdir / f"{pattern}{ext}"
                        if candidate.exists():
                            log.debug(f"Found model file: {candidate}")
                            return candidate
        
        log.warning(f"Model file not found for: {model_name}")
        return None
    
    def _load_model_file(self, model_path: Path, model_name: str) -> Any:
        """
        Load model from file based on extension
        """
        try:
            if model_path.suffix in ['.pkl', '.pickle']:
                with open(model_path, 'rb') as f:
                    # Suppress XGBoost version compatibility warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*older version of XGBoost.*")
                        model_info = pickle.load(f)
                    
                # Check if this is a training bridge model (has 'model' key)
                if isinstance(model_info, dict) and 'model' in model_info:
                    log.info(f"Loading training bridge model: {model_name}")
                    return model_info['model']
                else:
                    return model_info
                    
            elif model_path.suffix == '.joblib':
                return joblib.load(model_path)
            else:
                # Try joblib first, then pickle
                try:
                    return joblib.load(model_path)
                except:
                    with open(model_path, 'rb') as f:
                        # Suppress XGBoost version compatibility warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message=".*older version of XGBoost.*")
                            model_info = pickle.load(f)
                        # Handle training bridge format
                        if isinstance(model_info, dict) and 'model' in model_info:
                            return model_info['model']
                        return model_info
        except Exception as e:
            log.error(f"Failed to load model file {model_path}: {e}")
            raise
    
    def _wrap_model(self, model: Any, model_name: str) -> BaseModel:
        """
        Wrap raw model in appropriate wrapper class
        """
        # Check for sklearn-style models
        if hasattr(model, 'predict') and hasattr(model, 'fit'):
            return SklearnModel(model, model_name)
            
        # Check for statsmodels
        if hasattr(model, 'fittedvalues') or 'statsmodels' in str(type(model)):
            return StatsModel(model, model_name)
            
        # Default to sklearn wrapper
        log.warning(f"Unknown model type for {model_name}, using sklearn wrapper")
        return SklearnModel(model, model_name)
    
    def get_model_info(self, model_name: str) -> Dict:
        """
        Get metadata about a loaded model
        """
        if model_name not in self.model_cache:
            self.load_model(model_name)
            
        model = self.model_cache[model_name]
        
        info = {
            'name': model_name,
            'type': type(model.model).__name__,
            'wrapper': type(model).__name__,
            'has_predict_proba': hasattr(model.model, 'predict_proba'),
        }
        
        # Add model-specific info
        if hasattr(model.model, 'feature_importances_'):
            info['has_feature_importance'] = True
        if hasattr(model.model, 'coef_'):
            info['has_coefficients'] = True
            
        return info
    
    def clear_cache(self) -> None:
        """Clear the model cache"""
        self.model_cache.clear()
        self.model_metadata.clear()
        log.info("Model cache cleared")
    
    def list_available_models(self) -> List[str]:
        """
        List all available models in the model directory
        """
        # First check training bridge models
        bridge_models = self.training_bridge.get_available_trained_models()
        available = []
        
        # Map training model names to prediction service names
        model_mapping = self.training_bridge.create_model_mapping()
        for training_name in bridge_models:
            if training_name in model_mapping:
                available.append(model_mapping[training_name])
        
        # Also check standard model directory
        model_dir = Path(self.config.model_directory)
        if model_dir.exists():
            extensions = ['.pkl', '.joblib', '.pickle', '.model']
            
            for file_path in model_dir.rglob('*'):
                if file_path.suffix in extensions:
                    # Extract model name from filename
                    name = file_path.stem
                    if any(keyword in name.lower() for keyword in ['model', 'gradient', 'forest', 'elastic', 'ols']):
                        # Remove _model suffix if present
                        clean_name = name.replace('_model', '')
                        if clean_name not in available:
                            available.append(clean_name)
                            
        return sorted(list(set(available)))

class ModelEnsemble:
    """
    Ensemble predictor that combines multiple models
    """
    
    def __init__(self, models: Dict[str, BaseModel], method: str = "mean", weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.method = method
        self.weights = weights or {name: 1.0 for name in models.keys()}
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions
        """
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                log.warning(f"Model {name} failed to predict: {e}")
                continue
                
        if not predictions:
            raise RuntimeError("No models produced predictions")
            
        # Combine predictions
        pred_array = np.column_stack(list(predictions.values()))
        
        if self.method == "mean":
            return np.mean(pred_array, axis=1)
        elif self.method == "median":
            return np.median(pred_array, axis=1)
        elif self.method == "weighted_mean":
            weights_array = np.array([self.weights.get(name, 1.0) for name in predictions.keys()])
            weights_array = weights_array / weights_array.sum()
            return np.average(pred_array, axis=1, weights=weights_array)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model
        """
        predictions = {}
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                log.warning(f"Model {name} failed to predict: {e}")
                
        return predictions