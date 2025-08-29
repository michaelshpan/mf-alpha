"""
Bridge module to interface with the main training pipeline in /mf-alpha/src
"""
import sys
import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, "/Users/michaelshpan/mf-alpha/src")

try:
    from models import get_model_specs, fit_and_predict
    from data_io import read_characteristics
    from utils import load_config
except ImportError as e:
    logging.warning(f"Could not import from main training pipeline: {e}")
    get_model_specs = None
    fit_and_predict = None
    read_characteristics = None
    load_config = None

log = logging.getLogger(__name__)

class TrainingBridge:
    """
    Bridge to interface with the main training pipeline for model access
    """
    
    def __init__(self, src_config_path: str = "/Users/michaelshpan/mf-alpha/src/config.yaml"):
        self.src_config_path = Path(src_config_path)
        self.config = None
        self.models_dir = Path("/Users/michaelshpan/mf-alpha/outputs/local/models")
        
        # Load configuration if available
        if self.src_config_path.exists() and load_config:
            try:
                self.config = load_config(self.src_config_path)
            except Exception as e:
                log.warning(f"Could not load training config: {e}")
    
    def get_available_trained_models(self) -> List[str]:
        """
        Get list of trained models available from the main pipeline
        """
        if not self.models_dir.exists():
            return []
            
        model_files = list(self.models_dir.glob("*_model.pkl"))
        return [f.stem.replace("_model", "") for f in model_files]
    
    def load_trained_model(self, model_name: str) -> Optional[Dict]:
        """
        Load a trained model saved by the main pipeline
        """
        model_file = self.models_dir / f"{model_name}_model.pkl"
        
        if not model_file.exists():
            log.warning(f"Trained model not found: {model_file}")
            return None
            
        try:
            with open(model_file, 'rb') as f:
                model_info = pickle.load(f)
            
            log.info(f"Loaded trained model: {model_name}")
            return model_info
            
        except Exception as e:
            log.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def get_training_feature_names(self) -> List[str]:
        """
        Get the feature names used in training (DeMiguel characteristics)
        """
        if self.config:
            # This matches the vars_list from main.py but excludes the target
            return [
                "alpha6_1mo_12m_lag_1", "alpha6_tstat_lag_1", "mtna_lag_1", "exp_ratio_lag_1",
                "age_lag_1", "flow_12m_lag_1", "mgr_tenure_lag_1", "turn_ratio_lag_1", 
                "flow_vol_12m_lag_1", "value_added_12m_lag_1", "beta_market_tstat_lag_1", 
                "beta_profit_tstat_lag_1", "beta_invest_tstat_lag_1", "beta_size_tstat_lag_1", 
                "beta_value_tstat_lag_1", "beta_mom_tstat_lag_1", "R2_lag_1"
            ]
        else:
            # Fallback to DeMiguel feature names
            return [
                "realized alpha lagged", "alpha (intercept t-stat)", "total net assets", "expense ratio",
                "age", "flows", "manager_tenure", "turnover ratio", "vol_of_flows", "value_added",
                "market beta t-stat", "profit. beta t-stat", "invest. beta t-stat", 
                "size beta t-stat", "value beta t-stat", "momentum beta t-stat", "R2"
            ]
    
    def train_models_if_needed(self, force_retrain: bool = False) -> bool:
        """
        Train models if they don't exist or if forced
        """
        if not force_retrain and self.get_available_trained_models():
            log.info("Trained models already exist")
            return True
            
        if not self.config or not get_model_specs:
            log.error("Cannot train models: main pipeline not available")
            return False
            
        log.info("Training models via main pipeline...")
        
        try:
            # This would require running the full main.py pipeline
            # For now, just indicate that manual training is needed
            log.warning("Automated training not implemented. Please run: python src/main.py")
            return False
            
        except Exception as e:
            log.error(f"Training failed: {e}")
            return False
    
    def create_model_mapping(self) -> Dict[str, str]:
        """
        Create mapping between training model names and prediction service model names
        """
        return {
            "ols": "ols",
            "enet": "elastic_net", 
            "rf": "random_forest",
            "xgb": "gradient_boosting"  # XGBoost is often called gradient boosting
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the training configuration and available models
        """
        available_models = self.get_available_trained_models()
        mapping = self.create_model_mapping()
        
        info = {
            "src_config_path": str(self.src_config_path),
            "models_directory": str(self.models_dir),
            "config_loaded": self.config is not None,
            "pipeline_available": get_model_specs is not None,
            "available_trained_models": available_models,
            "model_name_mapping": mapping,
            "training_features": self.get_training_feature_names()
        }
        
        if self.config:
            info["training_config"] = {
                "strategies": self.config.get("experiment", {}).get("models", []),
                "panel_years": self.config.get("experiment", {}).get("panel_years"),
                "seed": self.config.get("experiment", {}).get("seed")
            }
        
        return info

def create_linkage_config() -> Dict[str, Any]:
    """
    Create a configuration that links the prediction service with the training pipeline
    """
    bridge = TrainingBridge()
    
    # Get model mapping
    model_mapping = bridge.create_model_mapping()
    available_models = bridge.get_available_trained_models()
    
    # Map training model names to prediction service names
    prediction_service_models = []
    for training_name in available_models:
        if training_name in model_mapping:
            prediction_service_models.append(model_mapping[training_name])
    
    linkage_config = {
        "bridge_info": bridge.get_model_info(),
        "model_directory": str(bridge.models_dir),
        "available_models": prediction_service_models,
        "feature_mapping": {
            # Map ETL features to training features
            "training_features": bridge.get_training_feature_names(),
            "requires_transformation": True
        },
        "recommended_action": (
            "Run 'python src/main.py' to train models" 
            if not available_models 
            else "Models available for prediction"
        )
    }
    
    return linkage_config

if __name__ == "__main__":
    # Demonstrate the bridge functionality
    bridge = TrainingBridge()
    info = bridge.get_model_info()
    
    print("Training Bridge Information:")
    print("="*50)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Show linkage configuration
    print("\nLinkage Configuration:")
    print("="*50)
    linkage = create_linkage_config()
    for key, value in linkage.items():
        print(f"{key}: {value}")