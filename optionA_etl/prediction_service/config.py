"""
Configuration for the prediction service
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass
class PredictionConfig:
    """Configuration for prediction service"""
    
    # Model settings  
    model_directory: str = "/Users/michaelshpan/mf-alpha/outputs/local/models"
    supported_models: List[str] = field(default_factory=lambda: [
        "xgb", "rf", "enet", "ols"
    ])
    default_models: List[str] = field(default_factory=lambda: [
        "xgb", "rf"
    ])
    
    # Data processing settings
    batch_size: int = 1000
    min_observations: int = 30
    max_missing_ratio: float = 0.5
    
    # Output settings
    output_format: str = "parquet"  # parquet, csv, json
    include_confidence_intervals: bool = True
    ensemble_method: str = "mean"  # mean, weighted_mean, median
    
    # Performance settings
    enable_model_caching: bool = True
    parallel_prediction: bool = True
    n_jobs: int = -1
    
    # Training pipeline features (17 features, excluding current alpha)
    required_features: List[str] = field(default_factory=lambda: [
        "realized alpha lagged", 
        "alpha (intercept t-stat)",
        "total net assets",
        "expense ratio",
        "age",
        "flows",
        "manager_tenure", 
        "turnover ratio",
        "vol_of_flows",
        "value_added",
        "market beta t-stat",
        "profit. beta t-stat",
        "invest. beta t-stat", 
        "size beta t-stat",
        "value beta t-stat",
        "momentum beta t-stat",
        "R2"
    ])
    
    # Feature engineering settings
    feature_scaling: bool = True
    handle_missing_values: str = "median_fill"  # median_fill, mean_fill, drop, forward_fill
    outlier_handling: str = "winsorize"  # winsorize, clip, remove
    winsorize_limits: tuple = (0.01, 0.99)
    
    def validate(self) -> None:
        """Validate configuration settings"""
        if not Path(self.model_directory).exists():
            raise ValueError(f"Model directory does not exist: {self.model_directory}")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
            
        if not 0 <= self.max_missing_ratio <= 1:
            raise ValueError("Max missing ratio must be between 0 and 1")
            
        if self.ensemble_method not in ["mean", "weighted_mean", "median"]:
            raise ValueError("Invalid ensemble method")
            
        if len(self.required_features) != 17:  # 17 features for training
            raise ValueError(f"Expected 17 required features, got {len(self.required_features)}")

# Default configuration instance
DEFAULT_CONFIG = PredictionConfig()