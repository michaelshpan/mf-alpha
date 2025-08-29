"""
Data preprocessor for transforming ETL output to ML-ready format
"""
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging
from .config import PredictionConfig, DEFAULT_CONFIG

log = logging.getLogger(__name__)

class ETLDataPreprocessor:
    """
    Transforms ETL pipeline output into ML-ready format following DeMiguel paper preprocessing
    """
    
    def __init__(self, config: PredictionConfig = DEFAULT_CONFIG):
        self.config = config
        self.feature_stats: Optional[Dict] = None
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'ETLDataPreprocessor':
        """
        Fit the preprocessor on training data to learn statistics for scaling and normalization
        """
        log.info("Fitting preprocessor on training data")
        
        # Extract features and validate
        features_df = self._extract_features(df)
        
        # Compute statistics for scaling with proper empty slice handling
        self.feature_stats = {}
        for col in features_df.columns:
            if features_df[col].dtype in ['float64', 'int64']:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Mean of empty slice")
                    warnings.filterwarnings("ignore", message=".*empty slice.*")
                    
                    # Safe computation with fallback for empty data
                    valid_data = features_df[col].dropna()
                    if len(valid_data) > 0:
                        self.feature_stats[col] = {
                            'mean': valid_data.mean(),
                            'std': valid_data.std() if len(valid_data) > 1 else 0.0,
                            'median': valid_data.median(),
                            'q25': valid_data.quantile(0.25) if len(valid_data) > 1 else valid_data.iloc[0] if len(valid_data) == 1 else 0.0,
                            'q75': valid_data.quantile(0.75) if len(valid_data) > 1 else valid_data.iloc[0] if len(valid_data) == 1 else 0.0,
                            'min': valid_data.min(),
                            'max': valid_data.max()
                        }
                    else:
                        # Handle completely empty columns
                        self.feature_stats[col] = {
                            'mean': 0.0, 'std': 1.0, 'median': 0.0,
                            'q25': 0.0, 'q75': 0.0, 'min': 0.0, 'max': 1.0
                        }
        
        self.is_fitted = True
        log.info(f"Preprocessor fitted on {len(features_df)} samples with {len(features_df.columns)} features")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform ETL data into ML-ready format
        """
        log.info(f"Transforming {len(df)} samples")
        
        # Extract and validate features
        features_df = self._extract_features(df)
        
        # Convert object columns to numeric first
        features_df = self._convert_to_numeric(features_df)
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        # Handle outliers
        features_df = self._handle_outliers(features_df)
        
        # Scale features if enabled
        if self.config.feature_scaling:
            features_df = self._scale_features(features_df)
        
        # Validate final output
        self._validate_output(features_df)
        
        log.info(f"Transformation complete: {len(features_df)} samples, {len(features_df.columns)} features")
        return features_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step
        """
        return self.fit(df).transform(df)
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the 17 DeMiguel characteristics from ETL output
        """
        log.debug("Extracting DeMiguel characteristics")
        
        # Create feature mapping from ETL columns to DeMiguel features
        feature_mapping = self._get_feature_mapping()
        
        features_df = pd.DataFrame()
        missing_features = []
        
        for demiguel_feature, etl_columns in feature_mapping.items():
            found = False
            for etl_col in etl_columns:
                if etl_col in df.columns:
                    features_df[demiguel_feature] = df[etl_col]
                    found = True
                    break
            
            if not found:
                missing_features.append(demiguel_feature)
                features_df[demiguel_feature] = np.nan
                
        if missing_features:
            log.warning(f"Missing features will be imputed: {missing_features}")
            
        # Add metadata columns
        metadata_cols = ['class_id', 'month_end', 'cik']
        for col in metadata_cols:
            if col in df.columns:
                features_df[col] = df[col]
        
        return features_df
    
    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert object columns to numeric, handling missing values
        """
        df = df.copy()
        
        for col in df.columns:
            if col not in ['class_id', 'month_end', 'cik'] and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    
    def _get_feature_mapping(self) -> Dict[str, List[str]]:
        """
        Map DeMiguel paper features to ETL output columns
        """
        return {
            "realized alpha": ["realized alpha", "alpha_hat"],
            "realized alpha lagged": ["realized alpha lagged", "alpha_hat_lag"],
            "alpha (intercept t-stat)": ["alpha (intercept t-stat)", "alpha_t"],
            "total net assets": ["total net assets", "total_investments", "tna_proxy"],
            "expense ratio": ["expense ratio", "net_expense_ratio", "net_expense_ratio_x", "net_expense_ratio_y"],
            "age": ["age", "fund_age"],
            "flows": ["flows", "net_flow"],
            "manager_tenure": ["manager tenure", "manager_tenure"],
            "turnover ratio": ["turnover ratio", "turnover_pct", "turnover_pct_x", "turnover_pct_y"],
            "vol_of_flows": ["vol. of flows", "vol_of_flows", "flow_volatility"],
            "value_added": ["value added", "value_added"],
            "market beta t-stat": ["market beta t-stat", "market_beta_t"],
            "profit. beta t-stat": ["profit. beta t-stat", "profit_beta_t"],
            "invest. beta t-stat": ["invest. beta t-stat", "invest_beta_t"],
            "size beta t-stat": ["size beta t-stat", "size_beta_t"],
            "value beta t-stat": ["value beta t-stat", "value_beta_t"],
            "momentum beta t-stat": ["momentum beta t-stat", "momentum_beta_t"],
            "R2": ["R2", "r_squared"]
        }
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values according to configuration
        """
        log.debug(f"Handling missing values with method: {self.config.handle_missing_values}")
        
        df = df.copy()
        
        if self.config.handle_missing_values == "median_fill":
            for col in df.columns:
                if col not in ['class_id', 'month_end', 'cik'] and df[col].dtype in ['float64', 'int64']:
                    if self.is_fitted and col in self.feature_stats:
                        fill_value = self.feature_stats[col]['median']
                    else:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="Mean of empty slice")
                            fill_value = df[col].median() if len(df[col].dropna()) > 0 else 0.0
                    df[col] = df[col].fillna(fill_value)
                    
        elif self.config.handle_missing_values == "mean_fill":
            for col in df.columns:
                if col not in ['class_id', 'month_end', 'cik'] and df[col].dtype in ['float64', 'int64']:
                    if self.is_fitted and col in self.feature_stats:
                        fill_value = self.feature_stats[col]['mean']
                    else:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="Mean of empty slice")
                            fill_value = df[col].mean() if len(df[col].dropna()) > 0 else 0.0
                    df[col] = df[col].fillna(fill_value)
                    
        elif self.config.handle_missing_values == "forward_fill":
            df = df.sort_values(['class_id', 'month_end'])
            df = df.groupby('class_id').fillna(method='ffill')
            
        elif self.config.handle_missing_values == "drop":
            # Only drop rows with too many missing values
            missing_ratio = df.isnull().sum(axis=1) / len(df.columns)
            df = df[missing_ratio <= self.config.max_missing_ratio]
            
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers according to configuration
        """
        log.debug(f"Handling outliers with method: {self.config.outlier_handling}")
        
        if self.config.outlier_handling == "winsorize":
            df = df.copy()
            for col in df.columns:
                if col not in ['class_id', 'month_end', 'cik'] and df[col].dtype in ['float64', 'int64']:
                    lower, upper = self.config.winsorize_limits
                    df[col] = df[col].clip(
                        lower=df[col].quantile(lower),
                        upper=df[col].quantile(upper)
                    )
                    
        elif self.config.outlier_handling == "clip":
            df = df.copy()
            for col in df.columns:
                if col not in ['class_id', 'month_end', 'cik'] and df[col].dtype in ['float64', 'int64']:
                    q1, q3 = df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    
        elif self.config.outlier_handling == "remove":
            # Remove rows with extreme outliers (z-score > 3)
            df = df.copy()
            for col in df.columns:
                if col not in ['class_id', 'month_end', 'cik'] and df[col].dtype in ['float64', 'int64']:
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    df = df[z_scores < 3]
                    
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale features to match DeMiguel paper preprocessing
        """
        log.debug("Scaling features")
        
        df = df.copy()
        
        for col in df.columns:
            if col not in ['class_id', 'month_end', 'cik'] and df[col].dtype in ['float64', 'int64']:
                if self.is_fitted and col in self.feature_stats:
                    # Use fitted statistics
                    mean = self.feature_stats[col]['mean']
                    std = self.feature_stats[col]['std']
                else:
                    # Compute on current data with safe handling
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="Mean of empty slice")
                        valid_data = df[col].dropna()
                        if len(valid_data) > 0:
                            mean = valid_data.mean()
                            std = valid_data.std() if len(valid_data) > 1 else 1.0
                        else:
                            mean, std = 0.0, 1.0
                
                if std > 0:
                    df[col] = (df[col] - mean) / std
                    
        return df
    
    def _validate_output(self, df: pd.DataFrame) -> None:
        """
        Validate the preprocessed output
        """
        # Check for infinite values
        inf_cols = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
        if inf_cols:
            log.warning(f"Infinite values found in columns: {inf_cols}")
            
        # Check missing value ratio (only log once per session to avoid spam)
        missing_ratio = df.isnull().sum() / len(df)
        high_missing = missing_ratio[missing_ratio > self.config.max_missing_ratio]
        if not high_missing.empty and not hasattr(self, '_missing_warned'):
            log.warning(f"High missing value ratio detected in columns: {high_missing.to_dict()}")
            log.warning("Note: Missing value warnings will be suppressed for subsequent chunks")
            self._missing_warned = True
        elif not high_missing.empty:
            # Just log at debug level for subsequent warnings
            log.debug(f"High missing value ratio in columns: {high_missing.to_dict()}")
            
        # Check minimum sample size
        if len(df) < self.config.min_observations:
            log.warning(f"Low sample size: {len(df)} < {self.config.min_observations}")
            
        log.debug(f"Data validation complete: {len(df)} samples, {len(df.columns)} features")
    
    def get_feature_importance_mapping(self) -> Dict[str, str]:
        """
        Get mapping from DeMiguel features to ETL columns for interpretation
        """
        return {
            demiguel_feature: etl_columns[0] 
            for demiguel_feature, etl_columns in self._get_feature_mapping().items()
        }