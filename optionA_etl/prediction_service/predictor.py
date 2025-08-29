"""
Main prediction orchestrator for external fund alpha prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
from datetime import datetime

from .config import PredictionConfig, DEFAULT_CONFIG
from .data_preprocessor import ETLDataPreprocessor
from .model_loader import ModelLoader, ModelEnsemble

log = logging.getLogger(__name__)

class FundAlphaPredictor:
    """
    Main orchestrator for predicting mutual fund alpha using trained ML models
    """
    
    def __init__(self, config: PredictionConfig = DEFAULT_CONFIG):
        self.config = config
        self.preprocessor = ETLDataPreprocessor(config)
        self.model_loader = ModelLoader(config)
        self.models: Optional[Dict] = None
        self.ensemble: Optional[ModelEnsemble] = None
        
    def load_models(self, model_names: Optional[List[str]] = None) -> None:
        """
        Load the specified ML models
        """
        log.info("Loading ML models for prediction")
        
        if model_names is None:
            model_names = self.config.default_models
            
        # Validate model names
        available_models = self.model_loader.list_available_models()
        invalid_models = [name for name in model_names if name not in available_models and name not in self.config.supported_models]
        
        if invalid_models:
            log.warning(f"Unknown models requested: {invalid_models}")
            
        # Load models
        self.models = self.model_loader.load_models(model_names)
        
        if not self.models:
            raise RuntimeError("No models could be loaded")
            
        # Create ensemble
        self.ensemble = ModelEnsemble(
            models=self.models,
            method=self.config.ensemble_method
        )
        
        log.info(f"Successfully loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def predict_batch(self, 
                     etl_data: pd.DataFrame, 
                     include_individual: bool = True,
                     include_metadata: bool = True) -> pd.DataFrame:
        """
        Make batch predictions on ETL data
        """
        log.info(f"Making batch predictions on {len(etl_data)} samples")
        
        if self.models is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
            
        # Validate input data
        self._validate_input_data(etl_data)
        
        # Process data in batches if large
        if len(etl_data) > self.config.batch_size:
            return self._predict_large_batch(etl_data, include_individual, include_metadata)
        else:
            return self._predict_single_batch(etl_data, include_individual, include_metadata)
    
    def predict_single_fund(self, 
                           etl_data: pd.DataFrame, 
                           class_id: str) -> Dict:
        """
        Make predictions for a single fund class
        """
        fund_data = etl_data[etl_data['class_id'] == class_id].copy()
        
        if fund_data.empty:
            raise ValueError(f"No data found for class_id: {class_id}")
            
        predictions = self.predict_batch(fund_data, include_individual=True, include_metadata=True)
        
        return {
            'class_id': class_id,
            'predictions': predictions.to_dict('records'),
            'summary': {
                'mean_alpha': predictions['ensemble_prediction'].mean(),
                'std_alpha': predictions['ensemble_prediction'].std(),
                'n_observations': len(predictions),
                'latest_prediction': predictions.iloc[-1]['ensemble_prediction'] if not predictions.empty else None
            }
        }
    
    def _predict_single_batch(self, 
                             etl_data: pd.DataFrame,
                             include_individual: bool,
                             include_metadata: bool) -> pd.DataFrame:
        """
        Make predictions on a single batch
        """
        # Preprocess data
        processed_data = self.preprocessor.transform(etl_data)
        
        # Prepare feature matrix (exclude metadata columns)
        metadata_cols = ['class_id', 'month_end', 'cik']
        feature_cols = [col for col in processed_data.columns if col not in metadata_cols]
        X = processed_data[feature_cols]
        
        # Handle missing feature columns
        missing_features = [col for col in self.config.required_features if col not in X.columns]
        if missing_features:
            log.warning(f"Missing required features, filling with zeros: {missing_features}")
            for col in missing_features:
                X[col] = 0.0
        
        # Ensure correct feature order
        X = X[self.config.required_features]
        
        # Make ensemble predictions
        ensemble_pred = self.ensemble.predict(X)
        
        # Prepare results
        results = pd.DataFrame()
        
        # Add metadata if requested
        if include_metadata:
            for col in metadata_cols:
                if col in processed_data.columns:
                    results[col] = processed_data[col].reset_index(drop=True)
        
        # Add ensemble prediction
        results['ensemble_prediction'] = ensemble_pred
        results['prediction_timestamp'] = datetime.now()
        
        # Add individual model predictions if requested
        if include_individual:
            individual_preds = self.ensemble.get_individual_predictions(X)
            for model_name, pred in individual_preds.items():
                results[f'{model_name}_prediction'] = pred
                
        # Add prediction statistics
        if include_individual and len(individual_preds) > 1:
            pred_matrix = np.column_stack(list(individual_preds.values()))
            results['prediction_std'] = np.std(pred_matrix, axis=1)
            results['prediction_min'] = np.min(pred_matrix, axis=1)
            results['prediction_max'] = np.max(pred_matrix, axis=1)
            
        return results
    
    def _predict_large_batch(self, 
                            etl_data: pd.DataFrame,
                            include_individual: bool,
                            include_metadata: bool) -> pd.DataFrame:
        """
        Handle large batches by processing in chunks
        """
        log.info(f"Processing large batch in chunks of {self.config.batch_size}")
        
        chunks = []
        for i in range(0, len(etl_data), self.config.batch_size):
            chunk = etl_data.iloc[i:i + self.config.batch_size]
            chunk_result = self._predict_single_batch(chunk, include_individual, include_metadata)
            chunks.append(chunk_result)
            
            if (i // self.config.batch_size + 1) % 10 == 0:
                log.info(f"Processed {i + len(chunk)} / {len(etl_data)} samples")
        
        return pd.concat(chunks, ignore_index=True)
    
    def _validate_input_data(self, etl_data: pd.DataFrame) -> None:
        """
        Validate ETL input data
        """
        if etl_data.empty:
            raise ValueError("Input data is empty")
            
        # Check for required metadata columns
        required_metadata = ['class_id', 'month_end']
        missing_metadata = [col for col in required_metadata if col not in etl_data.columns]
        if missing_metadata:
            raise ValueError(f"Missing required metadata columns: {missing_metadata}")
            
        # Check data quality
        if len(etl_data) < self.config.min_observations:
            log.warning(f"Low sample size: {len(etl_data)} < {self.config.min_observations}")
            
        log.debug(f"Input validation passed: {len(etl_data)} samples, {len(etl_data.columns)} columns")
    
    def generate_prediction_report(self, predictions: pd.DataFrame) -> Dict:
        """
        Generate a summary report of predictions
        """
        report = {
            'summary': {
                'total_predictions': len(predictions),
                'unique_funds': predictions['class_id'].nunique() if 'class_id' in predictions.columns else None,
                'date_range': {
                    'start': predictions['month_end'].min() if 'month_end' in predictions.columns else None,
                    'end': predictions['month_end'].max() if 'month_end' in predictions.columns else None
                },
                'prediction_timestamp': datetime.now().isoformat()
            },
            'statistics': {
                'ensemble_prediction': {
                    'mean': predictions['ensemble_prediction'].mean(),
                    'median': predictions['ensemble_prediction'].median(),
                    'std': predictions['ensemble_prediction'].std(),
                    'min': predictions['ensemble_prediction'].min(),
                    'max': predictions['ensemble_prediction'].max(),
                    'quantiles': predictions['ensemble_prediction'].quantile([0.25, 0.5, 0.75]).to_dict()
                }
            },
            'models_used': list(self.models.keys()) if self.models else [],
            'config': {
                'ensemble_method': self.config.ensemble_method,
                'batch_size': self.config.batch_size,
                'feature_scaling': self.config.feature_scaling
            }
        }
        
        # Add individual model statistics if available
        individual_cols = [col for col in predictions.columns if col.endswith('_prediction') and col != 'ensemble_prediction']
        if individual_cols:
            report['individual_models'] = {}
            for col in individual_cols:
                model_name = col.replace('_prediction', '')
                report['individual_models'][model_name] = {
                    'mean': predictions[col].mean(),
                    'std': predictions[col].std(),
                    'correlation_with_ensemble': predictions[col].corr(predictions['ensemble_prediction'])
                }
        
        return report
    
    def save_predictions(self, 
                        predictions: pd.DataFrame, 
                        output_path: Union[str, Path],
                        include_report: bool = True) -> None:
        """
        Save predictions to file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.output_format == "parquet":
            predictions.to_parquet(output_path, index=False)
        elif self.config.output_format == "csv":
            predictions.to_csv(output_path, index=False)
        elif self.config.output_format == "json":
            predictions.to_json(output_path, orient='records', date_format='iso')
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")
            
        log.info(f"Predictions saved to: {output_path}")
        
        # Save report if requested
        if include_report:
            report = self.generate_prediction_report(predictions)
            report_path = output_path.with_suffix('.report.json')
            
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            log.info(f"Prediction report saved to: {report_path}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models
        """
        if self.models is None:
            return {"status": "No models loaded"}
            
        info = {
            "loaded_models": len(self.models),
            "model_details": {
                name: self.model_loader.get_model_info(name) 
                for name in self.models.keys()
            },
            "ensemble_method": self.config.ensemble_method,
            "preprocessor_fitted": self.preprocessor.is_fitted
        }
        
        return info