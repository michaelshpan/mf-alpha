"""
Command-line interface for the prediction service
"""
import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
import json
from typing import Optional, List

from .predictor import FundAlphaPredictor
from .config import PredictionConfig, DEFAULT_CONFIG

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def load_etl_data(input_path: Path) -> pd.DataFrame:
    """Load ETL data from various formats"""
    if input_path.suffix.lower() == '.parquet':
        return pd.read_parquet(input_path)
    elif input_path.suffix.lower() == '.csv':
        return pd.read_csv(input_path)
    elif input_path.suffix.lower() == '.json':
        return pd.read_json(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

def predict_command(args) -> None:
    """Execute prediction command"""
    setup_logging(args.log_level)
    log = logging.getLogger(__name__)
    
    log.info(f"Starting prediction for {args.input}")
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = PredictionConfig(**config_dict)
    else:
        config = DEFAULT_CONFIG
    
    # Override config with command line arguments
    if args.model_dir:
        config.model_directory = args.model_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.models:
        config.default_models = args.models
    if args.output_format:
        config.output_format = args.output_format
    
    try:
        # Load ETL data
        log.info("Loading ETL data...")
        etl_data = load_etl_data(Path(args.input))
        log.info(f"Loaded {len(etl_data)} samples from {args.input}")
        
        # Filter by fund if specified
        if args.fund_id:
            initial_count = len(etl_data)
            etl_data = etl_data[etl_data['class_id'] == args.fund_id]
            log.info(f"Filtered to {len(etl_data)} samples for fund {args.fund_id} (from {initial_count})")
            
            if etl_data.empty:
                log.error(f"No data found for fund ID: {args.fund_id}")
                sys.exit(1)
        
        # Initialize predictor
        log.info("Initializing predictor...")
        predictor = FundAlphaPredictor(config)
        
        # Load models
        log.info("Loading ML models...")
        predictor.load_models()
        
        # Make predictions
        log.info("Making predictions...")
        predictions = predictor.predict_batch(
            etl_data, 
            include_individual=args.include_individual,
            include_metadata=True
        )
        
        # Save results
        output_path = Path(args.output)
        predictor.save_predictions(predictions, output_path, include_report=args.include_report)
        
        # Generate summary
        report = predictor.generate_prediction_report(predictions)
        
        log.info("Prediction Summary:")
        log.info(f"  Total predictions: {report['summary']['total_predictions']}")
        log.info(f"  Unique funds: {report['summary']['unique_funds']}")
        log.info(f"  Mean alpha: {report['statistics']['ensemble_prediction']['mean']:.4f}")
        log.info(f"  Std alpha: {report['statistics']['ensemble_prediction']['std']:.4f}")
        log.info(f"  Models used: {', '.join(report['models_used'])}")
        
        if args.verbose:
            log.info(f"Full report: {json.dumps(report, indent=2, default=str)}")
        
        log.info(f"Predictions saved to: {output_path}")
        
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        if args.verbose:
            import traceback
            log.error(traceback.format_exc())
        sys.exit(1)

def info_command(args) -> None:
    """Show information about models and configuration"""
    setup_logging(args.log_level)
    log = logging.getLogger(__name__)
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = PredictionConfig(**config_dict)
    else:
        config = DEFAULT_CONFIG
    
    if args.model_dir:
        config.model_directory = args.model_dir
    
    try:
        predictor = FundAlphaPredictor(config)
        
        # Show available models
        available_models = predictor.model_loader.list_available_models()
        log.info(f"Available models in {config.model_directory}:")
        for model in available_models:
            log.info(f"  - {model}")
        
        # Show configuration
        log.info(f"\nConfiguration:")
        log.info(f"  Model directory: {config.model_directory}")
        log.info(f"  Default models: {', '.join(config.default_models)}")
        log.info(f"  Batch size: {config.batch_size}")
        log.info(f"  Output format: {config.output_format}")
        log.info(f"  Ensemble method: {config.ensemble_method}")
        log.info(f"  Required features: {len(config.required_features)}")
        
        if args.verbose:
            log.info(f"  All features: {', '.join(config.required_features)}")
            
    except Exception as e:
        log.error(f"Failed to load model info: {e}")
        sys.exit(1)

def validate_command(args) -> None:
    """Validate ETL data for prediction compatibility"""
    setup_logging(args.log_level)
    log = logging.getLogger(__name__)
    
    try:
        # Load data
        etl_data = load_etl_data(Path(args.input))
        log.info(f"Loaded {len(etl_data)} samples")
        
        # Load configuration
        if args.config:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            config = PredictionConfig(**config_dict)
        else:
            config = DEFAULT_CONFIG
        
        # Check data compatibility
        predictor = FundAlphaPredictor(config)
        
        log.info("Validating data structure...")
        predictor._validate_input_data(etl_data)
        
        log.info("Checking feature availability...")
        preprocessor = predictor.preprocessor
        processed_data = preprocessor.transform(etl_data)
        
        # Check which features are available
        feature_mapping = preprocessor._get_feature_mapping()
        available_features = []
        missing_features = []
        
        for demiguel_feature, etl_columns in feature_mapping.items():
            found = any(col in etl_data.columns for col in etl_columns)
            if found:
                available_features.append(demiguel_feature)
            else:
                missing_features.append(demiguel_feature)
        
        log.info(f"Available features ({len(available_features)}/{len(config.required_features)}):")
        for feature in available_features:
            log.info(f"  ✓ {feature}")
        
        if missing_features:
            log.warning(f"Missing features ({len(missing_features)}):")
            for feature in missing_features:
                log.warning(f"  ✗ {feature}")
        
        # Check data quality
        missing_ratio = etl_data.isnull().sum() / len(etl_data)
        high_missing = missing_ratio[missing_ratio > config.max_missing_ratio]
        
        if not high_missing.empty:
            log.warning("Columns with high missing values:")
            for col, ratio in high_missing.items():
                log.warning(f"  {col}: {ratio:.2%}")
        
        # Summary
        if len(missing_features) == 0:
            log.info("✓ Data is fully compatible for prediction")
        elif len(available_features) >= len(config.required_features) * 0.8:
            log.info("⚠ Data is mostly compatible (some features will be imputed)")
        else:
            log.error("✗ Data has too many missing features for reliable prediction")
            sys.exit(1)
            
    except Exception as e:
        log.error(f"Validation failed: {e}")
        if args.verbose:
            import traceback
            log.error(traceback.format_exc())
        sys.exit(1)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Mutual Fund Alpha Prediction Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction
  python -m prediction_service.cli predict data/etl_output.parquet predictions/alpha.parquet
  
  # Predict for specific fund
  python -m prediction_service.cli predict data/etl_output.parquet predictions/fund_alpha.parquet --fund-id FUND123
  
  # Use custom models and configuration
  python -m prediction_service.cli predict data/etl_output.parquet predictions/alpha.parquet \\
    --models gradient_boosting random_forest elastic_net --config custom_config.json
  
  # Validate data before prediction
  python -m prediction_service.cli validate data/etl_output.parquet
  
  # Show available models
  python -m prediction_service.cli info --model-dir /path/to/models
        """
    )
    
    # Global arguments
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--config', type=str, 
                       help='Path to JSON configuration file')
    parser.add_argument('--model-dir', type=str, 
                       help='Directory containing trained models')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make alpha predictions')
    predict_parser.add_argument('input', help='Input ETL data file (parquet/csv/json)')
    predict_parser.add_argument('output', help='Output predictions file')
    predict_parser.add_argument('--fund-id', type=str, 
                               help='Predict for specific fund class_id only')
    predict_parser.add_argument('--models', nargs='+', 
                               help='Models to use for prediction')
    predict_parser.add_argument('--batch-size', type=int, 
                               help='Batch size for processing')
    predict_parser.add_argument('--output-format', choices=['parquet', 'csv', 'json'], 
                               help='Output file format')
    predict_parser.add_argument('--include-individual', action='store_true', 
                               help='Include individual model predictions')
    predict_parser.add_argument('--include-report', action='store_true', default=True,
                               help='Include prediction report (default: True)')
    predict_parser.set_defaults(func=predict_command)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model and configuration info')
    info_parser.set_defaults(func=info_command)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate ETL data compatibility')
    validate_parser.add_argument('input', help='Input ETL data file to validate')
    validate_parser.set_defaults(func=validate_command)
    
    # Parse arguments and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)

if __name__ == "__main__":
    main()