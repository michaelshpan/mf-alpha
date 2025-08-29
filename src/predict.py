import argparse
import logging
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from utils import setup_logging, load_config
from portfolio import weights_in_top_funds

logger = logging.getLogger(__name__)

REQUIRED_FEATURES = [
    "alpha6_1mo_12m_lag_1", "alpha6_tstat_lag_1", "mtna_lag_1", "exp_ratio_lag_1",
    "age_lag_1", "flow_12m_lag_1", "mgr_tenure_lag_1", "turn_ratio_lag_1", 
    "flow_vol_12m_lag_1", "value_added_12m_lag_1", "beta_market_tstat_lag_1", 
    "beta_profit_tstat_lag_1", "beta_invest_tstat_lag_1", "beta_size_tstat_lag_1", 
    "beta_value_tstat_lag_1", "beta_mom_tstat_lag_1", "R2_lag_1"
]

def load_models(models_dir: Path) -> Dict[str, Any]:
    """Load all trained models from the models directory."""
    models = {}
    model_files = list(models_dir.glob("*_model.pkl"))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    for model_file in model_files:
        model_name = model_file.stem.replace("_model", "")
        with open(model_file, 'rb') as f:
            model_info = pickle.load(f)
        models[model_name] = model_info
        logger.info("Loaded %s model (trained on %d samples, formation date: %s)", 
                   model_name, model_info['train_size'], model_info['formation_date'])
    
    return models

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare input data for prediction."""
    # Check required columns
    missing_cols = set(REQUIRED_FEATURES) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for missing values
    missing_data = df[REQUIRED_FEATURES].isnull().sum()
    if missing_data.any():
        logger.warning("Found missing values:\n%s", missing_data[missing_data > 0])
        logger.warning("Missing values will be filled with column medians")
        df[REQUIRED_FEATURES] = df[REQUIRED_FEATURES].fillna(df[REQUIRED_FEATURES].median())
    
    # Ensure we have fund identifiers
    if 'fundno' not in df.columns and 'fund_id' not in df.columns:
        df['fundno'] = [f"fund_{i}" for i in range(len(df))]
        logger.warning("No fund identifier found, generated sequential IDs")
    
    fund_id_col = 'fundno' if 'fundno' in df.columns else 'fund_id'
    
    return df, fund_id_col

def predict_alpha(models: Dict[str, Any], input_data: pd.DataFrame, 
                 fund_id_col: str, top_cut: float = 0.90) -> pd.DataFrame:
    """Generate alpha predictions for input funds."""
    results = []
    
    # Prepare feature matrix
    X = input_data[REQUIRED_FEATURES].values
    fund_ids = input_data[fund_id_col].values
    
    for model_name, model_info in models.items():
        model = model_info['model']
        
        # Generate predictions
        alpha_pred = model.predict(X)
        
        # Calculate portfolio weights (top decile)
        weights = weights_in_top_funds(alpha_pred, top_cut)
        
        # Create results dataframe for this model
        model_results = pd.DataFrame({
            'fund_id': fund_ids,
            'predicted_alpha': alpha_pred,
            'portfolio_weight': weights,
            'model': model_name,
            'in_top_decile': weights > 0
        })
        
        results.append(model_results)
    
    # Combine all model results
    all_results = pd.concat(results, ignore_index=True)
    
    # Create summary by fund
    summary = all_results.pivot(index='fund_id', columns='model', 
                               values=['predicted_alpha', 'portfolio_weight', 'in_top_decile'])
    
    # Flatten column names
    summary.columns = [f'{col[1]}_{col[0]}' for col in summary.columns]
    summary = summary.reset_index()
    
    # Add ensemble predictions
    alpha_cols = [col for col in summary.columns if col.endswith('_predicted_alpha')]
    summary['ensemble_predicted_alpha'] = summary[alpha_cols].mean(axis=1)
    
    # Count how many models put fund in top decile
    top_decile_cols = [col for col in summary.columns if col.endswith('_in_top_decile')]
    summary['models_top_decile_count'] = summary[top_decile_cols].sum(axis=1)
    summary['consensus_top_decile'] = summary['models_top_decile_count'] >= len(models) / 2
    
    return all_results, summary

def parse_args():
    parser = argparse.ArgumentParser(description="Predict mutual fund alpha using trained ML models")
    parser.add_argument("--input", required=True, help="Input CSV file with fund characteristics")
    parser.add_argument("--models_dir", default="outputs/models", help="Directory containing trained models")
    parser.add_argument("--output", default="outputs/predictions.csv", help="Output file for predictions")
    parser.add_argument("--config", default="src/config.yaml", help="Configuration file")
    parser.add_argument("--top_cut", type=float, default=0.90, help="Top percentile for portfolio selection")
    parser.add_argument("--summary_only", action="store_true", help="Only output summary (one row per fund)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    cfg = load_config(Path(args.config))
    setup_logging(cfg["logging"]["level"])
    
    # Load models
    models_dir = Path(args.models_dir)
    models = load_models(models_dir)
    logger.info("Loaded %d models: %s", len(models), list(models.keys()))
    
    # Load and validate input data
    logger.info("Loading input data from %s", args.input)
    input_df = pd.read_csv(args.input)
    logger.info("Loaded %d funds for prediction", len(input_df))
    
    input_df, fund_id_col = validate_input_data(input_df)
    
    # Generate predictions
    logger.info("Generating predictions...")
    all_results, summary = predict_alpha(models, input_df, fund_id_col, args.top_cut)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.summary_only:
        summary.to_csv(output_path, index=False)
        logger.info("Saved summary predictions to %s", output_path)
    else:
        all_results.to_csv(output_path, index=False)
        summary_path = output_path.parent / f"{output_path.stem}_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Saved detailed predictions to %s", output_path)
        logger.info("Saved summary predictions to %s", summary_path)
    
    # Print summary statistics
    print("\n=== PREDICTION SUMMARY ===")
    print(f"Total funds analyzed: {len(summary)}")
    print(f"Top {100*(1-args.top_cut):.0f}% funds by consensus: {summary['consensus_top_decile'].sum()}")
    print("\nTop 5 funds by ensemble alpha prediction:")
    top_funds = summary.nlargest(5, 'ensemble_predicted_alpha')[['fund_id', 'ensemble_predicted_alpha', 'models_top_decile_count']]
    print(top_funds.to_string(index=False))

if __name__ == "__main__":
    main()