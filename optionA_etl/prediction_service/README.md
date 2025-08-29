# Mutual Fund Alpha Prediction Service

A standalone service for predicting mutual fund alpha using trained ML models based on the DeMiguel et al. methodology.

## Overview

This service transforms ETL pipeline output into ML-ready format and makes predictions using pre-trained models on the 17 DeMiguel characteristics:

1. Realized alpha
2. Realized alpha lagged  
3. Alpha (intercept t-stat)
4. Total net assets
5. Expense ratio
6. Age
7. Flows
8. Manager tenure
9. Turnover ratio
10. Vol. of flows
11. Value added
12. Market beta t-stat
13. Profit. beta t-stat
14. Invest. beta t-stat
15. Size beta t-stat
16. Value beta t-stat
17. Momentum beta t-stat
18. R²

## Installation

The service is included in the ETL pipeline. No separate installation required.

## Usage

### Command Line Interface

Basic prediction:
```bash
python -m prediction_service.cli predict data/pilot_fact_class_month.parquet predictions/alpha.parquet
```

Predict for specific fund:
```bash
python -m prediction_service.cli predict data/etl_output.parquet predictions/fund_alpha.parquet --fund-id FUND123
```

Use custom models:
```bash
python -m prediction_service.cli predict data/etl_output.parquet predictions/alpha.parquet \
  --models gradient_boosting random_forest elastic_net
```

Validate data compatibility:
```bash
python -m prediction_service.cli validate data/etl_output.parquet
```

Show available models:
```bash
python -m prediction_service.cli info --model-dir ../DeMiguel\ et\ al\ replication_files/models
```

### Python API

```python
from prediction_service import FundAlphaPredictor, PredictionConfig
import pandas as pd

# Load ETL data
etl_data = pd.read_parquet("data/pilot_fact_class_month.parquet")

# Initialize predictor
predictor = FundAlphaPredictor()

# Load models
predictor.load_models(["gradient_boosting", "random_forest"])

# Make predictions
predictions = predictor.predict_batch(etl_data)

# Save results
predictor.save_predictions(predictions, "predictions/alpha.parquet")
```

## Configuration

Default configuration can be overridden using a JSON file:

```json
{
  "model_directory": "../DeMiguel et al replication_files/models",
  "default_models": ["gradient_boosting", "random_forest"],
  "batch_size": 1000,
  "ensemble_method": "mean",
  "feature_scaling": true,
  "handle_missing_values": "median_fill"
}
```

## Output

The service produces:
- Predictions file (parquet/csv/json)
- Prediction report with statistics
- Individual model predictions (optional)
- Confidence intervals and metadata

## Requirements

- pandas
- numpy
- scikit-learn
- statsmodels
- scipy
- pathlib
- logging