# Duplicate Predictions Analysis & Solutions

## Problem Summary
Multiple ensemble predictions exist for the same class_id and month_end combinations:
- **9,482 total predictions** but only **1,049 unique class_id-month_end combinations**
- **985 combinations have duplicates** (93.9% of all combinations!)
- Some combinations have up to 16 duplicate predictions with different values

## Root Cause Identified

### The Issue Chain:
1. **N-PORT Filing Overlap**: Each quarterly N-PORT filing reports 3 months of returns
   - Q3 filing reports: July, August, September
   - Q4 filing reports: October, November, December
   - **Problem**: September data appears in BOTH Q3 filing (as rtn1) and Q4 filing (as rtn3)

2. **ETL Data Duplication**: The pilot pipeline collects all these overlapping returns
   - Same month reported in multiple filings → multiple rows in ETL data
   - Different filings may have slightly different values (corrections, restatements)
   - Result: 5,694 ETL records but only 614 unique class_id-month_end combinations

3. **Prediction Service Propagation**: The prediction service processes each ETL row
   - Each duplicate ETL row gets its own prediction
   - Different return values → different feature calculations → different predictions
   - Result: Multiple predictions for same fund-month

## Example
For fund C000001973, month 2020-08-31:
- **4 different ETL records** with returns: -0.0453, 0.0815, 0.1505, 0.0437
- **4 different predictions**: -0.0152, -0.0168, -0.0184, -0.0179
- These likely come from different quarterly filings reporting the same month

## Solution Options

### Solution 1: Deduplicate at ETL Level (Recommended)
**Fix the problem at its source in the pilot pipeline**

```python
# In pilot_pipeline.py, after collecting all facts:
def deduplicate_monthly_data(facts):
    """
    Deduplicate monthly data, keeping the most recent filing's data
    """
    # Sort by class_id, month_end, and filing_date
    facts_sorted = facts.sort_values(['class_id', 'month_end', 'filing_date'])
    
    # Keep the last (most recent) filing for each class_id-month_end
    facts_deduped = facts_sorted.drop_duplicates(
        subset=['class_id', 'month_end'], 
        keep='last'  # Keep most recent filing
    )
    
    log.info(f"Deduplication: {len(facts)} → {len(facts_deduped)} records")
    log.info(f"Removed {len(facts) - len(facts_deduped)} duplicate records")
    
    return facts_deduped

# Add this after line 99 in pilot_pipeline.py:
facts = deduplicate_monthly_data(facts)
```

**Pros:**
- Fixes root cause
- Ensures data consistency throughout pipeline
- Improves alpha calculation accuracy
- Reduces data size and processing time

**Cons:**
- Need to decide which duplicate to keep (most recent? average? specific filing type?)

### Solution 2: Aggregate Duplicates at ETL Level
**Combine duplicate records using a statistical method**

```python
def aggregate_duplicate_data(facts):
    """
    Aggregate duplicate records by taking mean/median of numeric columns
    """
    # Identify numeric columns to aggregate
    numeric_cols = ['return', 'sales', 'redemptions', 'reinvest', 'total_investments']
    
    # Group by class_id and month_end
    agg_dict = {col: 'mean' for col in numeric_cols if col in facts.columns}
    
    # Keep first value for non-numeric columns
    for col in facts.columns:
        if col not in ['class_id', 'month_end'] and col not in agg_dict:
            agg_dict[col] = 'first'
    
    facts_agg = facts.groupby(['class_id', 'month_end']).agg(agg_dict).reset_index()
    
    return facts_agg
```

**Pros:**
- Preserves information from all filings
- Reduces noise through averaging
- May be more robust to reporting errors

**Cons:**
- Loses filing-specific information
- May mask data quality issues

### Solution 3: Deduplicate at Prediction Level
**Handle duplicates when making predictions**

```python
# In predictor.py, add deduplication before prediction:
def predict_batch(self, etl_data, ...):
    # Deduplicate input data
    if self.config.deduplicate_input:
        etl_data = self._deduplicate_input(etl_data)
    
    # Continue with existing prediction logic
    ...

def _deduplicate_input(self, etl_data):
    """Remove duplicate class_id-month_end combinations"""
    return etl_data.drop_duplicates(
        subset=['class_id', 'month_end'],
        keep='last'  # or 'first', or custom logic
    )
```

**Pros:**
- Quick fix without changing ETL pipeline
- Can be toggled on/off via configuration

**Cons:**
- Doesn't fix underlying data quality issue
- May produce inconsistent results if not coordinated with ETL

### Solution 4: Track Filing Source
**Keep duplicates but track their source**

```python
# Modify N-PORT parser to track filing source:
def parse_nport_primary_xml(xml_text, filing_metadata=None):
    # ... existing parsing logic ...
    
    # Add filing metadata to each row
    for row in rows:
        if filing_metadata:
            row['filing_date'] = filing_metadata.get('filing_date')
            row['filing_type'] = filing_metadata.get('filing_type')  # Q1, Q2, Q3, Q4
            row['accession_number'] = filing_metadata.get('accession')
            row['data_position'] = filing_metadata.get('position')  # rtn1, rtn2, rtn3
```

Then filter based on filing characteristics:
```python
# Keep only the primary filing for each month
def filter_primary_filings(facts):
    # Prefer rtn1 (most recent in filing) over rtn2/rtn3
    # Prefer later filing dates for same month
    ...
```

**Pros:**
- Maintains audit trail
- Allows for sophisticated filtering rules
- Can detect and analyze discrepancies

**Cons:**
- More complex implementation
- Requires schema changes

## Recommended Implementation Plan

### Immediate Fix (1 hour)
Add deduplication to the pilot pipeline (Solution 1):
```python
# In pilot_pipeline.py, after line 99:
facts = facts.sort_values(['class_id', 'month_end', 'filing_date'])
facts = facts.drop_duplicates(subset=['class_id', 'month_end'], keep='last')
log.info(f"Deduplication removed {orig_len - len(facts)} duplicate records")
```

### Proper Fix (1 day)
1. Modify N-PORT parser to track filing metadata
2. Implement intelligent deduplication rules:
   - Prefer corrections/amendments over original filings
   - Prefer rtn1 (current month) over rtn2/rtn3 (historical)
   - Log discrepancies for data quality monitoring

### Long-term Solution (1 week)
1. Create a data quality module that:
   - Detects and logs discrepancies
   - Validates return consistency
   - Implements configurable deduplication strategies
2. Add data lineage tracking
3. Create reconciliation reports

## Testing After Fix

```python
# Verify deduplication worked
import pandas as pd

# Check ETL data
etl = pd.read_parquet('data/pilot_fact_class_month.parquet')
print(f"ETL records: {len(etl)}")
print(f"Unique combinations: {len(etl[['class_id', 'month_end']].drop_duplicates())}")
assert len(etl) == len(etl[['class_id', 'month_end']].drop_duplicates()), "Still have duplicates!"

# Check predictions
pred = pd.read_parquet('predictions/alpha.parquet')
print(f"Predictions: {len(pred)}")
print(f"Unique combinations: {len(pred[['class_id', 'month_end']].drop_duplicates())}")
assert len(pred) == len(pred[['class_id', 'month_end']].drop_duplicates()), "Still have duplicates!"

print("✓ No duplicates found!")
```

## Impact on Analysis
After fixing duplicates:
- Predictions will be deterministic and consistent
- Alpha calculations will use consistent return data
- Comparison between predicted and actual alpha will be more accurate
- Model training (if using this data) will be more stable

## Quick Command to Fix Existing Data

```python
# Fix existing prediction file
import pandas as pd

# Load and deduplicate
df = pd.read_parquet('optionA_etl/predictions/alpha.parquet')
df_dedup = df.sort_values(['class_id', 'month_end', 'prediction_timestamp'])
df_dedup = df_dedup.drop_duplicates(subset=['class_id', 'month_end'], keep='last')

# Save cleaned version
df_dedup.to_parquet('optionA_etl/predictions/alpha_cleaned.parquet', index=False)
print(f"Reduced from {len(df)} to {len(df_dedup)} records")
```