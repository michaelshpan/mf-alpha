#!/usr/bin/env python3
"""
SEC Risk/Return (RR) data integration module.
Loads structured TSV data from SEC quarterly datasets and provides expense ratios 
and turnover rates that can be joined with N-PORT data.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from .series_class_mapper import SeriesClassMapper, integrate_turnover_with_series_mapping

log = logging.getLogger(__name__)

class SECRRDataLoader:
    """Load and process SEC Risk/Return quarterly datasets"""
    
    def __init__(self, base_dir: str = "sec_rr_datasets", use_series_mapping: bool = True):
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise ValueError(f"SEC RR data directory {base_dir} does not exist")
        
        # Find available quarters
        self.available_quarters = self._find_quarters()
        log.info(f"Found {len(self.available_quarters)} quarters of SEC RR data")
        
        # Initialize series/class mapper if requested
        self.mapper = None
        if use_series_mapping:
            try:
                self.mapper = SeriesClassMapper(cache_path="data/series_class_mapping.csv")
                stats = self.mapper.get_mapping_stats()
                log.info(f"Loaded series/class mapping: {stats.get('unique_series', 0)} series, "
                        f"{stats.get('unique_classes', 0)} classes")
            except Exception as e:
                log.warning(f"Could not load series/class mapping: {e}")
    
    def _find_quarters(self) -> List[str]:
        """Find all extracted quarterly datasets"""
        quarters = []
        for dir_path in sorted(self.base_dir.glob("*q*_mfrr")):
            if dir_path.is_dir():
                # Check if contains expected files
                if (dir_path / "sub.tsv").exists() and (dir_path / "num.tsv").exists():
                    quarters.append(dir_path.name)
        return sorted(quarters, reverse=True)  # Most recent first
    
    def load_quarter(self, quarter: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load submission and numeric data for a specific quarter"""
        quarter_dir = self.base_dir / quarter
        
        if not quarter_dir.exists():
            raise ValueError(f"Quarter {quarter} not found in {self.base_dir}")
        
        # Load TSV files with proper dtypes
        sub_df = pd.read_csv(
            quarter_dir / "sub.tsv", 
            sep="\t", 
            dtype={'cik': str, 'adsh': str},
            low_memory=False
        )
        
        num_df = pd.read_csv(
            quarter_dir / "num.tsv", 
            sep="\t",
            dtype={'adsh': str, 'tag': str, 'series': str, 'class': str},
            low_memory=False
        )
        
        return sub_df, num_df
    
    def extract_expense_turnover(self, 
                                  ciks: Optional[List[str]] = None,
                                  quarters: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract expense ratios and turnover rates for specified CIKs and quarters.
        
        Args:
            ciks: List of CIK strings to filter (None = all)
            quarters: List of quarters to process (None = most recent)
        
        Returns:
            DataFrame with columns: cik, class_id, expense_ratio, turnover_rate, quarter, filing_date
        """
        if quarters is None:
            quarters = self.available_quarters[:4]  # Last year of data
        
        all_results = []
        
        for quarter in quarters:
            if quarter not in self.available_quarters:
                log.warning(f"Quarter {quarter} not available, skipping")
                continue
            
            log.info(f"Processing quarter {quarter}")
            sub_df, num_df = self.load_quarter(quarter)
            
            # Filter by CIKs if specified
            if ciks:
                # Try multiple CIK formats - unpadded and padded
                cik_variants = []
                for cik in ciks:
                    cik_str = str(cik)
                    cik_variants.append(cik_str)  # Original
                    cik_variants.append(cik_str.lstrip('0'))  # Remove leading zeros
                    cik_variants.append(cik_str.zfill(10))  # Pad to 10
                sub_df = sub_df[sub_df['cik'].isin(cik_variants)]
            
            if len(sub_df) == 0:
                log.warning(f"No matching CIKs found in {quarter}")
                continue
            
            # Get relevant numeric data
            expense_tags = ['NetExpensesOverAssets', 'ExpensesOverAssets']
            turnover_tags = ['PortfolioTurnoverRate', 'PortfolioTurnoverPercent']
            
            # Filter numeric data to relevant submissions
            relevant_adsh = sub_df['adsh'].unique()
            num_filtered = num_df[num_df['adsh'].isin(relevant_adsh)]
            
            # Extract expense ratios
            expense_data = num_filtered[num_filtered['tag'].isin(expense_tags)].copy()
            expense_data['metric_type'] = 'expense_ratio'
            
            # Extract turnover rates
            turnover_data = num_filtered[num_filtered['tag'].isin(turnover_tags)].copy()
            
            # If we have a mapper and turnover is at series level, map to class level
            if self.mapper and len(turnover_data) > 0:
                # Check if turnover is at series level (class is NaN)
                series_level_turnover = turnover_data[turnover_data['class'].isna() & turnover_data['series'].notna()]
                if len(series_level_turnover) > 0:
                    log.info(f"Mapping {len(series_level_turnover)} series-level turnover records to class level")
                    mapped_turnover = integrate_turnover_with_series_mapping(num_filtered, self.mapper)
                    if not mapped_turnover.empty:
                        # Replace series-level with class-level turnover
                        turnover_data = mapped_turnover[mapped_turnover['tag'].isin(turnover_tags)].copy()
            
            turnover_data['metric_type'] = 'turnover_rate'
            
            # Combine data
            combined = pd.concat([expense_data, turnover_data], ignore_index=True)
            
            if len(combined) == 0:
                log.warning(f"No expense/turnover data found in {quarter}")
                continue
            
            # Merge with submission info
            combined = combined.merge(
                sub_df[['adsh', 'cik', 'name', 'filed']], 
                on='adsh', 
                how='left'
            )
            
            # Add quarter identifier
            combined['quarter'] = quarter
            
            all_results.append(combined)
        
        if not all_results:
            log.warning("No data extracted from any quarter")
            return pd.DataFrame()
        
        # Combine all quarters
        result_df = pd.concat(all_results, ignore_index=True)
        
        # Process and pivot data
        result_df = self._process_metrics(result_df)
        
        return result_df
    
    def _process_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and restructure metrics data with proper ADSH priority handling"""
        
        # Ensure value column exists and is numeric
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        else:
            log.warning("No 'value' column found, metrics may be incomplete")
            df['value'] = np.nan
        
        # Extract last 6 digits from ADSH for priority sorting
        df['adsh_priority'] = df['adsh'].str[-6:].astype(int, errors='ignore')
        
        processed = []
        
        # Process expense ratios (class_id level)
        expense_data = df[df['metric_type'] == 'expense_ratio'].copy()
        if not expense_data.empty:
            for (cik, class_id), group in expense_data.groupby(['cik', 'class']):
                if pd.notna(class_id):  # Only process records with valid class_id
                    # Sort by ADSH priority (highest last 6 digits first)
                    group = group.sort_values(['adsh_priority', 'filed'], ascending=[False, False])
                    
                    # Prefer NetExpensesOverAssets over other expense tags
                    net_expense = group[group['tag'] == 'NetExpensesOverAssets']
                    if len(net_expense) > 0:
                        best_record = net_expense.iloc[0]
                    else:
                        best_record = group.iloc[0]
                    
                    processed.append({
                        'cik': str(cik).lstrip('0') or '0',
                        'class_id': class_id,
                        'series_id': best_record['series'] if pd.notna(best_record['series']) else None,
                        'expense_ratio': best_record['value'],
                        'turnover_rate': None,  # Will be filled in series processing
                        'filing_date': best_record['filed'] if 'filed' in best_record else None,
                        'quarter': best_record['quarter'] if 'quarter' in best_record else None,
                    })
        
        # Process turnover rates (series_id level)  
        turnover_data = df[df['metric_type'] == 'turnover_rate'].copy()
        series_turnover = {}
        
        if not turnover_data.empty:
            for (cik, series_id), group in turnover_data.groupby(['cik', 'series']):
                if pd.notna(series_id):  # Only process records with valid series_id
                    # Sort by ADSH priority (highest last 6 digits first)
                    group = group.sort_values(['adsh_priority', 'filed'], ascending=[False, False])
                    best_record = group.iloc[0]
                    
                    turnover_rate = best_record['value']
                    # Convert percentage to decimal if needed
                    if pd.notna(turnover_rate) and turnover_rate > 10:
                        turnover_rate = turnover_rate / 100.0
                    
                    series_turnover[(str(cik).lstrip('0') or '0', series_id)] = {
                        'turnover_rate': turnover_rate,
                        'filing_date': best_record['filed'] if 'filed' in best_record else None,
                        'quarter': best_record['quarter'] if 'quarter' in best_record else None,
                    }
        
        # Create class-level records, filling in turnover from series level
        class_records = {}
        for record in processed:
            key = (record['cik'], record['class_id'])
            class_records[key] = record
        
        # Add turnover data to class records based on series_id
        for record in processed:
            if record['series_id'] and (record['cik'], record['series_id']) in series_turnover:
                turnover_info = series_turnover[(record['cik'], record['series_id'])]
                record['turnover_rate'] = turnover_info['turnover_rate']
        
        # Add series-only turnover records for series without expense ratio data
        for (cik, series_id), turnover_info in series_turnover.items():
            # Check if we already have class records for this series
            series_has_class_records = any(r['cik'] == cik and r['series_id'] == series_id for r in processed)
            
            if not series_has_class_records:
                # Add a series-level record
                processed.append({
                    'cik': cik,
                    'class_id': None,
                    'series_id': series_id,
                    'expense_ratio': None,
                    'turnover_rate': turnover_info['turnover_rate'],
                    'filing_date': turnover_info['filing_date'],
                    'quarter': turnover_info['quarter'],
                })
        
        result = pd.DataFrame(processed)
        
        # Log summary statistics
        log.info(f"Processed {len(result)} records")
        log.info(f"Records with expense ratio: {result['expense_ratio'].notna().sum()}")
        log.info(f"Records with turnover rate: {result['turnover_rate'].notna().sum()}")
        
        return result
    
    def merge_with_nport_data(self, nport_df: pd.DataFrame, rr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge SEC RR data with N-PORT data on CIK and class_id.
        
        Args:
            nport_df: DataFrame with N-PORT data (must have 'cik' and 'class_id' columns)
            rr_df: DataFrame from extract_expense_turnover()
        
        Returns:
            Merged DataFrame with expense ratios and turnover rates added
        """
        # Ensure CIK format matches - use unpadded format
        if 'cik' in nport_df.columns:
            nport_df['cik_unpadded'] = nport_df['cik'].astype(str).str.lstrip('0')
            if nport_df['cik_unpadded'].str.len().eq(0).any():
                nport_df.loc[nport_df['cik_unpadded'].str.len().eq(0), 'cik_unpadded'] = '0'
        else:
            raise ValueError("N-PORT data must have 'cik' column")
        
        if 'cik' in rr_df.columns:
            rr_df['cik_unpadded'] = rr_df['cik'].astype(str).str.lstrip('0')
            if rr_df['cik_unpadded'].str.len().eq(0).any():
                rr_df.loc[rr_df['cik_unpadded'].str.len().eq(0), 'cik_unpadded'] = '0'
        
        # First try to merge on both CIK and class_id
        merged = nport_df.merge(
            rr_df[['cik_unpadded', 'class_id', 'expense_ratio', 'turnover_rate', 'filing_date']],
            left_on=['cik_unpadded', 'class_id'],
            right_on=['cik_unpadded', 'class_id'],
            how='left',
            suffixes=('', '_rr')
        )
        
        # For records without class match, try fund-level data (class_id = None)
        missing_mask = merged['expense_ratio'].isna()
        if missing_mask.sum() > 0:
            log.info(f"Attempting fund-level match for {missing_mask.sum()} records without class match")
            
            fund_level = rr_df[rr_df['class_id'].isna()]
            if len(fund_level) > 0:
                # Need to ensure cik_unpadded column exists in the subset
                nport_subset = nport_df[missing_mask].copy()
                nport_subset['cik_unpadded'] = nport_subset['cik'].astype(str).str.lstrip('0')
                
                fund_merge = nport_subset.merge(
                    fund_level[['cik_unpadded', 'expense_ratio', 'turnover_rate', 'filing_date']],
                    on='cik_unpadded',
                    how='left',
                    suffixes=('', '_fund')
                )
                
                # Update missing values with fund-level data if merge was successful
                if 'expense_ratio_fund' in fund_merge.columns:
                    merged.loc[missing_mask, 'expense_ratio'] = fund_merge['expense_ratio_fund'].values
                    merged.loc[missing_mask, 'turnover_rate'] = fund_merge['turnover_rate_fund'].values
                    merged.loc[missing_mask, 'filing_date'] = fund_merge['filing_date_fund'].values
                else:
                    # No suffix needed, direct columns
                    if 'expense_ratio' in fund_merge.columns:
                        expense_vals = fund_merge['expense_ratio'].values
                        turnover_vals = fund_merge['turnover_rate'].values
                        filing_vals = fund_merge['filing_date'].values
                        
                        # Get the indices to update
                        indices = merged.index[missing_mask]
                        for i, idx in enumerate(indices):
                            if i < len(expense_vals):
                                merged.loc[idx, 'expense_ratio'] = expense_vals[i]
                                merged.loc[idx, 'turnover_rate'] = turnover_vals[i]
                                merged.loc[idx, 'filing_date'] = filing_vals[i]
        
        # Clean up temporary columns
        merged = merged.drop(columns=['cik_unpadded'], errors='ignore')
        
        # Log merge statistics
        log.info(f"Merge complete: {len(merged)} total records")
        log.info(f"Records with expense ratio: {merged['expense_ratio'].notna().sum()}")
        log.info(f"Records with turnover rate: {merged['turnover_rate'].notna().sum()}")
        
        return merged


def integrate_sec_rr_data(pilot_data_path: str = "data/pilot_fact_class_month.parquet",
                          output_path: str = "data/pilot_with_sec_rr.parquet") -> pd.DataFrame:
    """
    Main integration function to add SEC RR expense/turnover data to pilot pipeline output.
    
    Args:
        pilot_data_path: Path to existing pilot data
        output_path: Path to save enhanced data
    
    Returns:
        Enhanced DataFrame with SEC RR data
    """
    # Load existing pilot data
    log.info(f"Loading pilot data from {pilot_data_path}")
    pilot_df = pd.read_parquet(pilot_data_path)
    
    # Get unique CIKs from pilot data
    unique_ciks = pilot_df['cik'].unique().tolist()
    log.info(f"Found {len(unique_ciks)} unique CIKs in pilot data")
    
    # Initialize SEC RR loader
    loader = SECRRDataLoader()
    
    # Extract expense and turnover data for these CIKs
    log.info("Extracting expense and turnover data from SEC RR datasets")
    rr_df = loader.extract_expense_turnover(ciks=unique_ciks)
    
    if len(rr_df) == 0:
        log.warning("No SEC RR data extracted, returning original data")
        return pilot_df
    
    # Merge with pilot data
    log.info("Merging SEC RR data with pilot data")
    enhanced_df = loader.merge_with_nport_data(pilot_df, rr_df)
    
    # Save enhanced data
    enhanced_df.to_parquet(output_path, index=False)
    log.info(f"Saved enhanced data to {output_path}")
    
    # Print summary statistics
    print("\n=== Integration Summary ===")
    print(f"Original records: {len(pilot_df)}")
    print(f"Enhanced records: {len(enhanced_df)}")
    print(f"Records with expense ratio: {enhanced_df['expense_ratio'].notna().sum()} "
          f"({enhanced_df['expense_ratio'].notna().sum() / len(enhanced_df) * 100:.1f}%)")
    print(f"Records with turnover rate: {enhanced_df['turnover_rate'].notna().sum()} "
          f"({enhanced_df['turnover_rate'].notna().sum() / len(enhanced_df) * 100:.1f}%)")
    
    return enhanced_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    integrate_sec_rr_data()