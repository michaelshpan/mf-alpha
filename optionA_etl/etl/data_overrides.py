#!/usr/bin/env python3
"""
Data Override Module for Manual Data Corrections

Provides functionality to override missing expense ratios, manager tenure, and fund age
after SEC RR data integration but before dependent calculations (e.g., value_added).

Override Logic:
- Expense ratio: Applied at class_id level
- Manager tenure: Applied at series_id level (all classes in series)  
- Fund age: Applied at series_id level (all classes in series)
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np

log = logging.getLogger(__name__)

class DataOverrideProcessor:
    """Process and apply manual data overrides to fund data"""
    
    def __init__(self, override_file_path: str = "etl/data/manual_overrides.csv"):
        self.override_file_path = Path(override_file_path)
        self.overrides_df = None
        self._load_overrides()
    
    def _load_overrides(self) -> None:
        """Load override data from CSV file"""
        if not self.override_file_path.exists():
            log.info(f"No override file found at {self.override_file_path}")
            self.overrides_df = pd.DataFrame()
            return
        
        try:
            self.overrides_df = pd.read_csv(self.override_file_path)
            log.info(f"Loaded {len(self.overrides_df)} override records from {self.override_file_path}")
            
            # Validate override data structure
            self._validate_override_data()
            
        except Exception as e:
            log.error(f"Failed to load override data from {self.override_file_path}: {e}")
            self.overrides_df = pd.DataFrame()
    
    def _validate_override_data(self) -> None:
        """Validate the structure and content of override data"""
        if self.overrides_df.empty:
            return
        
        required_columns = ['identifier_type', 'identifier_value', 'field', 'override_value']
        missing_columns = set(required_columns) - set(self.overrides_df.columns)
        
        if missing_columns:
            raise ValueError(f"Override file missing required columns: {missing_columns}")
        
        # Validate identifier types
        valid_id_types = ['class_id', 'series_id']
        invalid_types = self.overrides_df[~self.overrides_df['identifier_type'].isin(valid_id_types)]
        if len(invalid_types) > 0:
            log.warning(f"Invalid identifier types found: {invalid_types['identifier_type'].unique()}")
        
        # Validate field names
        valid_fields = ['expense_ratio', 'manager_tenure', 'age']
        invalid_fields = self.overrides_df[~self.overrides_df['field'].isin(valid_fields)]
        if len(invalid_fields) > 0:
            log.warning(f"Invalid field names found: {invalid_fields['field'].unique()}")
        
        # Convert override values to numeric
        self.overrides_df['override_value'] = pd.to_numeric(self.overrides_df['override_value'], errors='coerce')
        
        # Remove rows with invalid numeric values
        before_count = len(self.overrides_df)
        self.overrides_df = self.overrides_df.dropna(subset=['override_value'])
        after_count = len(self.overrides_df)
        
        if before_count != after_count:
            log.warning(f"Removed {before_count - after_count} override records with invalid numeric values")
    
    def apply_overrides(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply manual overrides to fund data
        
        Args:
            df: DataFrame with fund data (must contain class_id, series_id columns)
            
        Returns:
            DataFrame with overrides applied
        """
        if self.overrides_df.empty:
            log.info("No override data to apply")
            return df
        
        df = df.copy()
        
        # Track override statistics
        override_stats = {
            'expense_ratio': {'applied': 0, 'skipped': 0},
            'manager_tenure': {'applied': 0, 'skipped': 0}, 
            'age': {'applied': 0, 'skipped': 0}
        }
        
        # Process each override record
        for _, override_row in self.overrides_df.iterrows():
            identifier_type = override_row['identifier_type']
            identifier_value = override_row['identifier_value']
            field = override_row['field']
            override_value = override_row['override_value']
            
            # Skip invalid records
            if pd.isna(override_value) or field not in override_stats:
                continue
            
            # Find matching records in the main dataset
            if identifier_type == 'class_id':
                mask = df['class_id'] == identifier_value
            elif identifier_type == 'series_id':
                mask = df['series_id'] == identifier_value
            else:
                log.warning(f"Unknown identifier type: {identifier_type}")
                continue
            
            matching_records = df[mask]
            if len(matching_records) == 0:
                log.debug(f"No records found for {identifier_type}={identifier_value}")
                continue
            
            # Determine target column name
            target_column = self._get_target_column(field)
            if target_column not in df.columns:
                log.warning(f"Target column {target_column} not found in dataset")
                continue
            
            # Apply override logic based on field and identifier type
            applied_count = self._apply_field_override(
                df, mask, field, target_column, override_value, identifier_type, identifier_value
            )
            
            override_stats[field]['applied'] += applied_count
        
        # Log override summary
        self._log_override_summary(override_stats)
        
        return df
    
    def _get_target_column(self, field: str) -> str:
        """Map override field names to actual dataframe column names"""
        field_mapping = {
            'expense_ratio': 'net_expense_ratio',
            'manager_tenure': 'manager_tenure',
            'age': 'fund_age'
        }
        return field_mapping.get(field, field)
    
    def _apply_field_override(
        self, df: pd.DataFrame, mask: pd.Series, field: str, target_column: str, 
        override_value: float, identifier_type: str, identifier_value: str
    ) -> int:
        """Apply override for a specific field with proper validation"""
        
        # Validate field-identifier compatibility
        if field == 'expense_ratio' and identifier_type != 'class_id':
            log.warning(f"Expense ratio override should use class_id, not {identifier_type} for {identifier_value}")
            return 0
        
        if field in ['manager_tenure', 'age'] and identifier_type != 'series_id':
            log.warning(f"{field} override should use series_id, not {identifier_type} for {identifier_value}")
            return 0
        
        # Count records that would be affected
        matching_records = df[mask]
        missing_mask = matching_records[target_column].isna()
        records_to_override = matching_records[missing_mask]
        
        if len(records_to_override) == 0:
            log.debug(f"No missing {field} values found for {identifier_type}={identifier_value}")
            return 0
        
        # Apply the override only to records with missing values
        override_mask = mask & df[target_column].isna()
        df.loc[override_mask, target_column] = override_value
        
        applied_count = override_mask.sum()
        log.debug(f"Applied {field} override ({override_value}) to {applied_count} records "
                 f"for {identifier_type}={identifier_value}")
        
        return applied_count
    
    def _log_override_summary(self, stats: Dict) -> None:
        """Log summary of overrides applied"""
        total_applied = sum(field_stats['applied'] for field_stats in stats.values())
        
        if total_applied > 0:
            log.info(f"Override Summary: Applied {total_applied} manual data overrides")
            for field, field_stats in stats.items():
                if field_stats['applied'] > 0:
                    log.info(f"  {field}: {field_stats['applied']} overrides applied")
        else:
            log.info("No manual overrides were applied (no matching missing values found)")
    
    def generate_missing_data_template(
        self, df: pd.DataFrame, output_path: str = "etl/data/manual_overrides_template.csv"
    ) -> None:
        """
        Generate a template CSV with fund identifiers that have missing data
        
        Args:
            df: DataFrame with fund data
            output_path: Path to save the template CSV
        """
        template_records = []
        
        # Check for missing expense ratios (class_id level)
        missing_expense = df[df['net_expense_ratio'].isna() & df['class_id'].notna()]
        for _, row in missing_expense.iterrows():
            template_records.append({
                'identifier_type': 'class_id',
                'identifier_value': row['class_id'],
                'field': 'expense_ratio',
                'override_value': '',  # Empty for user to fill
                'fund_name': row.get('name', ''),
                'cik': row.get('cik', ''),
                'series_id': row.get('series_id', ''),
                'notes': 'Missing expense ratio data'
            })
        
        # Check for missing manager tenure (series_id level)
        missing_manager = df[df['manager_tenure'].isna() & df['series_id'].notna()]
        unique_series_manager = missing_manager.drop_duplicates(subset=['series_id'])
        for _, row in unique_series_manager.iterrows():
            template_records.append({
                'identifier_type': 'series_id',
                'identifier_value': row['series_id'],
                'field': 'manager_tenure',
                'override_value': '',  # Empty for user to fill
                'fund_name': row.get('name', ''),
                'cik': row.get('cik', ''),
                'series_id': row.get('series_id', ''),
                'notes': 'Missing manager tenure data'
            })
        
        # Check for missing fund age (series_id level)
        missing_age = df[df['fund_age'].isna() & df['series_id'].notna()]
        unique_series_age = missing_age.drop_duplicates(subset=['series_id'])
        for _, row in unique_series_age.iterrows():
            template_records.append({
                'identifier_type': 'series_id',
                'identifier_value': row['series_id'],
                'field': 'age',
                'override_value': '',  # Empty for user to fill
                'fund_name': row.get('name', ''),
                'cik': row.get('cik', ''),
                'series_id': row.get('series_id', ''),
                'notes': 'Missing fund age data'
            })
        
        if template_records:
            template_df = pd.DataFrame(template_records)
            
            # Create output directory if needed
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save template
            template_df.to_csv(output_path, index=False)
            log.info(f"Generated override template with {len(template_df)} missing data items: {output_path}")
            
            # Log summary by field
            field_counts = template_df['field'].value_counts()
            for field, count in field_counts.items():
                log.info(f"  {field}: {count} missing values")
        else:
            log.info("No missing data found - no override template needed")


def apply_data_overrides(df: pd.DataFrame, override_file: str = "etl/data/manual_overrides.csv") -> pd.DataFrame:
    """
    Convenience function to apply data overrides to a DataFrame
    
    Args:
        df: DataFrame with fund data
        override_file: Path to override CSV file
        
    Returns:
        DataFrame with overrides applied
    """
    processor = DataOverrideProcessor(override_file)
    return processor.apply_overrides(df)


def generate_override_template(df: pd.DataFrame, template_path: str = "etl/data/manual_overrides_template.csv") -> None:
    """
    Convenience function to generate missing data template
    
    Args:
        df: DataFrame with fund data  
        template_path: Path to save template CSV
    """
    processor = DataOverrideProcessor()
    processor.generate_missing_data_template(df, template_path)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data with missing values for testing
    sample_data = pd.DataFrame({
        'class_id': ['C000001', 'C000002', 'C000003'],
        'series_id': ['S000001', 'S000001', 'S000002'], 
        'cik': ['12345', '12345', '67890'],
        'net_expense_ratio': [0.01, None, 0.02],
        'manager_tenure': [None, None, 5.0],
        'fund_age': [None, None, None],
        'name': ['Fund A Class A', 'Fund A Class B', 'Fund B Class A']
    })
    
    print("Sample data before overrides:")
    print(sample_data)
    
    # Generate template for missing data
    generate_override_template(sample_data, "sample_template.csv")