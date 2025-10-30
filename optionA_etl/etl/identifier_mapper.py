#!/usr/bin/env python3
"""
Identifier Mapping Module for Mutual Fund Data ETL

This module creates and maintains a cached database of relationships between:
- CIK (Central Index Key)
- Series ID 
- Class ID
- Mutual Fund Ticker

The cache can be updated independently and supports fund selection by series_id,
class_id, or ticker symbol.
"""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import pandas as pd
import requests

from .series_class_mapper import fetch_series_class_mapping, sec_headers

log = logging.getLogger(__name__)


class IdentifierMapper:
    """
    Manages cached mapping between mutual fund identifiers.
    
    Features:
    - Automatic cache creation and refresh
    - Fund selection by series_id (includes all class_ids), class_id, or ticker
    - Reverse lookups (ticker -> CIK/series/class)
    - Cache persistence with configurable TTL
    """
    
    def __init__(self, cache_dir: str = None, cache_ttl_days: int = 7):
        """
        Initialize the identifier mapper.
        
        Args:
            cache_dir: Directory for cache storage (default: ./cache)
            cache_ttl_days: Days before cache refresh (default: 7)
        """
        self.cache_dir = Path(cache_dir or "./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / "identifier_mapping.pkl"
        self.metadata_file = self.cache_dir / "identifier_metadata.json"
        self.cache_ttl = timedelta(days=cache_ttl_days)
        
        # Internal mapping data
        self.mapping_df: Optional[pd.DataFrame] = None
        self._cik_to_series: Dict[str, Set[str]] = {}
        self._series_to_classes: Dict[str, Set[str]] = {}
        self._class_to_info: Dict[str, Dict] = {}
        self._ticker_to_class: Dict[str, str] = {}
        
        # Load cache or build if needed
        self._load_or_build_cache()
    
    def _load_or_build_cache(self) -> None:
        """Load cache from disk or build if missing/expired."""
        if self._is_cache_valid():
            log.info("Loading identifier mapping from cache")
            self._load_cache()
        else:
            log.info("Building new identifier mapping cache")
            self.update_cache()
    
    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is not expired."""
        if not self.cache_file.exists() or not self.metadata_file.exists():
            return False
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                cached_time = datetime.fromisoformat(metadata['timestamp'])
                if datetime.now() - cached_time > self.cache_ttl:
                    log.info(f"Cache expired (older than {self.cache_ttl_days} days)")
                    return False
                return True
        except Exception as e:
            log.warning(f"Error checking cache validity: {e}")
            return False
    
    def _load_cache(self) -> None:
        """Load cached mapping data from disk."""
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            self.mapping_df = cache_data['mapping_df']
            self._cik_to_series = cache_data['cik_to_series']
            self._series_to_classes = cache_data['series_to_classes']
            self._class_to_info = cache_data['class_to_info']
            self._ticker_to_class = cache_data['ticker_to_class']
            
            log.info(f"Loaded cache with {len(self.mapping_df)} records")
            
        except Exception as e:
            log.error(f"Failed to load cache: {e}")
            self.update_cache()
    
    def update_cache(self, force: bool = False) -> None:
        """
        Update the identifier mapping cache.
        
        Args:
            force: Force update even if cache is valid
        """
        if not force and self._is_cache_valid():
            log.info("Cache is still valid, skipping update")
            return
        
        log.info("Fetching fresh identifier mapping data from SEC")
        
        try:
            # Fetch mapping data from SEC
            self.mapping_df = fetch_series_class_mapping()
            
            if self.mapping_df.empty:
                log.error("Failed to fetch identifier mapping data")
                return
            
            # Build internal lookup structures
            self._build_lookups()
            
            # Save to cache
            self._save_cache()
            
            log.info(f"Cache updated with {len(self.mapping_df)} records")
            
        except Exception as e:
            log.error(f"Failed to update cache: {e}")
            raise
    
    def _build_lookups(self) -> None:
        """Build internal lookup structures for efficient querying."""
        self._cik_to_series = {}
        self._series_to_classes = {}
        self._class_to_info = {}
        self._ticker_to_class = {}
        
        for _, row in self.mapping_df.iterrows():
            cik = str(row['cik']) if pd.notna(row['cik']) else None
            series_id = str(row['series_id']) if pd.notna(row['series_id']) else None
            class_id = str(row['class_id']) if pd.notna(row['class_id']) else None
            ticker = str(row['ticker']) if pd.notna(row.get('ticker')) else None
            
            if cik and series_id:
                if cik not in self._cik_to_series:
                    self._cik_to_series[cik] = set()
                self._cik_to_series[cik].add(series_id)
            
            if series_id and class_id:
                if series_id not in self._series_to_classes:
                    self._series_to_classes[series_id] = set()
                self._series_to_classes[series_id].add(class_id)
            
            if class_id:
                self._class_to_info[class_id] = {
                    'cik': cik,
                    'series_id': series_id,
                    'ticker': ticker,
                    'series_name': row.get('series_name'),
                    'class_name': row.get('class_name')
                }
            
            if ticker and class_id:
                self._ticker_to_class[ticker.upper()] = class_id
        
        log.info(f"Built lookups: {len(self._cik_to_series)} CIKs, "
                f"{len(self._series_to_classes)} series, "
                f"{len(self._class_to_info)} classes, "
                f"{len(self._ticker_to_class)} tickers")
    
    def _save_cache(self) -> None:
        """Save mapping data to cache files."""
        cache_data = {
            'mapping_df': self.mapping_df,
            'cik_to_series': self._cik_to_series,
            'series_to_classes': self._series_to_classes,
            'class_to_info': self._class_to_info,
            'ticker_to_class': self._ticker_to_class
        }
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'record_count': len(self.mapping_df),
            'cik_count': len(self._cik_to_series),
            'series_count': len(self._series_to_classes),
            'class_count': len(self._class_to_info),
            'ticker_count': len(self._ticker_to_class)
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log.info(f"Cache saved to {self.cache_dir}")
    
    # Query methods
    
    def get_classes_for_series(self, series_id: str) -> List[str]:
        """
        Get all class IDs for a given series ID.
        
        Args:
            series_id: Series ID (e.g., 'S000002594')
            
        Returns:
            List of class IDs
        """
        series_id = series_id.upper()
        return list(self._series_to_classes.get(series_id, []))
    
    def get_series_for_cik(self, cik: str) -> List[str]:
        """
        Get all series IDs for a given CIK.
        
        Args:
            cik: Central Index Key
            
        Returns:
            List of series IDs
        """
        # Normalize CIK (remove leading zeros)
        cik = str(int(cik))
        return list(self._cik_to_series.get(cik, []))
    
    def get_info_for_class(self, class_id: str) -> Optional[Dict]:
        """
        Get complete information for a class ID.
        
        Args:
            class_id: Class ID (e.g., 'C000007113')
            
        Returns:
            Dictionary with cik, series_id, ticker, names
        """
        class_id = class_id.upper()
        return self._class_to_info.get(class_id)
    
    def get_class_for_ticker(self, ticker: str) -> Optional[str]:
        """
        Get class ID for a ticker symbol.
        
        Args:
            ticker: Mutual fund ticker (e.g., 'VHCOX')
            
        Returns:
            Class ID or None
        """
        ticker = ticker.upper()
        return self._ticker_to_class.get(ticker)
    
    def get_info_for_ticker(self, ticker: str) -> Optional[Dict]:
        """
        Get complete information for a ticker symbol.
        
        Args:
            ticker: Mutual fund ticker
            
        Returns:
            Dictionary with class_id, cik, series_id, names
        """
        class_id = self.get_class_for_ticker(ticker)
        if class_id:
            return self.get_info_for_class(class_id)
        return None
    
    def select_funds(
        self,
        series_ids: List[str] = None,
        class_ids: List[str] = None,
        tickers: List[str] = None,
        ciks: List[str] = None
    ) -> pd.DataFrame:
        """
        Select funds based on various identifier criteria.
        
        Args:
            series_ids: List of series IDs (will include all underlying classes)
            class_ids: List of specific class IDs
            tickers: List of ticker symbols
            ciks: List of CIKs (will include all underlying series and classes)
            
        Returns:
            DataFrame with selected fund information
        """
        selected_classes = set()
        
        # Add classes from series IDs
        if series_ids:
            for series_id in series_ids:
                classes = self.get_classes_for_series(series_id)
                selected_classes.update(classes)
                log.info(f"Series {series_id}: found {len(classes)} classes")
        
        # Add specific class IDs
        if class_ids:
            for class_id in class_ids:
                class_id = class_id.upper()
                if class_id in self._class_to_info:
                    selected_classes.add(class_id)
        
        # Add classes from tickers
        if tickers:
            for ticker in tickers:
                class_id = self.get_class_for_ticker(ticker)
                if class_id:
                    selected_classes.add(class_id)
        
        # Add classes from CIKs
        if ciks:
            for cik in ciks:
                series_list = self.get_series_for_cik(cik)
                for series_id in series_list:
                    classes = self.get_classes_for_series(series_id)
                    selected_classes.update(classes)
        
        # Build result DataFrame
        results = []
        for class_id in selected_classes:
            info = self.get_info_for_class(class_id)
            if info:
                results.append({
                    'class_id': class_id,
                    'series_id': info['series_id'],
                    'cik': info['cik'],
                    'ticker': info['ticker'],
                    'series_name': info.get('series_name'),
                    'class_name': info.get('class_name')
                })
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(['cik', 'series_id', 'class_id'])
        
        log.info(f"Selected {len(df)} funds based on criteria")
        return df
    
    def get_stats(self) -> Dict:
        """Get statistics about the cached mapping data."""
        if not self._is_cache_valid():
            return {"status": "cache_expired"}
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return {
            "status": "valid",
            "last_updated": metadata['timestamp'],
            "total_records": metadata['record_count'],
            "unique_ciks": metadata['cik_count'],
            "unique_series": metadata['series_count'],
            "unique_classes": metadata['class_count'],
            "unique_tickers": metadata['ticker_count'],
            "cache_ttl_days": self.cache_ttl.days
        }