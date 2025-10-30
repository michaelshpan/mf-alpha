#!/usr/bin/env python3
"""
Simple SEC client wrapper for N-PORT filing operations.
Uses existing sec_edgar.py functions.
"""

import logging
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

from .sec_edgar import list_recent_nport_p_accessions, download_filing_xml
from .config import AppConfig

log = logging.getLogger(__name__)


class SECClient:
    """Simple wrapper for SEC EDGAR operations."""
    
    def __init__(self, config: AppConfig = None):
        """Initialize SEC client with configuration."""
        self.config = config or AppConfig()
    
    def get_nport_filings(
        self,
        cik: str,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Get N-PORT filings for a CIK within date range.
        
        Args:
            cik: Central Index Key
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of filing dictionaries with accession_number, primary_document, filing_date
        """
        try:
            # Convert start_date to the format expected by list_recent_nport_p_accessions
            since_date = start_date.replace('-', '')  # YYYY-MM-DD -> YYYYMMDD
            
            # Get recent N-PORT filings
            accessions = list_recent_nport_p_accessions(
                cik=cik,
                cfg=self.config,
                since_yyyymmdd=since_date
            )
            
            # Filter by date range and format output
            filings = []
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            for acc in accessions:
                filing_date = acc.get('filing_date', '')
                
                if filing_date:
                    try:
                        filing_dt = pd.to_datetime(filing_date)
                        
                        # Check if filing is within requested date range
                        if start_dt <= filing_dt <= end_dt:
                            filings.append({
                                'accession_number': acc.get('accession', ''),
                                'primary_document': acc.get('primary_doc', ''),
                                'filing_date': filing_date
                            })
                    except Exception as e:
                        log.debug(f"Could not parse filing date {filing_date}: {e}")
                        continue
            
            log.info(f"Found {len(filings)} N-PORT filings for CIK {cik} in date range")
            return filings
            
        except Exception as e:
            log.error(f"Failed to get N-PORT filings for CIK {cik}: {e}")
            return []
    
    def download_filing(
        self,
        cik: str,
        accession_number: str,
        primary_document: str
    ) -> str:
        """
        Download filing XML content.
        
        Args:
            cik: Central Index Key  
            accession_number: SEC accession number
            primary_document: Primary document filename
            
        Returns:
            XML content as string
        """
        try:
            xml_content = download_filing_xml(
                cik=cik,
                accession=accession_number,
                primary_doc=primary_document,
                cfg=self.config
            )
            
            log.debug(f"Downloaded filing {accession_number} for CIK {cik}")
            return xml_content
            
        except Exception as e:
            log.error(f"Failed to download filing {accession_number} for CIK {cik}: {e}")
            raise