import os
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import requests
import pandas as pd
from lxml import html

log = logging.getLogger(__name__)

def _sec_headers() -> dict:
    ua = os.getenv("SEC_USER_AGENT", "Example Contact email@example.com")
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}

def extract_manager_tenure_from_n1a(doc_text: str, filing_date: str) -> Optional[float]:
    """
    Extract portfolio manager tenure from N-1A filing text.
    Returns tenure in years, or None if not found.
    """
    try:
        filing_dt = datetime.strptime(filing_date, "%Y-%m-%d").date()
    except:
        filing_dt = date.today()
    
    # Common patterns for manager tenure/start dates
    patterns = [
        # "since YYYY" or "Since YYYY"
        r"(?:since|Since)\s+(\d{4})",
        # "Mr./Ms. [Name] has managed... since [Month] YYYY"
        r"has\s+(?:managed|been\s+(?:managing|the\s+portfolio\s+manager))[^.]{0,100}since\s+(?:\w+\s+)?(\d{4})",
        # "Portfolio Manager since YYYY"
        r"Portfolio\s+Manager[^.]{0,50}since\s+(?:\w+\s+)?(\d{4})",
        # "[Name] joined... in YYYY" or "joined the fund in YYYY"
        r"joined[^.]{0,100}(?:in|since)\s+(?:\w+\s+)?(\d{4})",
        # "appointed... in YYYY"
        r"appointed[^.]{0,100}(?:in|since)\s+(?:\w+\s+)?(\d{4})",
    ]
    
    start_years = []
    for pattern in patterns:
        matches = re.findall(pattern, doc_text, re.IGNORECASE)
        for match in matches:
            try:
                year = int(match)
                if 1970 <= year <= filing_dt.year:  # Reasonable range
                    start_years.append(year)
            except ValueError:
                continue
    
    if not start_years:
        return None
    
    # Use the most recent (latest) start year found
    start_year = max(start_years)
    tenure_years = filing_dt.year - start_year
    
    # Return tenure, minimum 0 years
    return max(0.0, float(tenure_years))

def extract_fund_inception_date(doc_text: str) -> Optional[str]:
    """
    Extract fund inception date from N-1A filing.
    Returns date as YYYY-MM-DD string, or None if not found.
    """
    patterns = [
        # "commenced operations on [Date]"
        r"commenced\s+operations[^.]{0,50}(?:on|in)\s+(\w+\s+\d{1,2},?\s+\d{4})",
        # "inception date: [Date]" or "inception: [Date]"
        r"inception[^:]{0,20}:?\s*(\w+\s+\d{1,2},?\s+\d{4})",
        # "established [Date]" or "established in [Date]"
        r"established[^.]{0,50}(?:on|in)?\s+(\w+\s+\d{1,2},?\s+\d{4})",
        # "fund began operations [Date]"
        r"(?:fund|portfolio)\s+began\s+operations[^.]{0,50}(\w+\s+\d{1,2},?\s+\d{4})",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, doc_text, re.IGNORECASE)
        for match in matches:
            try:
                # Try to parse the date
                date_str = match.replace(',', '').strip()
                
                # Handle various date formats
                for fmt in ["%B %d %Y", "%b %d %Y", "%B %Y", "%b %Y"]:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        return parsed_date.strftime("%Y-%m-%d")
                    except ValueError:
                        continue
            except:
                continue
    
    return None

def get_manager_data_for_entities(entities: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Fetch manager tenure and fund inception data from N-1A filings for entities.
    """
    from .oef_rr_extractor_robust import get_submissions_for_cik, fetch_latest_rr_doc_for_cik_filtered, download_rr_document
    
    rows = []
    for ent in entities:
        cik = ent.get("cik")
        if not cik:
            continue
            
        try:
            # Try multiple years to find accessible filings
            hit = None
            doc = None
            for min_year in [2020, 2015, 2010, 2005, 1990]:
                hit = fetch_latest_rr_doc_for_cik_filtered(cik, min_year)
                if hit:
                    try:
                        doc = download_rr_document(cik, hit["accession"], hit["primary_doc"])
                        break  # Success
                    except:
                        continue  # Try next year range
            
            if not hit or not doc:
                continue
            
            # Extract manager tenure and inception date
            manager_tenure = extract_manager_tenure_from_n1a(doc, hit["filing_date"])
            inception_date = extract_fund_inception_date(doc)
            
            # Calculate fund age in years if inception date found
            fund_age = None
            if inception_date:
                try:
                    inception_dt = datetime.strptime(inception_date, "%Y-%m-%d").date()
                    filing_dt = datetime.strptime(hit["filing_date"], "%Y-%m-%d").date()
                    fund_age = (filing_dt - inception_dt).days / 365.25
                except:
                    pass
            
            asof = hit.get("filing_date")
            sids = ent.get("series_ids") or [None]
            cids = ent.get("class_ids") or [None]
            
            for sid in sids:
                for cid in cids:
                    rows.append({
                        "cik": f"{int(cik)}",
                        "series_id": sid,
                        "class_id": cid,
                        "manager_tenure": manager_tenure,
                        "fund_age": fund_age,
                        "inception_date": inception_date,
                        "asof": asof
                    })
                    
        except Exception as e:
            log.exception("Manager tenure extraction failed for CIK %s: %s", cik, e)
            continue
    
    return pd.DataFrame(rows)