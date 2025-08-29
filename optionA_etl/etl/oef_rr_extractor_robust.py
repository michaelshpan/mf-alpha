import os
import re
import logging
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
from lxml import html

def _sec_headers()->dict:
    ua=os.getenv("SEC_USER_AGENT","Example Contact email@example.com")
    return {"User-Agent": ua, "Accept-Encoding":"gzip, deflate", "Host":"data.sec.gov"}

BASE="https://data.sec.gov"
# Look for forms that contain expense ratio and turnover data
# N-1A forms often contain this data in HTML format, while N-CEN has XBRL
RR_FORMS={"485BPOS","485APOS","485BXT","485A24F","N-1A","N-1A/A","N-CEN","N-CEN/A","497","497/A"}
log=logging.getLogger(__name__)

def get_submissions_for_cik(cik:str)->Dict[str,Any]:
    url=f"{BASE}/submissions/CIK{int(cik):010d}.json"
    r=requests.get(url, headers=_sec_headers(), timeout=30); r.raise_for_status(); return r.json()

def fetch_latest_rr_doc_for_cik_filtered(cik:str, min_year:int=2010)->Optional[Dict[str,str]]:
    sub=get_submissions_for_cik(cik); rec=sub.get("filings",{}).get("recent",{})
    forms=rec.get("form",[]); accs=rec.get("accessionNumber",[]); prim=rec.get("primaryDocument",[]); dates=rec.get("filingDate",[])
    
    # Filter for filings after min_year and look for latest first
    for f,a,p,d in zip(reversed(forms), reversed(accs), reversed(prim), reversed(dates)):
        if f in RR_FORMS:
            try:
                filing_year = int(d[:4]) if d else 0
                if filing_year >= min_year:
                    return {"accession":a,"primary_doc":p,"filing_date":d}
            except (ValueError, IndexError):
                continue
    return None

def download_rr_document(cik:str, accession:str, primary_doc:str)->str:
    acc=accession.replace("-","")
    url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{primary_doc}"
    headers = {"User-Agent": os.getenv("SEC_USER_AGENT","Example Contact email@example.com"), "Accept-Encoding":"gzip, deflate"}
    r=requests.get(url, headers=headers, timeout=60); r.raise_for_status(); return r.text

def _percent_from_text(s:str)->Optional[float]:
    if s is None: return None
    txt=s.strip()
    m=re.search(r"(\d+(?:\.\d+)?)\s*bps", txt, re.I)
    if m: return float(m.group(1))/10000.0
    m=re.search(r"(\d+(?:\.\d+)?)\s*%", txt)
    if m: return float(m.group(1))/100.0
    m=re.search(r"(\d+(?:\.\d+)?)", txt)
    if m:
        v=float(m.group(1)); return v if v<1.0 else v/100.0
    return None

def _extract_from_html_prospectus(doc_text: str) -> List[Dict[str, Any]]:
    """
    Extract expense ratios and turnover from HTML prospectus documents (N-1A forms)
    """
    results = []
    
    # Look for expense ratio tables or sections
    expense_patterns = [
        r"(?i)(?:net\s+)?expense\s+ratio.*?(\d+\.\d{1,3})%",
        r"(?i)annual\s+fund\s+operating\s+expenses.*?(\d+\.\d{1,3})%",
        r"(?i)total\s+annual\s+fund\s+operating\s+expenses.*?(\d+\.\d{1,3})%"
    ]
    
    turnover_patterns = [
        r"(?i)portfolio\s+turnover\s+rate.*?(\d+)%",
        r"(?i)turnover\s+rate.*?(\d+)%"
    ]
    
    # Find share class symbols/tickers (more specific patterns)
    ticker_patterns = [
        r'\b([A-Z]{4,5}X?)\)',  # ticker symbols in parentheses like (VHCOX)
        r'Fund\s+([A-Z]+)\s+Shares',  # "Fund XXX Shares" 
        r'([A-Z]+)\s+Shares',  # "XXX Shares"
        r'Class\s+([A-Z]+)',  # "Class A"
    ]
    
    classes_found = set()
    for pattern in ticker_patterns:
        matches = re.findall(pattern, doc_text)
        for match in matches:
            if match and 2 <= len(match) <= 6:  # reasonable ticker length
                classes_found.add(match)
    
    # Find expense ratios
    expense_ratios = []
    for pattern in expense_patterns:
        matches = re.finditer(pattern, doc_text)
        for match in matches:
            expense_ratios.append(float(match.group(1)) / 100.0)
    
    # Find turnover rates  
    turnovers = []
    for pattern in turnover_patterns:
        matches = re.finditer(pattern, doc_text)
        for match in matches:
            turnovers.append(float(match.group(1)) / 100.0)
    
    # If we found both classes and data, try to match them up
    if classes_found and (expense_ratios or turnovers):
        # Simple heuristic: if we have same number of classes and ratios, pair them
        if len(classes_found) == len(expense_ratios):
            for i, class_name in enumerate(classes_found):
                results.append({
                    'series_id': None,
                    'class_id': class_name,
                    'context_ref': f'html_class_{i}',
                    'net_expense_ratio': expense_ratios[i] if i < len(expense_ratios) else None,
                    'turnover_pct': turnovers[i] if i < len(turnovers) else None
                })
        else:
            # Apply the same values to all found classes (fund-level data)
            expense_ratio = expense_ratios[0] if expense_ratios else None
            turnover = turnovers[0] if turnovers else None
            
            for class_name in list(classes_found)[:5]:  # limit to first 5 unique classes
                results.append({
                    'series_id': None,
                    'class_id': class_name,
                    'context_ref': 'html_fund_level',
                    'net_expense_ratio': expense_ratio,
                    'turnover_pct': turnover
                })
    
    return results

def _extract_rr_fields_from_ixbrl(doc_text: str) -> List[Dict[str, Any]]:
    """
    Extract class-specific expense ratios and turnover from OEF/RR document
    Handles both XBRL and HTML formats. Returns a list of records, one per class/series combination found
    """
    results = []
    
    # Detect document type
    is_html_doc = '<html>' in doc_text.lower() or '<TYPE>' in doc_text
    is_xbrl_doc = 'xbrl' in doc_text.lower() or 'contextRef' in doc_text
    
    log.info(f"Document analysis: HTML={is_html_doc}, XBRL={is_xbrl_doc}, length={len(doc_text)}")
    
    # If it's an HTML document (like N-1A prospectus), use HTML parsing
    if is_html_doc and not is_xbrl_doc:
        log.info("Using HTML prospectus parsing")
        return _extract_from_html_prospectus(doc_text)
    
    # Otherwise, try XBRL parsing
    try:
        tree = html.fromstring(doc_text)
    except Exception:
        log.warning("Failed to parse document as XML/HTML")
        return results
    
    # First, extract class and series context information
    series_contexts = {}
    class_contexts = {}
    
    # Find all SeriesId and ClassContractId elements with multiple xpath strategies
    series_elements = []
    class_elements = []
    
    # Try multiple namespace variations and tag names
    series_patterns = [
        ".//dei:SeriesId",
        ".//dei:SeriesClassId", 
        ".//*[local-name()='SeriesId']",
        ".//*[local-name()='SeriesClassId']",
        ".//*[contains(local-name(), 'SeriesId')]"
    ]
    
    class_patterns = [
        ".//dei:ClassContractId",
        ".//dei:ClassId", 
        ".//*[local-name()='ClassContractId']",
        ".//*[local-name()='ClassId']",
        ".//*[contains(local-name(), 'ClassId')]",
        ".//*[contains(local-name(), 'ClassContract')]"
    ]
    
    namespaces = {
        'dei': 'http://xbrl.sec.gov/dei/2023',
        'dei2022': 'http://xbrl.sec.gov/dei/2022',
        'dei2021': 'http://xbrl.sec.gov/dei/2021'
    }
    
    # Try different namespace versions for series
    for pattern in series_patterns:
        if 'dei:' in pattern:
            for ns_prefix, ns_uri in namespaces.items():
                try:
                    elements = tree.xpath(pattern.replace('dei:', f'{ns_prefix}:'), namespaces={ns_prefix: ns_uri})
                    if elements:
                        series_elements.extend(elements)
                except:
                    continue
        else:
            try:
                elements = tree.xpath(pattern)
                if elements:
                    series_elements.extend(elements)
            except:
                continue
    
    # Try different namespace versions for classes
    for pattern in class_patterns:
        if 'dei:' in pattern:
            for ns_prefix, ns_uri in namespaces.items():
                try:
                    elements = tree.xpath(pattern.replace('dei:', f'{ns_prefix}:'), namespaces={ns_prefix: ns_uri})
                    if elements:
                        class_elements.extend(elements)
                except:
                    continue
        else:
            try:
                elements = tree.xpath(pattern)
                if elements:
                    class_elements.extend(elements)
            except:
                continue
    
    # Build context mappings
    for el in series_elements:
        context_ref = el.get('contextRef') or el.get('contextref')
        if context_ref and el.text:
            series_contexts[context_ref] = el.text.strip()
    
    for el in class_elements:
        context_ref = el.get('contextRef') or el.get('contextref')
        if context_ref and el.text:
            class_contexts[context_ref] = el.text.strip()
    
    log.info(f"Found {len(series_contexts)} series contexts, {len(class_contexts)} class contexts")
    
    # Now find expense ratio and turnover data with their contexts
    class_data = {}  # context_ref -> {expense_ratio, turnover, series_id, class_id}
    
    # Look for expense ratio elements
    er_xpath_patterns = [
        ".//*[contains(local-name(), 'NetExpenseRatio') or contains(local-name(), 'ExpenseRatio')]",
        ".//*[contains(@name, 'NetExpenseRatio') or contains(@name, 'ExpenseRatio')]"
    ]
    
    for xpath in er_xpath_patterns:
        er_elements = tree.xpath(xpath)
        for el in er_elements:
            context_ref = el.get('contextRef') or el.get('contextref')
            if context_ref and el.text:
                value = _percent_from_text(el.text.strip())
                if value is not None:
                    if context_ref not in class_data:
                        class_data[context_ref] = {}
                    class_data[context_ref]['net_expense_ratio'] = value
    
    # Look for turnover elements
    turnover_xpath_patterns = [
        ".//*[contains(local-name(), 'PortfolioTurnover') or contains(local-name(), 'Turnover')]",
        ".//*[contains(@name, 'PortfolioTurnover') or contains(@name, 'Turnover')]"
    ]
    
    for xpath in turnover_xpath_patterns:
        turnover_elements = tree.xpath(xpath)
        for el in turnover_elements:
            context_ref = el.get('contextRef') or el.get('contextref')
            if context_ref and el.text:
                value = _percent_from_text(el.text.strip())
                if value is not None:
                    if context_ref not in class_data:
                        class_data[context_ref] = {}
                    class_data[context_ref]['turnover_pct'] = value
    
    # Combine data with series/class context information
    for context_ref, data in class_data.items():
        series_id = series_contexts.get(context_ref)
        class_id = class_contexts.get(context_ref)
        
        result = {
            'series_id': series_id,
            'class_id': class_id,
            'context_ref': context_ref,
            'net_expense_ratio': data.get('net_expense_ratio'),
            'turnover_pct': data.get('turnover_pct')
        }
        
        # Only include if we have at least one data value
        if result['net_expense_ratio'] is not None or result['turnover_pct'] is not None:
            results.append(result)
    
    # Fallback: if no class-specific data found, try the old method for fund-level data
    if not results:
        log.info("No class-specific data found, falling back to fund-level extraction")
        nodes = tree.xpath("//*[contains(name(), 'non') and (@name or @contextref)]")
        net_er = None
        turnover = None
        
        for el in nodes:
            label = (el.get("name") or el.get("contextref") or "").lower()
            text = el.text_content().strip()
            if net_er is None and any(k in label for k in ["netexpenseratio","netexp","expenseratio"]):
                v = _percent_from_text(text)
                if v is not None:
                    net_er = v
            if turnover is None and any(k in label for k in ["portfolioturnover","turnover"]):
                v = _percent_from_text(text)
                if v is not None:
                    turnover = v
            if net_er is not None and turnover is not None:
                break
        
        # Regex fallback
        if net_er is None:
            m = re.search(r"net\s+expense\s+ratio[^%\d]{0,40}(\d+(?:\.\d+)?)\s*%", doc_text, re.I)
            if m:
                net_er = float(m.group(1))/100.0
        if turnover is None:
            m = re.search(r"portfolio\s+turnover[^%\d]{0,40}(\d+(?:\.\d+)?)\s*%", doc_text, re.I)
            if m:
                turnover = float(m.group(1))/100.0
        
        if net_er is not None or turnover is not None:
            results.append({
                'series_id': None,
                'class_id': None,
                'context_ref': 'fund_level',
                'net_expense_ratio': net_er,
                'turnover_pct': turnover
            })
    
    log.info(f"Extracted {len(results)} class-specific records")
    return results

def get_er_turnover_for_entities(entities:List[Dict[str,Any]])->pd.DataFrame:
    rows=[]
    for ent in entities:
        cik=ent.get("cik"); 
        if not cik: continue
        try:
            # Try multiple years starting from recent to find accessible filings
            hit = None
            doc = None
            for min_year in [2020, 2015, 2010, 2005, 1990]:  # Progressively try older years
                hit = fetch_latest_rr_doc_for_cik_filtered(cik, min_year)
                if hit:
                    try:
                        doc=download_rr_document(cik, hit["accession"], hit["primary_doc"])
                        log.info("Successfully found OEF/RR filing for CIK %s from year %d+", cik, min_year)
                        break  # Success - exit the year loop
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 404:
                            log.debug("404 error for CIK %s year %d+, trying older filings", cik, min_year)
                            hit = None  # Try next year range
                            continue
                        else:
                            raise  # Re-raise non-404 errors
            
            if not hit or not doc:
                log.warning("No accessible RR/OEF filings found for CIK %s", cik)
                continue
                
            # Extract class-specific data (now returns list of records)
            class_records = _extract_rr_fields_from_ixbrl(doc)
            if not class_records: 
                log.warning("No expense ratio/turnover data found in filing for CIK %s", cik)
                continue
                
            asof = hit.get("filing_date")
            
            # Process class-specific records
            for record in class_records:
                rows.append({
                    "cik": f"{int(cik)}",
                    "series_id": record.get("series_id"),
                    "class_id": record.get("class_id"),
                    "net_expense_ratio": record.get("net_expense_ratio"),
                    "turnover_pct": record.get("turnover_pct"),
                    "asof": asof,
                    "context_ref": record.get("context_ref")
                })
        except Exception as e:
            log.exception("OEF/RR fetch/parse failed for CIK %s: %s", cik, e)
            continue
    
    df = pd.DataFrame(rows)
    log.info(f"OEF/RR extraction completed: {len(df)} records across {df['cik'].nunique() if len(df) > 0 else 0} CIKs")
    return df

def extract_minimal_oef_rr(text:str, class_ids:List[str])->pd.DataFrame:
    class_records = _extract_rr_fields_from_ixbrl(text)
    rows = []
    
    # If we found class-specific records, use them
    if class_records:
        for record in class_records:
            if record.get("class_id") in class_ids:
                rows.append({
                    "class_id": record.get("class_id"),
                    "net_expense_ratio": record.get("net_expense_ratio"),
                    "turnover_pct": record.get("turnover_pct")
                })
    
    # If no matching class records found, apply fund-level data to requested classes
    if not rows and class_records:
        fund_level_record = next((r for r in class_records if r.get("context_ref") == "fund_level"), None)
        if fund_level_record:
            for cid in class_ids:
                rows.append({
                    "class_id": cid,
                    "net_expense_ratio": fund_level_record.get("net_expense_ratio"),
                    "turnover_pct": fund_level_record.get("turnover_pct")
                })
    
    return pd.DataFrame(rows)