import requests, logging
from typing import Dict, Any, List
from .config import AppConfig
from .utils import rate_limiter
log = logging.getLogger(__name__)
BASE = "https://data.sec.gov"
def _headers(cfg: AppConfig)->Dict[str,str]:
    return {"User-Agent": cfg.sec.user_agent, "Accept-Encoding":"gzip, deflate","Host":"data.sec.gov"}
def get_submissions_for_cik(cik:str, cfg:AppConfig)->Dict[str,Any]:
    url=f"{BASE}/submissions/CIK{int(cik):010d}.json"
    with rate_limiter(cfg.sec.max_rps) as wait:
        wait()
        try:
            r=requests.get(url, headers=_headers(cfg), timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 404:
                log.warning(f"CIK {cik} not found (404). This may be an invalid CIK or the entity may not file with SEC.")
                return {"filings": {"recent": {}}}
            else:
                raise
def list_recent_nport_p_accessions(cik:str, cfg:AppConfig, since_yyyymmdd:str="2023-01-01")->List[Dict[str,Any]]:
    sub=get_submissions_for_cik(cik,cfg); rec=sub.get("filings",{}).get("recent",{})
    out=[]; 
    for f,acc,prim,dt in zip(rec.get("form",[]), rec.get("accessionNumber",[]), rec.get("primaryDocument",[]), rec.get("filingDate",[])):
        if f.upper().startswith("NPORT-P") and dt>=since_yyyymmdd: out.append({"accession":acc,"primary_doc":prim,"filing_date":dt})
    return out
def download_filing_xml(cik:str, accession:str, primary_doc:str, cfg:AppConfig)->str:
    acc=accession.replace("-","")
    # Handle common SEC metadata issues where primary_doc has incorrect path prefixes
    if primary_doc.startswith("xslFormNPORT-P_X01/"):
        primary_doc = primary_doc.replace("xslFormNPORT-P_X01/", "")
    
    url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{primary_doc}"
    # Use correct headers for www.sec.gov (not data.sec.gov)
    headers = {"User-Agent": cfg.sec.user_agent, "Accept-Encoding": "gzip, deflate"}
    
    with rate_limiter(cfg.sec.max_rps) as wait:
        wait()
        try:
            r=requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            return r.text
        except requests.exceptions.HTTPError as e:
            if r.status_code == 404:
                # Try fallback: just use primary_doc.xml as filename
                fallback_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/primary_doc.xml"
                log.warning(f"Primary doc {primary_doc} not found, trying fallback: {fallback_url}")
                r2 = requests.get(fallback_url, headers=headers, timeout=60)
                r2.raise_for_status()
                return r2.text
            else:
                raise
