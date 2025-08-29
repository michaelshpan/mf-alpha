
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
RR_FORMS={"485BPOS","485APOS","485BXT","485A24F","N-1A","N-1A/A"}
log=logging.getLogger(__name__)

def get_submissions_for_cik(cik:str)->Dict[str,Any]:
    url=f"{BASE}/submissions/CIK{int(cik):010d}.json"
    r=requests.get(url, headers=_sec_headers(), timeout=30); r.raise_for_status(); return r.json()

def fetch_latest_rr_doc_for_cik(cik:str)->Optional[Dict[str,str]]:
    sub=get_submissions_for_cik(cik); rec=sub.get("filings",{}).get("recent",{})
    forms=rec.get("form",[]); accs=rec.get("accessionNumber",[]); prim=rec.get("primaryDocument",[]); dates=rec.get("filingDate",[])
    for f,a,p,d in zip(reversed(forms), reversed(accs), reversed(prim), reversed(dates)):
        if f in RR_FORMS: return {"accession":a,"primary_doc":p,"filing_date":d}
    return None

def download_rr_document(cik:str, accession:str, primary_doc:str)->str:
    acc=accession.replace("-",""); url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{primary_doc}"
    r=requests.get(url, headers=_sec_headers(), timeout=60); r.raise_for_status(); return r.text

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

def _extract_rr_fields_from_ixbrl(doc_text:str)->Dict[str,Any]:
    out={}
    try: tree=html.fromstring(doc_text)
    except Exception: return out
    nodes=tree.xpath("//*[contains(name(), 'non') and (@name or @contextref)]")
    net_er=None; turnover=None
    for el in nodes:
        label=(el.get("name") or el.get("contextref") or "").lower()
        text=el.text_content().strip()
        if net_er is None and any(k in label for k in ["netexpenseratio","netexp","expenseratio"]):
            v=_percent_from_text(text); 
            if v is not None: net_er=v
        if turnover is None and any(k in label for k in ["portfolioturnover","turnover"]):
            v=_percent_from_text(text); 
            if v is not None: turnover=v
        if net_er is not None and turnover is not None: break
    if net_er is None:
        m=re.search(r"net\s+expense\s+ratio[^%\d]{0,40}(\d+(?:\.\d+)?)\s*%", doc_text, re.I)
        if m: net_er=float(m.group(1))/100.0
    if turnover is None:
        m=re.search(r"portfolio\s+turnover[^%\d]{0,40}(\d+(?:\.\d+)?)\s*%", doc_text, re.I)
        if m: turnover=float(m.group(1))/100.0
    if net_er is not None: out["net_expense_ratio"]=net_er
    if turnover is not None: out["turnover_pct"]=turnover
    return out

def get_er_turnover_for_entities(entities:List[Dict[str,Any]])->pd.DataFrame:
    rows=[]
    for ent in entities:
        cik=ent.get("cik"); 
        if not cik: continue
        try:
            hit=fetch_latest_rr_doc_for_cik(cik)
            if not hit: continue
            doc=download_rr_document(cik, hit["accession"], hit["primary_doc"])
            vals=_extract_rr_fields_from_ixbrl(doc)
            if not vals: continue
            asof=hit.get("filing_date")
            sids=ent.get("series_ids") or [None]; cids=ent.get("class_ids") or [None]
            for sid in sids:
                for cid in cids:
                    rows.append({"cik":f"{int(cik)}","series_id":sid,"class_id":cid,"net_expense_ratio":vals.get("net_expense_ratio"),"turnover_pct":vals.get("turnover_pct"),"asof":asof})
        except Exception as e:
            log.exception("OEF/RR fetch/parse failed for CIK %s: %s", cik, e)
            continue
    return pd.DataFrame(rows)

def extract_minimal_oef_rr(text:str, class_ids:List[str])->pd.DataFrame:
    vals=_extract_rr_fields_from_ixbrl(text); rows=[]
    for cid in class_ids:
        rows.append({"class_id":cid,"net_expense_ratio":vals.get("net_expense_ratio"),"turnover_pct":vals.get("turnover_pct")})
    return pd.DataFrame(rows)
