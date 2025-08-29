import io, zipfile, requests, pandas as pd
from .config import AppConfig
from .sec_edgar import _headers
FF5_URL="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_URL="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
def _download_ff5_csv(url, headers):
    r=requests.get(url, headers=headers, timeout=60); r.raise_for_status()
    z=zipfile.ZipFile(io.BytesIO(r.content)); name=[n for n in z.namelist() if n.lower().endswith(".csv")][0]
    raw=z.read(name).decode("utf-8","ignore"); lines=raw.splitlines()
    
    # Find header line (contains "Mkt-RF")
    header_idx = next(i for i,l in enumerate(lines) if "Mkt-RF" in l)
    start=next(i for i,l in enumerate(lines) if l.strip()[:6].isdigit())
    
    try:
        end=next(i for i,l in enumerate(lines[start:], start) if l.strip().startswith("Annual Factors:"))
    except StopIteration:
        end = len(lines)  # Use all remaining lines if no "Annual Factors" found
    
    # Combine header with data
    csv_content = lines[header_idx] + "\n" + "\n".join(lines[start:end])
    return pd.read_csv(io.StringIO(csv_content.strip()))

def _download_mom_csv(url, headers):
    r=requests.get(url, headers=headers, timeout=60); r.raise_for_status()
    z=zipfile.ZipFile(io.BytesIO(r.content)); name=[n for n in z.namelist() if n.lower().endswith(".csv")][0]
    raw=z.read(name).decode("utf-8","ignore"); lines=raw.splitlines()
    
    # Find header line (contains "Mom")
    header_idx = next(i for i,l in enumerate(lines) if ",Mom" in l)
    start=next(i for i,l in enumerate(lines) if l.strip()[:6].isdigit())
    
    try:
        end=next(i for i,l in enumerate(lines[start:], start) if l.strip().startswith("Annual Factors:"))
    except StopIteration:
        end = len(lines)  # Use all remaining lines if no "Annual Factors" found
    
    # Combine header with data
    csv_content = lines[header_idx] + "\n" + "\n".join(lines[start:end])
    return pd.read_csv(io.StringIO(csv_content.strip()))
def get_monthly_ff5_mom(cfg:AppConfig)->pd.DataFrame:
    headers=_headers(cfg)
    ff5=_download_ff5_csv(FF5_URL, headers); ff5.columns=[c.strip().replace(" ","") for c in ff5.columns]; ff5=ff5.rename(columns={"Mkt-RF":"MKT_RF"})
    ff5["yyyymm"]=ff5.iloc[:,0].astype(str).str[:6]; ff5=ff5.drop(ff5.columns[0],axis=1)
    ff5["date"]=pd.to_datetime(ff5["yyyymm"]+"01",format="%Y%m%d"); ff5["month_end"]=ff5["date"]+pd.offsets.MonthEnd(0); ff5=ff5.drop(columns=["date","yyyymm"])
    mom=_download_mom_csv(MOM_URL, headers); mom.columns=[c.strip().replace(" ","") for c in mom.columns]; mom["yyyymm"]=mom.iloc[:,0].astype(str).str[:6]
    mom=mom.drop(mom.columns[0],axis=1); mom=mom.rename(columns={"Mom":"MOM"}); mom["date"]=pd.to_datetime(mom["yyyymm"]+"01",format="%Y%m%d")
    mom["month_end"]=mom["date"]+pd.offsets.MonthEnd(0); mom=mom[["month_end","MOM"]]
    fac=ff5.merge(mom,on="month_end", how="left")
    for c in ["MKT_RF","SMB","HML","RMW","CMA","RF","MOM"]: 
        if c in fac.columns:
            fac[c]=fac[c].astype(float)/100.0
        else:
            print(f"Warning: Missing column {c} in factor data. Available columns: {list(fac.columns)}")
    return fac
