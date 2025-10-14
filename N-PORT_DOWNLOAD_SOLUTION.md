# N-PORT Bulk Data Download Issues - Analysis & Solutions

## Problem Summary
All N-PORT bulk data downloads are failing with 404 errors. The attempted URL pattern:
```
https://www.sec.gov/files/structureddata/data/form-n-port-data-sets/2023q4_form_nport.zip
```
does not exist.

## Root Cause Analysis

### Investigation Results:
1. **SEC does NOT provide N-PORT data in quarterly bulk ZIP files** like financial statements
2. The URL pattern assumed in the code is incorrect
3. N-PORT data is available but through different mechanisms

## Solutions

### Solution 1: Use EDGAR Full Text Search API (Recommended)
The SEC provides N-PORT data through their EDGAR full text search API, not bulk downloads.

```python
def fetch_nport_via_api(cik, start_date, end_date):
    """
    Fetch N-PORT filings using SEC EDGAR API
    """
    base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        'action': 'getcompany',
        'CIK': cik,
        'type': 'NPORT',
        'dateb': end_date,
        'datea': start_date,
        'output': 'json'
    }
    
    response = requests.get(base_url, params=params)
    # Process individual filings
```

### Solution 2: Use data.sec.gov Submissions API
Individual company submissions including N-PORT:

```python
def get_nport_from_submissions(cik):
    """
    Get N-PORT filings from submissions endpoint
    """
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    headers = {'User-Agent': 'Research User research@university.edu'}
    response = requests.get(url, headers=headers)
    data = response.json()
    
    # Filter for NPORT filings
    recent = data['filings']['recent']
    nport_filings = [
        i for i, form in enumerate(recent['form']) 
        if 'NPORT' in form.upper()
    ]
    
    return nport_filings
```

### Solution 3: Use Individual Filing Downloads (Current Approach)
Your current pipeline already does this correctly:

```python
# From your existing code - this is the RIGHT approach
filings = list_recent_nport_p_accessions(cik, appcfg, since_yyyymmdd=a.since)
for f in filings:
    xml = download_filing_xml(cik, f["accession"], f["primary_doc"], appcfg)
    df = parse_nport_primary_xml(xml)
```

### Solution 4: Alternative - Financial Statement Data Sets
While not N-PORT specific, the SEC does provide quarterly financial data:

```python
def get_financial_data(quarter):
    """
    Get financial statement data (includes some fund data)
    """
    url = f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
    # This contains different data structure than N-PORT
```

## Recommended Fix for Your Pipeline

### Option A: Remove Bulk Download Feature (Simplest)
Since the bulk download doesn't exist as expected, rely on the existing individual filing approach:

```python
# In pilot_pipeline.py - just use the existing N-PORT parsing
# Remove or disable the --use-returns-db flag functionality
```

### Option B: Enhance Individual Filing Collection
Modify the pipeline to fetch more complete data:

```python
def enhanced_nport_collection(cik, start_date, end_date):
    """
    Enhanced collection that gets ALL N-PORT filings
    """
    # 1. Get all NPORT filings (not just NPORT-P)
    all_forms = ['NPORT-P', 'NPORT-EX', 'NPORT-NP']
    
    all_filings = []
    for form_type in all_forms:
        filings = list_recent_filings(cik, form_type, start_date)
        all_filings.extend(filings)
    
    # 2. Process all filings
    monthly_data = []
    for filing in all_filings:
        xml = download_filing_xml(filing)
        data = parse_nport_xml(xml)
        monthly_data.extend(data)
    
    # 3. Deduplicate
    df = pd.DataFrame(monthly_data)
    df = df.drop_duplicates(['class_id', 'month_end'], keep='last')
    
    return df
```

### Option C: Build Database from Individual Filings
Create the returns database using the data you already have:

```python
def build_returns_db_from_filings(facts_df):
    """
    Build returns database from parsed N-PORT filings
    """
    # Use the deduplicated facts data
    returns_db = facts_df[['class_id', 'month_end', 'return', 'cik']].copy()
    
    # Fill gaps using interpolation or forward fill
    for class_id in returns_db['class_id'].unique():
        class_data = returns_db[returns_db['class_id'] == class_id]
        
        # Create complete monthly index
        date_range = pd.date_range(
            start=class_data['month_end'].min(),
            end=class_data['month_end'].max(),
            freq='ME'
        )
        
        # Reindex and fill
        class_data = class_data.set_index('month_end').reindex(date_range)
        class_data['return'] = class_data['return'].interpolate(limit=2)
        
        returns_db.update(class_data)
    
    return returns_db
```

## Immediate Workaround

For now, disable the bulk download feature and use the existing approach:

```bash
# Run without --use-returns-db flag
python -m etl.pilot_pipeline \
    --pilot config/funds_pilot.yaml \
    --since 2020-01-01 \
    --out data/pilot_enhanced.parquet \
    --extra-history 48
```

This will:
- Fetch 48 months of historical N-PORT filings
- Parse all available returns
- Still enable alpha calculations after sufficient history

## Long-term Solution

1. **Accept the data structure**: N-PORT data comes from individual filings, not bulk downloads
2. **Optimize the existing approach**: Cache parsed XML data locally
3. **Fill gaps intelligently**: Use the deduplication and gap-filling logic already in place

## Code Changes Needed

### 1. Fix sec_bulk_downloader.py
Either remove it or update to use correct approach:

```python
class SECBulkDataDownloader:
    def __init__(self, ...):
        # Change approach to individual filings
        pass
    
    def build_returns_database(self, ...):
        # Instead of downloading bulk files:
        # 1. Use submissions API to get all N-PORT filings
        # 2. Download and parse each filing
        # 3. Build consolidated database
        pass
```

### 2. Update pilot_pipeline.py
Remove dependency on bulk downloads:

```python
# Comment out or modify this section
if a.use_returns_db:
    log.warning("Bulk download feature not available. Using individual filings.")
    # Continue with existing approach
```

## Testing
```bash
# Test with existing approach (works)
cd optionA_etl
python -m etl.pilot_pipeline \
    --pilot config/funds_pilot.yaml \
    --since 2020-01-01 \
    --extra-history 48 \
    --out data/pilot_test.parquet

# Verify coverage
python -c "
import pandas as pd
df = pd.read_parquet('data/pilot_test.parquet')
print(f'Records: {len(df)}')
print(f'Return coverage: {df.return.notna().mean():.1%}')
"
```

## Conclusion

The SEC does not provide N-PORT data as bulk quarterly downloads like they do for financial statements. The data must be collected from individual filings, which your pipeline already does correctly. The solution is to:

1. Continue using the individual filing approach
2. Enhance deduplication (already implemented)
3. Fetch more historical data with `--extra-history`
4. Accept that this is the intended data access pattern for N-PORT

The "bulk download" feature should be removed or reimplemented to aggregate individual filings rather than expecting pre-packaged quarterly files.