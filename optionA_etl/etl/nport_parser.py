
import pandas as pd
from lxml import etree

def parse_nport_primary_xml(xml_text: str) -> pd.DataFrame:
    root = etree.fromstring(xml_text.encode("utf-8"))
    rows = []
    
    # Define namespaces for the XML document
    namespaces = {
        'nport': 'http://www.sec.gov/edgar/nport',
        'com': 'http://www.sec.gov/edgar/common',
        'ncom': 'http://www.sec.gov/edgar/nportcommon'
    }
    
    # Extract report date from repPdDate (current format) or ReportPeriod (legacy)
    report_period = root.findtext(".//nport:repPdDate", namespaces=namespaces)
    if not report_period:
        report_period = root.findtext(".//ReportPeriod")  # Legacy format without namespace

    # Parse class IDs from seriesClassInfo section
    class_ids = []
    for class_el in root.findall(".//nport:classId", namespaces=namespaces):
        class_id = class_el.text
        if class_id:
            class_ids.append(class_id)

    # Parse monthly returns (backup only - Tradefeeds OHLCV is primary source)
    monthly_returns = {}
    for return_el in root.findall(".//nport:monthlyTotReturn", namespaces=namespaces):
        class_id = return_el.get("classId")
        if class_id:
            returns = []
            # Extract rtn1, rtn2, rtn3 (most recent 3 months)
            for rtn_attr in ["rtn3", "rtn2", "rtn1"]:  # rtn3 = oldest, rtn1 = newest
                rtn_val = return_el.get(rtn_attr)
                if rtn_val:
                    try:
                        returns.append(float(rtn_val))
                    except ValueError:
                        if rtn_val.strip().upper() not in ['N/A', 'NA', 'NULL', '', 'NONE']:
                            print(f"Warning: Unexpected return value '{rtn_val}' for {rtn_attr} - skipping")
                        # Skip this return value
            monthly_returns[class_id] = returns
    
    # Parse flow data from mon1Flow, mon2Flow, mon3Flow
    flow_data = {}
    for flow_num, month_num in [("mon3", 3), ("mon2", 2), ("mon1", 1)]:  # mon1 = newest
        flow_el = root.find(f".//nport:{flow_num}Flow", namespaces=namespaces)
        if flow_el is not None:
            def safe_float(value, field_name):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    if str(value).strip().upper() not in ['N/A', 'NA', 'NULL', '', 'NONE', '0']:
                        print(f"Warning: Unexpected {field_name} value '{value}' - setting to 0")
                    return 0.0
            
            sales = safe_float(flow_el.get("sales", 0), "sales")
            redemption = safe_float(flow_el.get("redemption", 0), "redemption") 
            reinvestment = safe_float(flow_el.get("reinvestment", 0), "reinvestment")
            flow_data[month_num] = {
                'sales': sales,
                'redemptions': redemption,
                'reinvest': reinvestment
            }

    # Extract TNA (Net Assets) from <fundInfo><netAssets>
    net_assets = None
    fund_info = root.find(".//nport:fundInfo", namespaces=namespaces)
    if fund_info is not None:
        net_assets_text = fund_info.findtext("nport:netAssets", namespaces=namespaces)
        if net_assets_text:
            try:
                # Handle numeric values
                net_assets = float(net_assets_text)
            except ValueError:
                # Handle non-numeric values like 'N/A'
                if net_assets_text.strip().upper() in ['N/A', 'NA', 'NULL', '', 'NONE']:
                    net_assets = None
                else:
                    # Log unexpected values for debugging
                    print(f"Warning: Unexpected netAssets value '{net_assets_text}' - setting to None")
                    net_assets = None
    
    # Extract other fund-level data
    total_assets = None
    total_liabs = None
    cash_not_reported = None
    if fund_info is not None:
        total_assets_text = fund_info.findtext("nport:totAssets", namespaces=namespaces)
        total_liabs_text = fund_info.findtext("nport:totLiabs", namespaces=namespaces) 
        cash_text = fund_info.findtext("nport:cshNotRptdInCorD", namespaces=namespaces)
        
        if total_assets_text:
            try:
                total_assets = float(total_assets_text)
            except ValueError:
                if total_assets_text.strip().upper() not in ['N/A', 'NA', 'NULL', '', 'NONE']:
                    print(f"Warning: Unexpected totAssets value '{total_assets_text}' - setting to None")
                total_assets = None
        if total_liabs_text:
            try:
                total_liabs = float(total_liabs_text)
            except ValueError:
                if total_liabs_text.strip().upper() not in ['N/A', 'NA', 'NULL', '', 'NONE']:
                    print(f"Warning: Unexpected totLiabs value '{total_liabs_text}' - setting to None")
                total_liabs = None
        if cash_text:
            try:
                cash_not_reported = float(cash_text)
            except ValueError:
                if cash_text.strip().upper() not in ['N/A', 'NA', 'NULL', '', 'NONE']:
                    print(f"Warning: Unexpected cshNotRptdInCorD value '{cash_text}' - setting to None")
                cash_not_reported = None

    # Create records for each class and month combination
    if not report_period:
        return pd.DataFrame()
    
    # Process each class
    for class_id in class_ids:
        # Get returns for this class (backup data only)
        class_returns = monthly_returns.get(class_id, [])
        
        # Create records for each month with data
        for month_idx in range(1, 4):  # mon1, mon2, mon3
            month_end = pd.to_datetime(report_period) - pd.offsets.MonthEnd(month_idx - 1)
            
            # Get return for this month (if available as backup)
            return_val = None
            if len(class_returns) >= month_idx:
                return_val = class_returns[month_idx - 1]
            
            # Get flow data for this month
            flows = flow_data.get(month_idx, {})
            
            rows.append({
                "class_id": class_id,
                "month_end": month_end,
                "return": return_val,  # Backup only - Tradefeeds OHLCV is primary
                "sales": flows.get('sales', None),
                "reinvest": flows.get('reinvest', None),
                "redemptions": flows.get('redemptions', None),
                "total_investments": None,  # Not available in new format
                "cash": cash_not_reported,
                "tna": net_assets,  # TNA from fundInfo/netAssets
                "total_assets": total_assets,
                "total_liabs": total_liabs,
            })

    return pd.DataFrame(rows)
