import pandas as pd
from lxml import etree

def safe_float(value, default=0.0):
    """Safely convert a value to float, handling 'N/A' and other non-numeric strings"""
    if value is None:
        return default
    if isinstance(value, str) and value.upper() in ['N/A', 'NA', '', 'NULL', 'NONE']:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def parse_nport_primary_xml(xml_text: str) -> pd.DataFrame:
    root = etree.fromstring(xml_text.encode("utf-8"))
    rows = []
    
    # Define namespace
    ns = {'nport': 'http://www.sec.gov/edgar/nport'}
    
    # Get report period end date
    report_period = root.findtext(".//nport:repPdEnd", namespaces=ns) or root.findtext(".//ReportPeriod")
    if not report_period:
        return pd.DataFrame()
    
    report_date = pd.to_datetime(report_period)
    
    # Find monthly total returns
    monthly_returns = root.findall(".//nport:monthlyTotReturn", namespaces=ns)
    
    # Find monthly flows  
    mon1_flow = root.find(".//nport:mon1Flow", namespaces=ns)
    mon2_flow = root.find(".//nport:mon2Flow", namespaces=ns) 
    mon3_flow = root.find(".//nport:mon3Flow", namespaces=ns)
    
    # Find total assets/investments
    total_assets = root.findtext(".//nport:totAssets", namespaces=ns)
    net_assets = root.findtext(".//nport:netAssets", namespaces=ns)
    
    for ret_el in monthly_returns:
        class_id = ret_el.get("classId")
        if not class_id:
            continue
            
        # Extract returns (rtn1=month-1, rtn2=month-2, rtn3=month-3)
        returns = []
        for i in range(1, 4):  # N-PORT typically has 3 months
            rtn = ret_el.get(f"rtn{i}")
            if rtn and rtn.upper() not in ['N/A', 'NULL', '']:
                rtn_value = safe_float(rtn)
                if rtn_value is not None:
                    returns.append(rtn_value / 100.0)  # Convert percentage to decimal
        
        # Extract flows for each month
        flows = []
        for flow_el in [mon1_flow, mon2_flow, mon3_flow]:
            if flow_el is not None:
                sales = safe_float(flow_el.get("sales", 0))
                redemptions = safe_float(flow_el.get("redemption", 0)) 
                reinvestment = safe_float(flow_el.get("reinvestment", 0))
                flows.append((sales, redemptions, reinvestment))
            else:
                flows.append((0, 0, 0))
        
        # Create rows for each month in chronological order
        # rtn1 = most recent month, rtn2 = month-1, rtn3 = month-2
        # Convert to chronological order: oldest to newest
        for j, rtn in enumerate(returns):
            # Calculate month_end in chronological order (oldest first)
            months_back = len(returns) - 1 - j  # 2, 1, 0 for 3 returns
            month_end = report_date - pd.offsets.MonthEnd(months_back)
            
            # Get corresponding flow data (flows are in same order as returns)
            sales, redemptions, reinvest = flows[j] if j < len(flows) else (0, 0, 0)
            
            # Data quality validation: flag extreme returns
            if abs(rtn) > 0.5:  # More than ±50%
                import logging
                log = logging.getLogger(__name__)
                log.warning(f"Extreme return detected: {class_id} {month_end.strftime('%Y-%m')} return={rtn:.1%}")
            
            rows.append({
                "class_id": class_id,
                "month_end": month_end,
                "return": rtn,
                "sales": sales,
                "reinvest": reinvest,
                "redemptions": redemptions,
                "total_investments": safe_float(total_assets),
                "cash": 0.0,  # Not directly available in this format
            })

    return pd.DataFrame(rows)