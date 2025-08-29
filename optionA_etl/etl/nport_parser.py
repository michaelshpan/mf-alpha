
import pandas as pd
from lxml import etree

def parse_nport_primary_xml(xml_text: str) -> pd.DataFrame:
    root = etree.fromstring(xml_text.encode("utf-8"))
    rows = []
    report_period = root.findtext(".//ReportPeriod")

    for class_el in root.findall(".//Class"):
        class_id = class_el.findtext("ClassId")
        returns, sales, reinv, redempt = [], [], [], []
        for i in range(1, 13):
            r = class_el.findtext(f"MonthlyTotalReturn{i}")
            if r is None:
                break
            returns.append(float(r))

            s = class_el.findtext(f"SalesFlowMon{i}") or 0
            reinv_ = class_el.findtext(f"ReinvestmentFlowMon{i}") or 0
            red_ = class_el.findtext(f"RedemptionFlowMon{i}") or 0
            sales.append(float(s))
            reinv.append(float(reinv_))
            redempt.append(float(red_))

        total_inv = class_el.findtext("TotalInvestments")
        cash = class_el.findtext("CashNotReported")
        total_inv = float(total_inv) if total_inv else None
        cash = float(cash) if cash else None

        for j, r in enumerate(returns):
            month_end = pd.to_datetime(report_period) - pd.offsets.MonthEnd(len(returns) - j)
            rows.append({
                "class_id": class_id,
                "month_end": month_end,
                "return": r,
                "sales": sales[j],
                "reinvest": reinv[j],
                "redemptions": redempt[j],
                "total_investments": total_inv,
                "cash": cash,
            })

    return pd.DataFrame(rows)
