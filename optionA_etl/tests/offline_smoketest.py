
import pandas as pd
from pathlib import Path
import sys

# Ensure module path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from etl.nport_parser import parse_nport_primary_xml
from etl.metrics import compute_net_flow, rolling_factor_regressions, value_added
from etl.tna_reducer import compute_tna_proxy

def main():
    xml_path = root / "tests" / "fixtures" / "nport_sample.xml"
    xml_text = xml_path.read_text()
    df = parse_nport_primary_xml(xml_text)
    assert not df.empty, "Parser returned empty DataFrame"

    df = compute_net_flow(df)

    # Fake factors for the three months (month_end derived from the XML: Jan/Feb/Mar 2024)
    fac = pd.DataFrame({
        "month_end": pd.to_datetime(["2023-12-31","2024-01-31","2024-02-29"]),
        "MKT_RF":[0.005, -0.002, 0.003],
        "SMB":[0.001, 0.0, -0.0005],
        "HML":[-0.0003, 0.0007, 0.0],
        "RMW":[0.0002, -0.0001, 0.0004],
        "CMA":[-0.0001, 0.0002, -0.0002],
        "RF":[0.0003, 0.0003, 0.0003],
        "MOM":[0.0005, -0.0004, 0.0001],
    })

    merged = df.merge(fac, on="month_end", how="left")
    reg = rolling_factor_regressions(merged, min_obs=3, window=3)
    out = merged.merge(reg, on=["class_id","month_end"], how="left")

    # TNA proxy
    tna_df = compute_tna_proxy(out)
    out = out.merge(tna_df, on=["class_id","month_end"], how="left")

    # Minimal ER map (e.g., 0.60%)
    er_map = pd.DataFrame({"class_id":["SAMPLE_CLASS"], "net_expense_ratio":[0.006]})

    out = value_added(out, er_map=er_map, tna_proxy=tna_df)

    # Show a compact summary
    cols = ["class_id","month_end","return","net_flow","tna","alpha_hat","alpha_t","R2","value_added"]
    print(out[cols].sort_values("month_end").to_string(index=False))

if __name__ == "__main__":
    main()
