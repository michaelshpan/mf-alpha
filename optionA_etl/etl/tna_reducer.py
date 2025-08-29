
import pandas as pd

def compute_tna_proxy(df: pd.DataFrame) -> pd.DataFrame:
    anchors = df.dropna(subset=["total_investments","cash"])
    anchors = anchors.copy()
    anchors["tna_anchor"] = anchors["total_investments"] + anchors["cash"]
    anchors = anchors[["class_id","month_end","tna_anchor"]]
    df = df.merge(anchors, on=["class_id","month_end"], how="left")
    df = df.sort_values(["class_id","month_end"])
    tnas = []
    for cid, g in df.groupby("class_id"):
        tna = None
        for _, row in g.iterrows():
            if pd.notnull(row["tna_anchor"]):
                tna = row["tna_anchor"]
            elif tna is not None and pd.notnull(row["return"]) and pd.notnull(row["net_flow"]):
                tna = tna * (1+row["return"]) + row["net_flow"]
            tnas.append({"class_id": cid, "month_end": row["month_end"], "tna": tna})
    return pd.DataFrame(tnas)
