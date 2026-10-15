import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from plantation import figures, metrics

DATA = Path(os.environ.get("PLANTATION_DATA_DIR", Path(__file__).parent / "data" / "out"))
PERIOD_KEYS = {"Jun-Dec": "JJASOND", "Jan-Dec": "JANDEC"}
SOURCES = {"GSMaP (JAXA, 0.1°)": "gsmap", "IMD (gauge, 0.25°)": "imd"}

st.set_page_config(page_title="plantation_planning", layout="wide")
st.title("plantation_planning")
st.caption("Decadal precip assessment (JJASOND)")


@st.cache_data
def load(source: str, mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"district_season_{source}.parquet")
    df["key"] = df["state"] + "|" + df["district"]
    return df


@st.cache_data
def load_geojson(mtime: float) -> dict:
    return json.loads((DATA / "districts.geojson").read_text())


with st.sidebar:
    src_label = st.radio("Rainfall source", list(SOURCES), key="source")
    per_label = st.radio("Period", list(PERIOD_KEYS), key="period", horizontal=True)
    threshold = st.number_input("Threshold (mm)", min_value=0, max_value=6000,
                                value=800, step=50, key="threshold",
                                help="Districts with a 10-year mean at or above this are shortlisted.")
    src = SOURCES[src_label]
    try:
        df = load(src, (DATA / f"district_season_{src}.parquet").stat().st_mtime)
        geojson = load_geojson((DATA / "districts.geojson").stat().st_mtime)
    except FileNotFoundError:
        st.error("Data files missing; run build_dataset.py")
        st.stop()
    states = st.multiselect("State", sorted(df["state"].unique()), key="states")
    query = st.text_input("Search district", key="query")

view = metrics.shortlist(df, PERIOD_KEYS[per_label], float(threshold), states or None, query)
if view.empty:
    st.info("No districts match the current filters.")
    st.stop()

n_above = int(view["above"].sum())
c1, c2, c3 = st.columns(3)
c1.metric("Shortlisted", f"{n_above} of {len(view)}", f"{100 * n_above / len(view):.0f}%", delta_color="off")
top, low = view.iloc[0], view.iloc[-1]
c2.metric("Highest 10-yr mean", f"{top['mean_mm']:.0f} mm", top["district"], delta_color="off")
c3.metric("Lowest 10-yr mean", f"{low['mean_mm']:.0f} mm", low["district"], delta_color="off")

tab_map, tab_table = st.tabs(["Map", "Ranked table"])
with tab_map:
    st.plotly_chart(figures.choropleth(geojson, view, float(threshold)), width="stretch")
with tab_table:
    table = figures.display_table(view)
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("Download CSV", table.to_csv(index=False).encode(),
                       file_name="plantation_planning_shortlist.csv", mime="text/csv")
    if src == "gsmap":
        st.caption("GSMaP: JAXA has no file for 2025-10-10; that day is excluded from the totals.")

years = sorted(c for c in df.columns if c.startswith("y") and c[1:].isdigit())
st.caption(f"Mean of annual totals ({per_label}), {years[0][1:]}-{years[-1][1:]}, from {src_label}. "
           "Each district value is an area-weighted mean over the grid cells inside it.")
