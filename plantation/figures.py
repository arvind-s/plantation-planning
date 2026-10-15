"""Plotly map and display table for the app."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

GREY = "#d9d9d9"


def choropleth(geojson: dict, view: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    common = dict(geojson=geojson, featureidkey="properties.key",
                  marker_line_width=0.4, marker_line_color="white")
    hover = "%{customdata[0]}, %{customdata[1]}<br>%{customdata[2]:.0f} mm<extra>%{customdata[3]}</extra>"
    for flag, label in ((False, "below threshold"), (True, "shortlisted")):
        part = view[view["above"] == flag]
        if part.empty:
            continue
        cd = pd.DataFrame({"d": part["district"], "s": part["state"],
                           "m": part["mean_mm"], "l": label}).values
        if flag:
            fig.add_trace(go.Choroplethmap(
                locations=part["key"], z=part["mean_mm"], colorscale="Blues",
                zmin=threshold, zmax=max(float(part["mean_mm"].max()), threshold + 1),
                colorbar=dict(title="10-yr mean (mm)"), customdata=cd,
                hovertemplate=hover, **common))
        else:
            fig.add_trace(go.Choroplethmap(
                locations=part["key"], z=[0] * len(part), showscale=False,
                colorscale=[[0, GREY], [1, GREY]], customdata=cd,
                hovertemplate=hover, **common))
    fig.update_layout(map_style="carto-positron", map_center={"lat": 16.0, "lon": 80.0},
                      map_zoom=4.2, margin=dict(l=0, r=0, t=0, b=0), height=560)
    return fig


def display_table(view: pd.DataFrame) -> pd.DataFrame:
    def notes(r):
        n = []
        if r["low_cells"]:
            n.append("few grid cells")
        if r["missing_days"] > 0:
            n.append("data gaps")
        return ", ".join(n)
    return pd.DataFrame({
        "State": view["state"], "District": view["district"],
        "10-yr mean (mm)": view["mean_mm"],
        "Shortlisted": view["above"].map({True: "Yes", False: "No"}),
        "Grid cells": view["n_cells"],
        "Notes": view.apply(notes, axis=1)}).reset_index(drop=True)
