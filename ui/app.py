import os
import json
import requests
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# API base URL – can be overridden by environment variable API_URL or the sidebar
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_TIMEOUT_S = 120  # large trips take several seconds (features + classifier + HMM + metrics)

st.set_page_config(page_title="PathSnap – Map Matching UI", layout="wide")
st.title("PathSnap – Map Matching Explorer")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def fetch_trips(api_url: str):
    try:
        resp = requests.get(f"{api_url}/trips", timeout=10)
        resp.raise_for_status()
        return resp.json().get("trips", [])
    except Exception as e:
        st.error(f"Failed to fetch trips from API at {api_url}: {e}")
        return []

def predict_trip(api_url: str, trip_id: str):
    try:
        resp = requests.post(f"{api_url}/predict", json={"trip_id": trip_id}, timeout=PREDICT_TIMEOUT_S)
    except Exception as err:
        st.error(f"Failed to contact API at {api_url}: {err}")
        return None
    if resp.status_code == 404:
        st.error(f"Trip {trip_id} not found (404).")
        return None
    if not resp.ok:
        st.error(f"API error {resp.status_code}: {resp.text}")
        return None
    return resp.json()

def _gps_latlon(data: dict):
    """API returns GPS as [{'lat':..,'lon':..}]; routes as [[lat, lon], ...]."""
    out = []
    for p in data.get("gps", []):
        if isinstance(p, dict):
            out.append([p["lat"], p["lon"]])
        else:
            out.append([p[0], p[1]])
    return out

def build_folium_map(data: dict) -> folium.Map:
    gps = _gps_latlon(data)
    if gps:
        lats = [p[0] for p in gps]
        lons = [p[1] for p in gps]
        m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=13)
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    else:
        m = folium.Map(location=[0, 0], zoom_start=2)

    def add_layer(coords, color, name, weight):
        group = folium.FeatureGroup(name=name)
        if coords:
            folium.PolyLine(coords, color=color, weight=weight, opacity=0.8, tooltip=name).add_to(group)
        group.add_to(m)

    add_layer(data.get("reference_route", []), "green", "Reference Route", 7)
    add_layer(data.get("classifier_route", []), "orange", "Classifier Route", 4)
    add_layer(data.get("predicted_route", []), "red", "Predicted Route (HMM)", 4)
    add_layer(gps, "blue", "GPS Track", 2)

    legend_html = """
    <div style='position: fixed; bottom: 50px; left: 50px; width: 190px;
                border:2px solid grey; z-index:9999; font-size:14px; background-color:white;
                opacity:0.85; padding:10px;'>
      <b>Legend</b><br>
      <i style='color:blue'>●</i> GPS Track<br>
      <i style='color:red'>●</i> Predicted Route (HMM)<br>
      <i style='color:orange'>●</i> Classifier Route<br>
      <i style='color:green'>●</i> Reference Route
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m

def _fmt(metrics, method, key):
    value = (metrics or {}).get(method) or {}
    value = value.get(key)
    return f"{value:.3f}" if isinstance(value, (int, float)) else "-"

def _cells(points, n=5):
    cells = []
    for p in points[:n]:
        if isinstance(p, dict):
            cells.append(f"{p['lat']:.5f}, {p['lon']:.5f}")
        else:
            cells.append(f"{p[0]:.5f}, {p[1]:.5f}")
    return cells

# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
# API URL can be changed in the sidebar; a change (or the reload button) re-fetches the trips.
api_url = st.sidebar.text_input("API URL", value=st.session_state.get("api_url", DEFAULT_API_URL)).rstrip("/")
reload_clicked = st.sidebar.button("Reload trips")
if reload_clicked or st.session_state.get("api_url") != api_url or not st.session_state.get("trips"):
    st.session_state.api_url = api_url
    st.session_state.trips = fetch_trips(api_url)
    st.session_state.pop("result", None)

if not st.session_state.trips:
    st.info("No trips loaded. Check that the API is running and reachable, then press 'Reload trips'.")
    st.stop()

st.subheader("Available Trips")
cols = st.columns(5)
for idx, trip_id in enumerate(st.session_state.trips):
    col = cols[idx % len(cols)]
    if col.button(trip_id, key=f"trip_{trip_id}"):
        with st.spinner(f"Running the pipeline on trip {trip_id} ..."):
            st.session_state.result = predict_trip(api_url, trip_id)
        st.session_state.selected_trip = trip_id

# Results are kept in session_state so map interactions (which rerun the script) do not clear them.
result = st.session_state.get("result")
if result:
    st.success(f"Selected trip: {result.get('trip_id', st.session_state.get('selected_trip'))}")
    metrics = result.get("metrics", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Classifier Top‑1", _fmt(metrics, "classifier", "top1_accuracy"))
    c1.metric("Classifier Continuity", _fmt(metrics, "classifier", "route_continuity"))
    c1.metric("Classifier Route F1", _fmt(metrics, "classifier", "seq_f1"))
    c2.metric("HMM Top‑1", _fmt(metrics, "hmm", "top1_accuracy"))
    c2.metric("HMM Continuity", _fmt(metrics, "hmm", "route_continuity"))
    c2.metric("HMM Route F1", _fmt(metrics, "hmm", "seq_f1"))
    c3.metric("Latency (ms)", result.get("latency_ms", "-"))
    c3.metric("GPS points", len(result.get("gps", [])))
    if metrics.get("hmm") is None:
        st.warning("No reference route on the server for this trip, so accuracy metrics are not available.")

    st.subheader("Map Preview")
    fmap = build_folium_map(result)
    st_folium(fmap, width=1000, height=600, returned_objects=[])

    # Comparison table (first 5 points); columns of different lengths are padded with blanks
    st.subheader("Route Comparison (first 5 points each)")
    columns = {
        "GPS": _cells(result.get("gps", [])),
        "Predicted": _cells(result.get("predicted_route", [])),
        "Classifier": _cells(result.get("classifier_route", [])),
        "Reference": _cells(result.get("reference_route", [])),
    }
    df = pd.DataFrame({k: pd.Series(v, dtype=object) for k, v in columns.items()}).fillna("")
    st.table(df)

    with st.expander("Show raw JSON response"):
        st.code(json.dumps(result, indent=2), language="json")
