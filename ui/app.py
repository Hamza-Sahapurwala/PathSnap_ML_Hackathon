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
# API base URL – can be overridden by environment variable API_URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="PathSnap – Map Matching UI", layout="wide")
st.title("PathSnap – Map Matching Explorer")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def fetch_trips():
    try:
        resp = requests.get(f"{API_URL}/trips", timeout=5)
        resp.raise_for_status()
        return resp.json().get("trips", [])
    except Exception as e:
        st.error(f"Failed to fetch trips from API: {e}")
        return []

def predict_trip(trip_id: str):
    try:
        resp = requests.post(f"{API_URL}/predict", json={"trip_id": trip_id}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError:
        if resp.status_code == 404:
            st.error(f"Trip {trip_id} not found (404).")
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
    except Exception as err:
        st.error(f"Failed to contact API: {err}")
    return None

def build_folium_map(data: dict) -> folium.Map:
    # Center map on GPS points (average)
    gps = data.get("gps", [])
    if gps:
        avg_lat = sum(p[0] for p in gps) / len(gps)
        avg_lon = sum(p[1] for p in gps) / len(gps)
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)
    else:
        m = folium.Map(location=[0, 0], zoom_start=2)

    def add_layer(coords, color, name):
        if coords:
            folium.PolyLine(coords, color=color, weight=3, opacity=0.8, tooltip=name).add_to(m)

    add_layer(data.get("gps", []), "blue", "GPS Track")
    add_layer(data.get("predicted_route", []), "red", "Predicted Route (HMM)")
    add_layer(data.get("classifier_route", []), "orange", "Classifier Route")
    add_layer(data.get("reference_route", []), "green", "Reference Route")

    # Legend via HTML overlay
    legend_html = """
    <div style='position: fixed; bottom: 50px; left: 50px; width: 150px; height: 120px; 
                border:2px solid grey; z-index:9999; font-size:14px; background-color:white; 
                opacity:0.8; padding:10px;'>
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

# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
# Allow user to edit API URL via sidebar
api_url_input = st.sidebar.text_input("API URL", value=API_URL)
if api_url_input != API_URL:
    API_URL = api_url_input

# Load trips once per session
if "trips" not in st.session_state:
    st.session_state.trips = fetch_trips()

if not st.session_state.trips:
    st.stop()

st.subheader("Available Trips")
cols = st.columns(5)
selected_trip = None
for idx, trip_id in enumerate(st.session_state.trips):
    col = cols[idx % len(cols)]
    if col.button(trip_id, key=f"trip_{trip_id}"):
        selected_trip = trip_id

if selected_trip:
    st.success(f"Selected trip: {selected_trip}")
    result = predict_trip(selected_trip)
    if result:
        # Metrics cards
        metrics = result.get("metrics", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Classifier Top‑1", metrics.get("classifier", {}).get("top1", "-"))
        c1.metric("Classifier Continuity", metrics.get("classifier", {}).get("continuity", "-"))
        c2.metric("HMM Top‑1", metrics.get("hmm", {}).get("top1", "-"))
        c2.metric("HMM Continuity", metrics.get("hmm", {}).get("continuity", "-"))
        c3.metric("Latency (ms)", result.get("latency_ms", "-"))

        # Map display
        st.subheader("Map Preview")
        fmap = build_folium_map(result)
        st_folium(fmap, width=800, height=600)

        # Comparison table (first 5 points)
        st.subheader("Route Comparison (first 5 points each)")
        df = pd.DataFrame({
            "GPS": result.get("gps", [])[:5],
            "Predicted": result.get("predicted_route", [])[:5],
            "Classifier": result.get("classifier_route", [])[:5],
            "Reference": result.get("reference_route", [])[:5],
        })
        st.table(df)

        # Collapsible raw JSON
        with st.expander("Show raw JSON response"):
            st.code(json.dumps(result, indent=2), language="json")
