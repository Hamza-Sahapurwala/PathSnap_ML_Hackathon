"""Member C: Folium maps comparing raw GPS, the reference route and predicted routes.

Layers (each can be toggled in the layer control):

* road network near the GPS track (grey; spatially filtered so large networks stay light)
* raw GPS observations (blue; hover shows index, timestamp and each method's arc)
* reference route (green; only when a reference is supplied: evaluation maps, never the API)
* classifier-only route (orange) and HMM route(s) (red; an untuned HMM is purple, dashed)
* mismatched observations per method (points whose predicted arc differs from the reference)

Routes are drawn in travel order from node coordinates; a break is left wherever two
consecutive arcs do not meet, so a disconnected prediction is visible as a gap.

Usage (from the repository root, after ``python -m src.hmm_experiments``)::

    python -m src.make_map --trip 00000077
"""
from __future__ import annotations

import argparse
import html
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import folium
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.hmm import NO_ARC, RoadGraph, fill_route, route_from_arcs

ROOT = Path(__file__).resolve().parents[1]

STYLE = {  # layer name -> (colour, weight, dash)
    "classifier_top1": ("#ff7f0e", 5, None),
    "hmm": ("#d62728", 4, None),
    "hmm_tuned": ("#d62728", 4, None),
    "hmm_untuned": ("#9467bd", 3, "6 6"),
}
LABEL = {"classifier_top1": "Classifier-only route", "hmm": "HMM route",
         "hmm_tuned": "HMM route (tuned)", "hmm_untuned": "HMM route (untuned)"}
GPS_COLOUR, NETWORK_COLOUR, REFERENCE_COLOUR = "#1f77b4", "#8c8c8c", "#2ca02c"


def _latlon(lonlat: np.ndarray, nodes: Sequence[int]) -> List[List[float]]:
    return [[round(float(lonlat[n, 1]), 6), round(float(lonlat[n, 0]), 6)] for n in nodes]


def route_lines(route: Sequence[int], u: np.ndarray, v: np.ndarray) -> List[List[int]]:
    """Node sequences of the route, split wherever consecutive arcs do not meet."""
    lines: List[List[int]] = []
    for a in route:
        p, q = int(u[a]), int(v[a])
        if lines and lines[-1][-1] == p:
            lines[-1].append(q)
        else:
            lines.append([p, q])
    return lines


def nearby_arcs(track_xy: np.ndarray, graph: RoadGraph, buffer_m: float, max_arcs: int) -> np.ndarray:
    """Arcs whose midpoint lies within ``buffer_m`` of a GPS point (nearest first if capped)."""
    ok = np.isfinite(track_xy).all(axis=1)
    if not ok.any() or graph.n_arcs == 0:
        return np.empty(0, dtype=np.int64)
    mid = np.column_stack([(graph.nodes_x[graph.u] + graph.nodes_x[graph.v]) / 2,
                           (graph.nodes_y[graph.u] + graph.nodes_y[graph.v]) / 2])
    dist, _ = cKDTree(track_xy[ok]).query(mid, distance_upper_bound=buffer_m)
    keep = np.flatnonzero(np.isfinite(dist))
    if keep.size > max_arcs:
        keep = keep[np.argsort(dist[keep], kind="stable")[:max_arcs]]
    return np.sort(keep)


def _legend(title: str, entries: Sequence[Tuple[str, str, str]], notes: Sequence[str]) -> str:
    rows = "".join(
        f'<div><span style="display:inline-block;width:22px;height:0;border-top:4px {dash} {colour};'
        f'vertical-align:middle;margin-right:6px"></span>{html.escape(text)}</div>'
        for text, colour, dash in entries)
    extra = "".join(f'<div style="color:#444">{html.escape(n)}</div>' for n in notes)
    return ('<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:rgba(255,255,255,.93);'
            'padding:8px 10px;border:1px solid #999;border-radius:4px;font:12px/1.5 Arial,sans-serif;max-width:340px">'
            f'<div style="font-weight:bold;margin-bottom:4px">{html.escape(title)}</div>{rows}'
            f'<div style="margin-top:4px">{extra}</div></div>')


def build_trip_map(trip: Mapping[str, Any], predictions: Optional[Mapping[str, Sequence[int]]] = None,
                   reference_route: Optional[Sequence[int]] = None,
                   reference_point_arcs: Optional[Sequence[int]] = None,
                   title: Optional[str] = None, notes: Sequence[str] = (),
                   network_buffer_m: float = 250.0, max_network_arcs: int = 15_000,
                   fill: bool = True) -> folium.Map:
    """Interactive map of one trip.

    ``trip`` is a dict from ``src.preprocess.load_trip_data`` (``track_raw``, ``nodes_raw`` are lon/lat).
    ``predictions`` maps a method name to its predicted arc per GPS point (``-1`` = none).
    ``reference_route`` / ``reference_point_arcs`` are optional ground truth (evaluation maps only).
    """
    predictions = dict(predictions or {})
    graph = RoadGraph.from_trip(trip)
    nodes_ll = np.asarray(trip["nodes_raw"], dtype=np.float64)[:, :2]
    track = np.asarray(trip["track_raw"], dtype=np.float64)
    n_points = len(track)
    for name, arcs in predictions.items():
        if len(arcs) != n_points:
            raise ValueError(f"predictions[{name!r}] has {len(arcs)} entries for {n_points} GPS points")

    finite = np.isfinite(track[:, :2]).all(axis=1)
    lat, lon = track[finite, 1], track[finite, 0]
    fmap = folium.Map(location=[float(lat.mean()), float(lon.mean())], tiles=None,
                      control_scale=True, prefer_canvas=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap (faded)", opacity=0.5).add_to(fmap)
    fmap.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]],
                    padding=(40, 40))
    entries: List[Tuple[str, str, str]] = []

    # road network near the track
    track_xy = np.column_stack([trip["track_x"], trip["track_y"]]).astype(np.float64)
    near = nearby_arcs(track_xy, graph, network_buffer_m, max_network_arcs)
    network = folium.FeatureGroup(name=f"Road network ({near.size:,} of {graph.n_arcs:,} arcs within "
                                       f"{network_buffer_m:.0f} m)", show=True)
    if near.size:
        folium.PolyLine([_latlon(nodes_ll, [graph.u[a], graph.v[a]]) for a in near],
                        color=NETWORK_COLOUR, weight=1.5, opacity=0.6).add_to(network)
    network.add_to(fmap)
    entries.append(("Road network", NETWORK_COLOUR, "solid"))

    # raw GPS observations (below the routes so the lines stay readable)
    ref_pts = None if reference_point_arcs is None else np.asarray(reference_point_arcs, dtype=np.int64)
    gps = folium.FeatureGroup(name=f"Raw GPS ({int(finite.sum()):,} points)", show=True)
    for i in np.flatnonzero(finite):
        parts = [f"point {i}", f"t = {track[i, 2]:.0f} s"]
        if ref_pts is not None:
            parts.append(f"reference arc {ref_pts[i]}")
        for name, arcs in predictions.items():
            a = int(arcs[i])
            mark = "" if ref_pts is None else (" ✓" if a == ref_pts[i] else " ✗")
            parts.append(f"{LABEL.get(name, name).replace(' route', '')}: arc {a}{mark}")
        folium.CircleMarker([float(track[i, 1]), float(track[i, 0])], radius=2, color=GPS_COLOUR,
                            fill=True, fill_opacity=0.7, weight=0, tooltip=" | ".join(parts)).add_to(gps)
    gps.add_to(fmap)
    entries.insert(0, ("Raw GPS observations", GPS_COLOUR, "dotted"))

    # reference route
    if reference_route is not None and len(reference_route):
        ref = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine([_latlon(nodes_ll, line) for line in route_lines(reference_route, graph.u, graph.v)],
                        color=REFERENCE_COLOUR, weight=8, opacity=0.55,
                        tooltip=f"Reference route ({len(reference_route)} arcs)").add_to(ref)
        ref.add_to(fmap)
        entries.append(("Reference route", REFERENCE_COLOUR, "solid"))

    # predicted routes (classifier-only first so the HMM is drawn on top)
    for name in sorted(predictions, key=lambda k: (k != "classifier_top1", k == "hmm_tuned")):
        route = route_from_arcs(predictions[name])
        breaks = 0
        if fill:
            route, breaks = fill_route(route, graph)
        colour, weight, dash = STYLE.get(name, ("#000000", 3, None))
        label = LABEL.get(name, name)
        group = folium.FeatureGroup(name=label, show=True)
        if route:
            folium.PolyLine([_latlon(nodes_ll, line) for line in route_lines(route, graph.u, graph.v)],
                            color=colour, weight=weight, opacity=0.9, dash_array=dash,
                            tooltip=f"{label}: {len(route)} arcs, {breaks} unconnected gap(s)").add_to(group)
        group.add_to(fmap)
        entries.append((label, colour, "dashed" if dash else "solid"))

    # mismatched observations
    if ref_pts is not None:
        for name, arcs in predictions.items():
            arcs = np.asarray(arcs, dtype=np.int64)
            wrong = np.flatnonzero(finite & (ref_pts != NO_ARC) & (arcs != ref_pts))
            colour = STYLE.get(name, ("#000000",))[0]
            group = folium.FeatureGroup(name=f"Mismatches: {LABEL.get(name, name).replace(' route', '')} "
                                             f"({wrong.size:,})", show=name != "classifier_top1")
            for i in wrong:
                folium.CircleMarker([float(track[i, 1]), float(track[i, 0])], radius=6, color=colour, weight=2,
                                    fill=False, tooltip=f"point {i}: predicted arc {arcs[i]}, "
                                                        f"reference arc {ref_pts[i]}").add_to(group)
            group.add_to(fmap)

    for i, label in ((int(np.flatnonzero(finite)[0]), "Start"), (int(np.flatnonzero(finite)[-1]), "End")):
        folium.Marker([float(track[i, 1]), float(track[i, 0])], tooltip=f"{label} (point {i}, t = {track[i, 2]:.0f} s)",
                      icon=folium.Icon(color="green" if label == "Start" else "black", icon="flag")).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(
        _legend(title or f"Trip {trip.get('trip_id', '')}", entries, notes)))
    return fmap


def screenshot(html_path: Path, png_path: Path, size: Tuple[int, int] = (1600, 1100)) -> bool:
    """PNG via headless Chrome or Edge (needs internet for map tiles). Returns False if no browser."""
    candidates = [shutil.which(b) for b in ("chrome", "msedge", "google-chrome", "chromium")]
    candidates += [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                   r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
    browser = next((c for c in candidates if c and Path(c).exists()), None)
    if browser is None:
        return False
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={size[0]},{size[1]}", "--virtual-time-budget=15000",
                    f"--screenshot={png_path.resolve()}", html_path.resolve().as_uri()],
                   check=True, capture_output=True, timeout=120)
    return png_path.exists()


def main(argv: Optional[Sequence[str]] = None) -> None:
    from src.hmm_experiments import read_reference_route
    from src.preprocess import load_trip_data

    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--trip", required=True, nargs="+")
    args.add_argument("--partition", default="student_test", choices=["student_test", "train"])
    args.add_argument("--predictions", type=Path, default=ROOT / "results" / "hmm_point_predictions.csv")
    args.add_argument("--comparison", type=Path, default=ROOT / "results" / "hmm_comparison.csv")
    args.add_argument("--out-dir", type=Path, default=ROOT / "maps")
    args.add_argument("--png", action="store_true", help="also save a PNG screenshot (headless Chrome/Edge)")
    a = args.parse_args(argv)
    points = pd.read_csv(a.predictions, dtype={"trip_id": str})
    comparison = pd.read_csv(a.comparison, dtype={"trip_id": str}) if a.comparison.exists() else None
    methods = [c[:-len("_arc_id")] for c in points.columns
               if c.endswith("_arc_id") and c != "reference_arc_id"]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data" / a.partition
    for tid in a.trip:
        trip = load_trip_data(data_dir, tid, load_route=False)
        p = points[points.trip_id == tid].sort_values("point_idx")
        preds = {("classifier_top1" if m == "top1" else m): p[f"{m}_arc_id"].to_numpy() for m in methods}
        notes = []
        if comparison is not None:
            for _, r in comparison[comparison.trip_id == tid].iterrows():
                notes.append(f"{LABEL.get(r['method'], r['method']).replace(' route', '')}: Top-1 "
                             f"{r['top1_accuracy']:.3f}, route F1 {r['filled_seq_f1']:.3f}")
        fmap = build_trip_map(trip, preds, read_reference_route(data_dir, tid), p["reference_arc_id"].to_numpy(),
                              title=f"Trip {tid} ({a.partition}): classifier-only vs HMM", notes=notes)
        out = a.out_dir / f"hmm_{a.partition}_{tid}.html"
        fmap.save(str(out))
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
        if a.png:
            png = out.with_suffix(".png")
            print(f"  png: {png}" if screenshot(out, png) else "  png: no Chrome/Edge found")


if __name__ == "__main__":
    main()
