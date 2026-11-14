#!/usr/bin/env python3
"""RSS multilateration of jammer positions from fleet CN0 drops.

Given 3 moving tractors with mean_cn0_dbhz telemetry at 1 Hz, fit a static
jammer position (lat, lon), path-loss exponent n, and nuisance constant K
by minimizing residuals of the log-distance model

    predicted_drop(t, i) = 10 * n * log10( d(t, i) / 1 m ) + K   [capped: see below]

against observed CN0 drops. Reports per-event fit plus a 1 sigma ellipse
derived from the Jacobian at the optimum.

Usage:
    starter/.venv/bin/python starter/submission/localize.py
Writes starter/submission/localization_report.json.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "localization_report.json"

LAT_M_PER_DEG = 111_320.0
LON_M_PER_DEG_AT_47N = math.cos(math.radians(46.907)) * 111_320.0  # ~76,080


def load_tele() -> pd.DataFrame:
    df = pd.read_csv(DATA / "farm/tractor_telemetry.csv.gz", parse_dates=["ts"])
    df = df[["ts", "tractor_id", "lat", "lon", "mean_cn0_dbhz", "speed_mps"]]
    return df


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def event_fit(tele: pd.DataFrame,
              base_start: str, base_end: str,
              ev_start: str, ev_end: str,
              drop_floor_db: float = 2.0,
              drop_cap_db: float = 25.0):
    base_ts_s = pd.Timestamp(base_start, tz="UTC")
    base_ts_e = pd.Timestamp(base_end, tz="UTC")
    ev_ts_s = pd.Timestamp(ev_start, tz="UTC")
    ev_ts_e = pd.Timestamp(ev_end, tz="UTC")

    baselines = {}
    for tr, sub in tele.groupby("tractor_id"):
        base = sub[(sub["ts"] >= base_ts_s) & (sub["ts"] < base_ts_e)]
        # Use samples where tractor is moving so parked periods don't skew
        base = base[base["speed_mps"] > 1.0]
        if len(base) < 60:
            # Fall back to all baseline samples
            base = sub[(sub["ts"] >= base_ts_s) & (sub["ts"] < base_ts_e)]
        baselines[tr] = float(base["mean_cn0_dbhz"].median())

    ev = tele[(tele["ts"] >= ev_ts_s) & (tele["ts"] < ev_ts_e)].copy()
    ev = ev[ev["speed_mps"] > 1.0]
    ev["baseline"] = ev["tractor_id"].map(baselines)
    ev["drop_db"] = ev["baseline"] - ev["mean_cn0_dbhz"]
    # Keep only samples with meaningful drop; cap extreme values (receiver saturation / near-field)
    fit_rows = ev[(ev["drop_db"] >= drop_floor_db) & (ev["drop_db"] <= drop_cap_db)].copy()

    tractor_stats = {}
    for tr, sub in ev.groupby("tractor_id"):
        sub_fit = sub[(sub["drop_db"] >= drop_floor_db) & (sub["drop_db"] <= drop_cap_db)]
        tractor_stats[tr] = {
            "baseline_cn0_dbhz": round(baselines[tr], 2),
            "mean_drop_db": round(float(sub["drop_db"].mean()), 2),
            "max_drop_db": round(float(sub["drop_db"].max()), 2),
            "n_samples_fit": int(len(sub_fit)),
        }

    if len(fit_rows) < 20:
        return {"error": "insufficient samples", "n_samples": int(len(fit_rows)),
                "tractor_stats": tractor_stats, "baselines": baselines}

    lats = fit_rows["lat"].to_numpy()
    lons = fit_rows["lon"].to_numpy()
    drops = fit_rows["drop_db"].to_numpy()

    # Centroid as initial guess
    c_lat = float(fit_rows["lat"].mean())
    c_lon = float(fit_rows["lon"].mean())

    # Parameterize lat/lon in METERS from a local reference so the optimizer
    # scales all parameters similarly and the Jacobian comes out cleanly.
    def pos_to_latlon(dn_m, de_m):
        return c_lat + dn_m / LAT_M_PER_DEG, c_lon + de_m / LON_M_PER_DEG_AT_47N

    def resid_vec(params):
        dn_m, de_m, n, K = params
        lat_j, lon_j = pos_to_latlon(dn_m, de_m)
        d = np.clip(haversine_m(lats, lons, lat_j, lon_j), 1.0, None)
        # Log-distance path loss: drop decreases with distance.
        # K absorbs ERP minus noise floor (~drop at d = 1 m).
        pred = K - 10.0 * n * np.log10(d)
        return pred - drops

    # n constrained to a physical range for low-altitude terrestrial propagation
    # (2.0 = free space, 4.0 = heavy obstruction). Beyond 4 is unphysical.
    # K is drop at d=1m, physically positive (and bounded by receiver dynamic range).
    bounds_lo = np.array([-5000.0, -5000.0, 2.0,   0.0])
    bounds_hi = np.array([ 5000.0,  5000.0, 4.0, 200.0])

    best = None
    # Multi-start to dodge local minima
    starts = [(0, 0), (500, 0), (-500, 0), (0, 500), (0, -500),
              (800, 800), (-800, -800), (0, 1500), (0, -1500),
              (1500, 0), (-1500, 0), (2000, 1000)]
    for dn0, de0 in starts:
        x_init = np.array([dn0, de0, 2.5, 90.0])
        x_init = np.clip(x_init, bounds_lo, bounds_hi)
        try:
            r = least_squares(resid_vec, x_init, bounds=(bounds_lo, bounds_hi),
                              method="trf", max_nfev=400)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r

    dn_hat, de_hat, n_hat, K_hat = best.x
    lat_j, lon_j = pos_to_latlon(dn_hat, de_hat)
    rss = float(2.0 * best.cost)  # cost is 0.5 * sum(resid^2)
    n_samples = len(drops)
    sigma2 = rss / max(1, n_samples - 4)

    # Covariance of ALL parameters from the Jacobian least_squares returned
    J = best.jac  # shape (n_samples, 4), partial derivatives wrt (dn_m, de_m, n, K)
    try:
        cov_full = sigma2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov_full = None

    sigma_n_param = None
    sigma_K_param = None
    sigma_major_m = sigma_minor_m = rot_deg = float("nan")
    if cov_full is not None and np.all(np.isfinite(cov_full)):
        # Position covariance in (m_N, m_E) is already the top-left 2x2 block
        cov_pos = cov_full[:2, :2]
        eigvals, eigvecs = np.linalg.eigh(cov_pos)
        eigvals = np.clip(eigvals, 0.0, None)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
        sigma_major_m = float(np.sqrt(eigvals[0]))
        sigma_minor_m = float(np.sqrt(eigvals[1]))
        maj_vec = eigvecs[:, 0]  # [dN, dE]
        rot_deg = float(math.degrees(math.atan2(maj_vec[1], maj_vec[0])))
        sigma_n_param = float(math.sqrt(max(0.0, cov_full[2, 2])))
        sigma_K_param = float(math.sqrt(max(0.0, cov_full[3, 3])))

    at_n_upper_bound = n_hat >= bounds_hi[2] - 1e-3
    at_n_lower_bound = n_hat <= bounds_lo[2] + 1e-3

    return {
        "window": [ev_start.replace(" ", "T") + "Z", ev_end.replace(" ", "T") + "Z"],
        "n_samples": int(n_samples),
        "rss": round(rss, 3),
        "rmse_db": round(float(math.sqrt(sigma2)), 3),
        "fit": {
            "lat": round(float(lat_j), 6),
            "lon": round(float(lon_j), 6),
            "n_path_loss": round(float(n_hat), 3),
            "K": round(float(K_hat), 3),
            "n_at_upper_bound": bool(at_n_upper_bound),
            "n_at_lower_bound": bool(at_n_lower_bound),
        },
        "uncertainty": {
            "sigma_major_m": round(sigma_major_m, 1) if not math.isnan(sigma_major_m) else None,
            "sigma_minor_m": round(sigma_minor_m, 1) if not math.isnan(sigma_minor_m) else None,
            "rotation_deg": round(rot_deg, 1) if not math.isnan(rot_deg) else None,
            "sigma_n": round(sigma_n_param, 3) if sigma_n_param is not None else None,
        },
        "tractor_stats": tractor_stats,
        "baselines_cn0_dbhz": {k: round(v, 2) for k, v in baselines.items()},
    }


def sensitivity_sweep(tele: pd.DataFrame, base_start, base_end, ev_start, ev_end,
                      n_values=(2.0, 2.5, 3.0, 3.5, 4.0)):
    """Refit position with n pinned at each of several values. Reports the
    spread of fitted positions as an honest robustness check — the formal
    1-sigma ellipse assumes n is known, which it isn't."""
    base_ts_s = pd.Timestamp(base_start, tz="UTC")
    base_ts_e = pd.Timestamp(base_end, tz="UTC")
    ev_ts_s = pd.Timestamp(ev_start, tz="UTC")
    ev_ts_e = pd.Timestamp(ev_end, tz="UTC")

    baselines = {}
    for tr, sub in tele.groupby("tractor_id"):
        base = sub[(sub["ts"] >= base_ts_s) & (sub["ts"] < base_ts_e)]
        base = base[base["speed_mps"] > 1.0]
        baselines[tr] = float(base["mean_cn0_dbhz"].median())

    ev = tele[(tele["ts"] >= ev_ts_s) & (tele["ts"] < ev_ts_e)].copy()
    ev = ev[ev["speed_mps"] > 1.0]
    ev["baseline"] = ev["tractor_id"].map(baselines)
    ev["drop_db"] = ev["baseline"] - ev["mean_cn0_dbhz"]
    fit_rows = ev[(ev["drop_db"] >= 2.0) & (ev["drop_db"] <= 25.0)]
    lats = fit_rows["lat"].to_numpy()
    lons = fit_rows["lon"].to_numpy()
    drops = fit_rows["drop_db"].to_numpy()
    c_lat = float(fit_rows["lat"].mean())
    c_lon = float(fit_rows["lon"].mean())

    out = []
    for n_fixed in n_values:
        def resid(p):
            dn_m, de_m, K = p
            lat_j = c_lat + dn_m / LAT_M_PER_DEG
            lon_j = c_lon + de_m / LON_M_PER_DEG_AT_47N
            d = np.clip(haversine_m(lats, lons, lat_j, lon_j), 1.0, None)
            return (K - 10.0 * n_fixed * np.log10(d)) - drops
        best = None
        for dn0, de0 in [(0, 0), (500, 0), (-500, 0), (0, 500), (0, -500),
                         (1200, 1200), (-1200, -1200), (0, 1500), (0, -1500)]:
            r = least_squares(resid, [dn0, de0, 90.0],
                              bounds=([-5000, -5000, 0], [5000, 5000, 200]),
                              method="trf", max_nfev=300)
            if best is None or r.cost < best.cost:
                best = r
        dn, de, K_fit = best.x
        lat_fit = c_lat + dn / LAT_M_PER_DEG
        lon_fit = c_lon + de / LON_M_PER_DEG_AT_47N
        rmse = float(math.sqrt(2 * best.cost / max(1, len(drops) - 3)))
        out.append({
            "n_path_loss": n_fixed,
            "lat": round(lat_fit, 6),
            "lon": round(lon_fit, 6),
            "K": round(K_fit, 3),
            "rmse_db": round(rmse, 3),
        })
    # Spread of fitted positions across n values
    lat_vals = [p["lat"] for p in out]
    lon_vals = [p["lon"] for p in out]
    centroid_lat = sum(lat_vals) / len(lat_vals)
    centroid_lon = sum(lon_vals) / len(lon_vals)
    max_dev_m = max(
        float(haversine_m(centroid_lat, centroid_lon, p["lat"], p["lon"]))
        for p in out
    )
    return {"per_n": out, "centroid_lat": round(centroid_lat, 6),
            "centroid_lon": round(centroid_lon, 6),
            "max_deviation_m": round(max_dev_m, 1)}


def main():
    tele = load_tele()

    # Event A: 18:30 (no SDA report)
    ev_a = event_fit(
        tele,
        base_start="2026-05-06 16:00:00",
        base_end="2026-05-06 18:00:00",
        ev_start="2026-05-06 18:30:00",
        ev_end="2026-05-06 19:00:00",
    )
    ev_a["label"] = "18:30 fleet jamming (no SDA report)"
    ev_a["rfi_report"] = None
    ev_a["n_sensitivity"] = sensitivity_sweep(
        tele, "2026-05-06 16:00:00", "2026-05-06 18:00:00",
        "2026-05-06 18:30:00", "2026-05-06 19:00:00")

    # Event B: 02:12 (RFI-003)
    ev_b = event_fit(
        tele,
        base_start="2026-05-06 00:00:00",
        base_end="2026-05-06 02:00:00",
        ev_start="2026-05-06 02:12:00",
        ev_end="2026-05-06 02:54:00",
    )
    ev_b["label"] = "02:12 fleet jamming (RFI-003 corroborated)"
    ev_b["n_sensitivity"] = sensitivity_sweep(
        tele, "2026-05-06 00:00:00", "2026-05-06 02:00:00",
        "2026-05-06 02:12:00", "2026-05-06 02:54:00")

    rfi_003 = None
    rfi_path = DATA / "sda/rfi_reports.json"
    if rfi_path.exists():
        rfi_obj = json.loads(rfi_path.read_text())
        rfi_list = rfi_obj.get("reports", rfi_obj) if isinstance(rfi_obj, dict) else rfi_obj
        for r in rfi_list:
            if r.get("id") == "RFI-003":
                rfi_003 = r
                break
    if rfi_003:
        r_lat = rfi_003.get("centerLat") or rfi_003.get("lat")
        r_lon = rfi_003.get("centerLon") or rfi_003.get("lon")
        d_m = float(haversine_m(
            ev_b["fit"]["lat"], ev_b["fit"]["lon"], r_lat, r_lon
        ))
        ev_b["rfi_report"] = {
            "id": "RFI-003",
            "lat": r_lat,
            "lon": r_lon,
            "semi_major_km": rfi_003.get("semiMajor_km") or rfi_003.get("semi_major_km"),
            "semi_minor_km": rfi_003.get("semiMinor_km") or rfi_003.get("semi_minor_km"),
            "distance_to_our_estimate_m": round(d_m, 1),
        }
    else:
        ev_b["rfi_report"] = None

    report = {"events": [ev_a, ev_b]}
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
