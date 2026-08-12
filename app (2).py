"""
MetroFlow AI — Offline Congestion & Signal-Timing Advisor
--------------------------------------------------------
An OFFLINE academic-MVP analytics tool. It never touches a live signal
controller — it reads historical / synthetic traffic-flow data, scores
congestion with a transparent rule-based model, and proposes green-time
splits that a human traffic engineer would review before applying.

Run:
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

# ======================================================================
# GLOBAL CONSTANTS  (all assumptions live here, in one visible place)
# ======================================================================
SATURATION_FLOW_PER_LANE_PER_MIN = 30   # veh/min a single lane can discharge on green (~1800 veh/hr/lane)
BASE_CAPACITY_PER_LANE_PER_MIN = 20     # veh/min/lane — sustained comfortable capacity (not saturation)
DEFAULT_CYCLE_LENGTH_SEC = 90           # typical urban signal cycle
MIN_GREEN_SEC = 15
MAX_GREEN_FRACTION = 0.70               # a single approach can't hog more than 70% of the cycle
CONGESTION_BANDS = [(30, "Low"), (55, "Moderate"), (80, "High"), (101, "Severe")]
BAND_COLORS = {"Low": "#2BB673", "Moderate": "#F4B740", "High": "#F07B3F", "Severe": "#D64550"}

REQUIRED_COLUMNS = [
    "timestamp", "intersection", "road", "direction",
    "vehicles_per_min", "average_speed_kmh", "lane_count",
    "speed_limit_kmh", "turning_ratio", "incident",
]

# ======================================================================
# 1. DATA GENERATION  (synthetic demo data)
# ======================================================================
def generate_synthetic_data(seed: int = 42, period: str = "day") -> pd.DataFrame:
    """Builds realistic traffic data for 3 intersections, each with 4 approaches,
    including AM/PM peaks and a couple of incidents.

    period="day"   -> one 24h day at 5-minute resolution (detailed view)
    period="month" -> 30 days at hourly resolution, with weaker weekend demand
                       (monthly / historic-trend view)
    """
    rng = np.random.default_rng(seed)

    intersections = {
        "Elm & 5th":      {"lanes": {"North": 2, "South": 2, "East": 3, "West": 2}, "limit": 50},
        "Oak & Market":   {"lanes": {"North": 3, "South": 3, "East": 2, "West": 2}, "limit": 60},
        "Pine & Central": {"lanes": {"North": 2, "South": 2, "East": 2, "West": 3}, "limit": 50},
    }
    road_names = {"North": "Elm Ave N", "South": "Elm Ave S", "East": "5th St E", "West": "5th St W"}

    if period == "month":
        timestamps = pd.date_range("2026-07-13 00:00", periods=30 * 24, freq="1h")  # 30 days @ hourly
    else:
        timestamps = pd.date_range("2026-08-11 00:00", periods=24 * 12, freq="5min")  # 24h @ 5-min steps

    rows = []

    for x_name, meta in intersections.items():
        for direction, lanes in meta["lanes"].items():
            # each approach gets its own baseline + peak weighting so demand differs by road
            base = rng.uniform(6, 12)
            am_weight = rng.uniform(0.8, 1.6)
            pm_weight = rng.uniform(0.8, 1.6)

            for ts in timestamps:
                hour = ts.hour + ts.minute / 60
                am_peak = np.exp(-((hour - 8.5) ** 2) / (2 * 0.9 ** 2)) * 55 * am_weight
                pm_peak = np.exp(-((hour - 18.0) ** 2) / (2 * 1.1 ** 2)) * 60 * pm_weight
                midday = np.exp(-((hour - 13) ** 2) / (2 * 3 ** 2)) * 12

                # monthly data: quieter weekends + a little day-to-day drift
                weekend_factor = 1.0
                if period == "month" and ts.dayofweek >= 5:
                    weekend_factor = 0.65
                day_drift = 1.0 + 0.06 * np.sin(ts.dayofyear / 5.0 + rng.uniform(0, 3))

                noise = rng.normal(0, 3)
                vpm = max(2, (base + am_peak + pm_peak + midday) * weekend_factor * day_drift + noise)

                # speed drops as demand rises relative to capacity
                capacity = lanes * BASE_CAPACITY_PER_LANE_PER_MIN
                vc = vpm / capacity
                speed = meta["limit"] * max(0.25, 1 - 0.55 * min(vc, 1.6)) + rng.normal(0, 2)
                speed = max(5, speed)

                incident = ""
                if period == "month":
                    # sprinkle a handful of incidents across the month, roughly weekly
                    if x_name == "Oak & Market" and direction == "East" and hour == 9 and ts.day % 7 == 3:
                        incident = "roadwork"
                        vpm *= 0.6
                        speed *= 0.5
                    if x_name == "Pine & Central" and direction == "South" and hour == 18 and ts.day % 9 == 5:
                        incident = "accident"
                        vpm *= 0.5
                        speed *= 0.4
                else:
                    # One synthetic incident: roadwork on Oak & Market, East approach, 9-10 AM
                    if x_name == "Oak & Market" and direction == "East" and 9 <= hour < 10:
                        incident = "roadwork"
                        vpm *= 0.6   # roadwork throttles throughput past the point, but demand upstream still queues
                        speed *= 0.5
                    # One synthetic accident: Pine & Central, South approach, 17:30-18:00
                    if x_name == "Pine & Central" and direction == "South" and 17.5 <= hour < 18.0:
                        incident = "accident"
                        vpm *= 0.5
                        speed *= 0.4

                rows.append({
                    "timestamp": ts,
                    "intersection": x_name,
                    "road": road_names[direction],
                    "direction": direction,
                    "vehicles_per_min": round(vpm, 1),
                    "average_speed_kmh": round(speed, 1),
                    "lane_count": lanes,
                    "speed_limit_kmh": meta["limit"],
                    "turning_ratio": round(rng.uniform(0.05, 0.35), 2),
                    "incident": incident,
                })

    return pd.DataFrame(rows)


# ======================================================================
# 2. DATA LOADING & VALIDATION
# ======================================================================
def load_data(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    """Reads a user CSV, fills in any missing optional columns gracefully,
    and returns (dataframe, list_of_warnings)."""
    warnings = []
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return pd.DataFrame(), [str(e)]

    df.columns = [c.strip().lower() for c in df.columns]

    hard_required = ["timestamp", "intersection", "vehicles_per_min"]
    missing_hard = [c for c in hard_required if c not in df.columns]
    if missing_hard:
        st.error(f"CSV is missing required columns: {missing_hard}")
        return pd.DataFrame(), [f"Missing required columns: {missing_hard}"]

    soft_defaults = {
        "road": "Unnamed Road", "direction": "N/A", "average_speed_kmh": np.nan,
        "lane_count": 2, "speed_limit_kmh": 50, "turning_ratio": 0.15, "incident": "",
    }
    for col, default in soft_defaults.items():
        if col not in df.columns:
            df[col] = default
            warnings.append(f"Column '{col}' missing — filled with default ({default}).")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_ts = df["timestamp"].isna().sum()
    if bad_ts:
        warnings.append(f"{bad_ts} row(s) had unparseable timestamps and were dropped.")
        df = df.dropna(subset=["timestamp"])

    df["vehicles_per_min"] = pd.to_numeric(df["vehicles_per_min"], errors="coerce")
    df = df.dropna(subset=["vehicles_per_min"])
    if df["average_speed_kmh"].isna().all():
        df["average_speed_kmh"] = df["speed_limit_kmh"] * 0.8  # fallback estimate
        warnings.append("No usable speed data — estimated speed as 80% of speed limit.")

    df["incident"] = df["incident"].fillna("").astype(str)
    return df.reset_index(drop=True), warnings


# ======================================================================
# 3. PREPROCESSING
# ======================================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["hour_label"] = df["hour"].apply(lambda h: f"{h:02d}:00")
    return df


# ======================================================================
# 4. CONGESTION SCORING  (transparent, rule-based)
# ======================================================================
def estimate_capacity(lane_count: pd.Series) -> pd.Series:
    return lane_count * BASE_CAPACITY_PER_LANE_PER_MIN


def score_congestion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["capacity_veh_min"] = estimate_capacity(df["lane_count"])
    df["vc_ratio"] = (df["vehicles_per_min"] / df["capacity_veh_min"]).clip(upper=2.5)

    free_flow_speed = df["speed_limit_kmh"].replace(0, np.nan)
    speed_ratio = (df["average_speed_kmh"] / free_flow_speed).clip(0, 1.2)
    df["speed_reduction"] = (1 - speed_ratio).clip(0, 1)

    # weighted blend: demand pressure (70%) + speed degradation (30%), scaled to 0-100
    raw_score = df["vc_ratio"].clip(upper=1.4) / 1.4 * 70 + df["speed_reduction"] * 30
    df["congestion_score"] = raw_score.clip(0, 100).round(1)

    def band(score):
        for threshold, label in CONGESTION_BANDS:
            if score < threshold:
                return label
        return "Severe"

    df["congestion_level"] = df["congestion_score"].apply(band)
    return df


# ======================================================================
# 5. PEAK & BOTTLENECK DETECTION
# ======================================================================
def detect_peak_hours(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    hourly = df.groupby("hour_label", as_index=False)["congestion_score"].mean()
    return hourly.sort_values("congestion_score", ascending=False).head(top_n)


def detect_bottlenecks(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    grouped = df.groupby(["intersection", "road", "direction"], as_index=False).agg(
        avg_congestion=("congestion_score", "mean"),
        pct_severe=("congestion_level", lambda s: (s == "Severe").mean() * 100),
        avg_vc=("vc_ratio", "mean"),
        avg_speed=("average_speed_kmh", "mean"),
    )
    grouped = grouped.sort_values(["pct_severe", "avg_congestion"], ascending=False)
    return grouped.head(top_n)


# ======================================================================
# 6. SIGNAL TIMING RECOMMENDATION
# ======================================================================
def recommend_signal_split(demand_df: pd.DataFrame, cycle_length: int = DEFAULT_CYCLE_LENGTH_SEC) -> pd.DataFrame:
    """
    demand_df: one row per approach (direction) for the selected intersection & time window,
               must contain columns 'direction', 'road', 'avg_demand', 'lane_count', 'current_green'.
    Rule: green split is nudged toward each approach's share of total demand,
          bounded by MIN_GREEN_SEC and MAX_GREEN_FRACTION of the cycle.
    """
    d = demand_df.copy()
    total_demand = d["avg_demand"].sum()
    if total_demand <= 0:
        d["demand_share"] = 1 / len(d)
    else:
        d["demand_share"] = d["avg_demand"] / total_demand

    d["current_green_share"] = d["current_green"] / cycle_length

    # proposed raw green = demand share of the cycle
    raw_green = (d["demand_share"] * cycle_length).clip(
        lower=MIN_GREEN_SEC, upper=cycle_length * MAX_GREEN_FRACTION
    )

    # normalize so all approaches still sum to the fixed cycle length
    scale = cycle_length / raw_green.sum()
    d["recommended_green"] = (raw_green * scale).round().astype(int)
    d["change_sec"] = d["recommended_green"] - d["current_green"]

    def reason(row):
        gap = row["demand_share"] - row["current_green_share"]
        if row["change_sec"] > 2:
            return (f"{row['road']} carries ~{row['demand_share']*100:.0f}% of approach demand "
                     f"but only {row['current_green_share']*100:.0f}% of current green time — "
                     f"green split is under-allocated relative to demand.")
        elif row["change_sec"] < -2:
            return (f"{row['road']} carries only ~{row['demand_share']*100:.0f}% of approach demand "
                     f"vs {row['current_green_share']*100:.0f}% of current green time — "
                     f"green split can be trimmed to free up cycle time for busier approaches.")
        else:
            return f"{row['road']}'s current green split already roughly matches its demand share."

    d["reason"] = d.apply(reason, axis=1)
    return d


# ======================================================================
# 7. BEFORE / AFTER QUEUE SIMULATION  (simple deterministic model)
# ======================================================================
def simulate_approach(arrival_rate_veh_min: float, green_sec: int, cycle_sec: int,
                       lanes: int, sim_minutes: int = 60) -> dict:
    """Minute-by-minute deterministic queue build/drain simulation.
    NOT a stochastic microsimulation — a transparent estimate only."""
    capacity_per_min = lanes * SATURATION_FLOW_PER_LANE_PER_MIN * (green_sec / cycle_sec)
    queue = 0.0
    queue_history = []
    served_total = 0.0

    for _ in range(sim_minutes):
        queue += arrival_rate_veh_min
        served = min(queue, capacity_per_min)
        queue -= served
        served_total += served
        queue_history.append(queue)

    avg_queue = float(np.mean(queue_history))
    avg_wait_sec = (avg_queue / arrival_rate_veh_min) * 60 if arrival_rate_veh_min > 0 else 0.0
    throughput_per_hr = served_total / sim_minutes * 60
    vc_effective = arrival_rate_veh_min / capacity_per_min if capacity_per_min > 0 else np.nan

    return {
        "avg_queue": round(avg_queue, 1),
        "avg_wait_sec": round(avg_wait_sec, 1),
        "throughput_per_hr": round(throughput_per_hr, 1),
        "vc_effective": round(vc_effective, 2),
        "capacity_per_min": round(capacity_per_min, 1),
    }


def congestion_score_from_vc(vc_ratio: float, baseline_speed_reduction: float, vc_before: float) -> float:
    """Rescales the speed-reduction component proportionally to the change in v/c ratio,
    since we don't have real post-change speed measurements (simulated estimate only)."""
    if vc_before <= 0:
        scaled_reduction = baseline_speed_reduction
    else:
        scaled_reduction = np.clip(baseline_speed_reduction * (vc_ratio / vc_before), 0, 1)
    score = np.clip(vc_ratio.clip(0, 1.4) / 1.4 * 70 if hasattr(vc_ratio, "clip") else min(vc_ratio, 1.4) / 1.4 * 70, 0, 70)
    score += scaled_reduction * 30
    return round(float(np.clip(score, 0, 100)), 1)


# ======================================================================
# 8. PDF REPORT GENERATION
# ======================================================================
def build_recommendation_report(fdf: pd.DataFrame, cycle_length: int, data_source: str) -> bytes:
    """Builds a PDF summarizing recommended signal-timing changes for every
    intersection in the (filtered) dataset. Returns raw PDF bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], textColor=colors.HexColor("#0B2545"))
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#134074"),
                               spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=13)
    note_style = ParagraphStyle("Note", parent=styles["Normal"], fontSize=8.5, leading=11,
                                 textColor=colors.HexColor("#5A6B85"))

    story = []
    story.append(Paragraph("MetroFlow AI — Recommended Signal-Timing Changes", title_style))
    story.append(Spacer(1, 4))
    date_range = f"{fdf['date'].min()} to {fdf['date'].max()}"
    story.append(Paragraph(
        f"Data source: {data_source} &nbsp;|&nbsp; Period covered: {date_range} &nbsp;|&nbsp; "
        f"Assumed cycle length: {cycle_length} sec", note_style
    ))
    story.append(Paragraph(
        "These are analytical, rule-based suggestions generated offline from historical traffic data. "
        "They are intended for a qualified traffic engineer to review — nothing is applied automatically "
        "to any live signal controller.", note_style
    ))
    story.append(Spacer(1, 10))

    for x_name in sorted(fdf["intersection"].unique()):
        idf = fdf[fdf["intersection"] == x_name]
        approaches = idf["direction"].unique()
        n_app = len(approaches)
        default_green = int(cycle_length * 0.9 / n_app)

        demand_rows = []
        for direction in approaches:
            sub = idf[idf["direction"] == direction]
            demand_rows.append({
                "direction": direction,
                "road": sub["road"].iloc[0],
                "avg_demand": sub["vehicles_per_min"].mean(),
                "lane_count": sub["lane_count"].iloc[0],
                "current_green": default_green,
            })
        rec_df = recommend_signal_split(pd.DataFrame(demand_rows), cycle_length)
        avg_score = idf["congestion_score"].mean()
        peak_row = idf.loc[idf["congestion_score"].idxmax()]

        story.append(Paragraph(x_name, h2_style))
        story.append(Paragraph(
            f"Average congestion score: <b>{avg_score:.1f}/100</b> &nbsp;|&nbsp; "
            f"Peak observed around <b>{peak_row['hour_label']}</b>", body_style
        ))
        story.append(Spacer(1, 4))

        table_data = [["Road (approach)", "Current green", "Recommended", "Change", "Reason"]]
        for _, row in rec_df.iterrows():
            change_txt = f"{row['change_sec']:+.0f} sec"
            reason_para = Paragraph(row["reason"], body_style)
            table_data.append([
                f"{row['road']} ({row['direction']})",
                f"{row['current_green']} sec",
                f"{row['recommended_green']} sec",
                change_txt,
                reason_para,
            ])

        tbl = Table(table_data, colWidths=[1.15 * inch, 0.75 * inch, 0.85 * inch, 0.6 * inch, 2.35 * inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2545")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7E0EC")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 6))

    story.append(PageBreak())
    story.append(Paragraph("Most Congested Approaches (All Intersections)", h2_style))
    bottleneck_tbl_data = [["Intersection", "Road", "Avg congestion", "% Severe", "Avg speed"]]
    for _, row in detect_bottlenecks(fdf, top_n=8).iterrows():
        bottleneck_tbl_data.append([
            row["intersection"], row["road"], f"{row['avg_congestion']:.1f}",
            f"{row['pct_severe']:.0f}%", f"{row['avg_speed']:.1f} km/h",
        ])
    bn_tbl = Table(bottleneck_tbl_data, colWidths=[1.3 * inch, 1.3 * inch, 1.1 * inch, 0.9 * inch, 1.1 * inch])
    bn_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2545")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7E0EC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(bn_tbl)
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Methodology: green-time recommendations shift each approach's share of the fixed cycle "
        "length toward its measured share of total demand, bounded by a minimum green time and a "
        "maximum share per approach. Congestion score blends volume/capacity ratio (70%) and "
        "speed reduction vs. the posted limit (30%). This is a simplified offline model, not a "
        "calibrated traffic-engineering study — field validation and engineer review are required "
        "before any real-world change.", note_style
    ))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ======================================================================
st.set_page_config(page_title="MetroFlow AI — Offline Signal Advisor", layout="wide", page_icon="🚦")

# ---- Custom styling: navy dashboard base + subtle red/amber/green signal accents ----
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
    --signal-red: #D64550;
    --signal-amber: #F4B740;
    --signal-green: #2BB673;
    --navy-deep: #0B2545;
    --navy-mid: #134074;
    --ink: #0B2545;
    --muted: #5A6B85;
}
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink); }

/* ---------- Hero banner with a small animated traffic signal ---------- */
.fs-hero {
    background: linear-gradient(120deg, #0B2545 0%, #134074 55%, #13315C 100%);
    padding: 22px 30px; border-radius: 16px; color: #F2F6FC; margin-bottom: 18px;
    box-shadow: 0 8px 22px rgba(11,37,69,0.28);
    display: flex; align-items: center; justify-content: space-between; gap: 20px;
}
.fs-hero-text h1 { margin: 0; font-size: 1.85rem; color: #F2F6FC; }
.fs-hero-text p { margin: 6px 0 0 0; color: #C9DAF0; font-size: 0.95rem; }
.fs-badge {
    display:inline-block; padding: 3px 10px; border-radius: 20px;
    background: rgba(43,182,115,0.18); color:#4FE3A0; font-size:0.72rem;
    font-weight:600; letter-spacing:0.04em; margin-top:10px;
}
.fs-signal {
    display:flex; flex-direction:column; gap:7px; background: rgba(255,255,255,0.06);
    padding: 10px 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
}
.fs-signal .dot { width:16px; height:16px; border-radius:50%; opacity:0.28; }
.fs-signal .dot.red { background: var(--signal-red); }
.fs-signal .dot.amber { background: var(--signal-amber); }
.fs-signal .dot.green { background: var(--signal-green); }
.fs-signal .dot.on { opacity:1; animation: fs-glow 2.4s ease-in-out infinite; }
.fs-signal .dot.red.on { animation-delay: 0s; }
.fs-signal .dot.amber.on { animation-delay: 0.8s; }
.fs-signal .dot.green.on { animation-delay: 1.6s; }
@keyframes fs-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,255,255,0); }
    50% { box-shadow: 0 0 14px 3px currentColor; }
}
.fs-signal .dot.red.on { color: var(--signal-red); }
.fs-signal .dot.amber.on { color: var(--signal-amber); }
.fs-signal .dot.green.on { color: var(--signal-green); }

/* ---------- KPI cards: hover lift + coloured left accent ---------- */
.fs-card {
    background:#FFFFFF; border:1px solid #E7ECF3; border-left: 4px solid var(--navy-mid);
    border-radius:12px; padding:14px 18px; box-shadow: 0 2px 8px rgba(19,64,116,0.06);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.fs-card:hover { transform: translateY(-3px); box-shadow: 0 8px 18px rgba(19,64,116,0.14); }
.fs-card.accent-green { border-left-color: var(--signal-green); }
.fs-card.accent-amber { border-left-color: var(--signal-amber); }
.fs-card.accent-red { border-left-color: var(--signal-red); }
.fs-card .label { font-size:0.75rem; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:0.03em;}
.fs-card .value { font-size:1.5rem; color:var(--ink); font-weight:700; font-family:'Space Grotesk',sans-serif; margin-top:2px;}

.fs-note {
    background:#EFF6FF; border-left:4px solid var(--navy-mid); padding:10px 14px;
    border-radius:6px; font-size:0.88rem; color:var(--ink);
}
.fs-sim-tag {
    display:inline-block; background:#FDECEA; color:var(--signal-red); font-weight:700;
    padding:2px 10px; border-radius:6px; font-size:0.72rem; letter-spacing:0.05em;
}

/* ---------- Tabs: clearer active state, signal-green underline ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #E7ECF3; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0; padding: 8px 14px; font-weight: 600; color: var(--muted);
}
.stTabs [aria-selected="true"] {
    color: var(--navy-deep) !important; background: #EFF6FF;
    border-bottom: 3px solid var(--signal-green) !important;
}

/* ---------- Buttons: signal-green primary action ---------- */
button[kind="primary"] {
    background: linear-gradient(120deg, var(--signal-green), #23A065) !important;
    border: none !important; transition: transform 0.12s ease, box-shadow 0.12s ease;
}
button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(43,182,115,0.35); }

/* ---------- Sidebar: subtle navy tint ---------- */
[data-testid="stSidebar"] { background: #F6F9FD; border-right: 1px solid #E7ECF3; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="fs-hero">
    <div class="fs-hero-text">
        <h1>🚦 MetroFlow AI</h1>
        <p>Offline congestion analytics &amp; signal-timing advisor.</p>
        <span class="fs-badge">ADVISORY ONLY · NO REAL-TIME CONTROL</span>
    </div>
    <div class="fs-signal" title="Analysing traffic flow">
        <div class="dot red on"></div>
        <div class="dot amber on"></div>
        <div class="dot green on"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar: data input (simple 3-way chooser) ----------------
st.sidebar.header("📥 Data")
data_mode = st.sidebar.radio(
    "Bring in data",
    ["🎲 Demo Data", "📄 Upload CSV", "✍️ Manual Entry"],
    label_visibility="collapsed",
)

# How much history: a single detailed day, or a full month of trend data.
# This choice feeds the demo generator and simplifies the manual-entry form.
period_choice = st.sidebar.radio(
    "History length",
    ["🕐 One Day", "📅 One Month"],
    horizontal=True,
    help="One Day = detailed 5-minute data for one day. One Month = daily trend "
         "data across ~30 days — good for spotting recurring patterns.",
)
is_monthly_pref = period_choice == "📅 One Month"

if "df_raw" not in st.session_state:
    st.session_state.df_raw = generate_synthetic_data(period="day")
    st.session_state.data_source = "Demo Data (1 Day)"

if data_mode == "🎲 Demo Data":
    if st.sidebar.button("Load demo data", use_container_width=True):
        period = "month" if is_monthly_pref else "day"
        st.session_state.df_raw = generate_synthetic_data(period=period)
        st.session_state.data_source = f"Demo Data ({'1 Month' if is_monthly_pref else '1 Day'})"

elif data_mode == "📄 Upload CSV":
    st.sidebar.caption("Any date range works — a single day or a full month of history.")
    uploaded = st.sidebar.file_uploader("Upload traffic CSV", type=["csv"])
    if uploaded is not None:
        df_loaded, warns = load_data(uploaded)
        if not df_loaded.empty:
            st.session_state.df_raw = df_loaded
            n_days = df_loaded["timestamp"].dt.date.nunique()
            st.session_state.data_source = f"Uploaded CSV ({n_days} day{'s' if n_days != 1 else ''})"
            for w in warns:
                st.sidebar.warning(w)

elif data_mode == "✍️ Manual Entry":
    st.sidebar.info("Scroll down to the **Manual Data Entry** panel below.")

st.sidebar.caption(f"Active dataset: **{st.session_state.data_source}** · "
                    f"{len(st.session_state.df_raw):,} rows")

# ---------------- Manual data entry panel (shown only when selected) ----------------
if data_mode == "✍️ Manual Entry":
    st.subheader("✍️ Manual Data Entry")

    if is_monthly_pref:
        st.caption("**Monthly summary mode** — add one row per day (per road). "
                   "Simpler than logging every 5 minutes: just the day's typical numbers.")
        if "manual_df_month" not in st.session_state:
            st.session_state.manual_df_month = pd.DataFrame([{
                "date": "2026-08-01", "intersection": "My Intersection",
                "road": "Main St", "direction": "North", "vehicles_per_min": 40,
                "average_speed_kmh": 35, "lane_count": 2, "speed_limit_kmh": 50,
                "turning_ratio": 0.15, "incident": "",
            }])
        edited_df = st.data_editor(
            st.session_state.manual_df_month,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "date": st.column_config.TextColumn("Date (YYYY-MM-DD)", required=True),
                "vehicles_per_min": st.column_config.NumberColumn("Avg vehicles/min", min_value=0),
                "average_speed_kmh": st.column_config.NumberColumn("Avg speed (km/h)", min_value=0),
                "lane_count": st.column_config.NumberColumn("Lanes", min_value=1, step=1),
                "speed_limit_kmh": st.column_config.NumberColumn("Speed limit (km/h)", min_value=0),
                "turning_ratio": st.column_config.NumberColumn("Turning ratio", min_value=0.0, max_value=1.0, step=0.01),
                "incident": st.column_config.SelectboxColumn("Incident", options=["", "accident", "roadwork", "other"]),
            },
            key="manual_editor_month",
        )
        st.session_state.manual_df_month = edited_df

        if st.button("✅ Load this data into the dashboard", type="primary"):
            prepped = edited_df.copy()
            prepped["timestamp"] = prepped["date"].astype(str) + " 08:00:00"
            prepped = prepped.drop(columns=["date"])
            buffer_csv = prepped.to_csv(index=False)
            df_loaded, warns = load_data(io.StringIO(buffer_csv))
            if not df_loaded.empty:
                st.session_state.df_raw = df_loaded
                n_days = df_loaded["timestamp"].dt.date.nunique()
                st.session_state.data_source = f"Manual Entry ({n_days} day{'s' if n_days != 1 else ''})"
                st.success(f"Loaded {len(df_loaded)} day(s) of data — see tabs below.")
                for w in warns:
                    st.warning(w)
    else:
        st.caption("Add or edit rows below — click the **+** at the bottom of the table to add a row, "
                   "or the trash icon to delete one. Leave a cell blank to use a sensible default.")

        if "manual_df" not in st.session_state:
            st.session_state.manual_df = pd.DataFrame([{
                "timestamp": "2026-08-12 08:00:00", "intersection": "My Intersection",
                "road": "Main St", "direction": "North", "vehicles_per_min": 50,
                "average_speed_kmh": 35, "lane_count": 2, "speed_limit_kmh": 50,
                "turning_ratio": 0.15, "incident": "",
            }])

        edited_df = st.data_editor(
            st.session_state.manual_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Timestamp (YYYY-MM-DD HH:MM:SS)", required=True),
                "vehicles_per_min": st.column_config.NumberColumn("Vehicles/min", min_value=0),
                "average_speed_kmh": st.column_config.NumberColumn("Avg speed (km/h)", min_value=0),
                "lane_count": st.column_config.NumberColumn("Lanes", min_value=1, step=1),
                "speed_limit_kmh": st.column_config.NumberColumn("Speed limit (km/h)", min_value=0),
                "turning_ratio": st.column_config.NumberColumn("Turning ratio", min_value=0.0, max_value=1.0, step=0.01),
                "incident": st.column_config.SelectboxColumn("Incident", options=["", "accident", "roadwork", "other"]),
            },
            key="manual_editor",
        )
        st.session_state.manual_df = edited_df

        if st.button("✅ Load this data into the dashboard", type="primary"):
            buffer_csv = edited_df.to_csv(index=False)
            df_loaded, warns = load_data(io.StringIO(buffer_csv))
            if not df_loaded.empty:
                st.session_state.df_raw = df_loaded
                st.session_state.data_source = "Manual Entry (Detailed)"
                st.success(f"Loaded {len(df_loaded)} row(s) into the dashboard — see tabs below.")
                for w in warns:
                    st.warning(w)
    st.divider()

df = preprocess(st.session_state.df_raw)
df = score_congestion(df)

# Auto-detect whether the *active* dataset spans multiple days — this drives
# simpler, daily-rolled-up charts instead of dense minute-by-minute lines.
is_monthly = df["date"].nunique() > 2

# ---------------- Sidebar: filters (collapsed by default to keep things simple) ----------------
with st.sidebar.expander("🔎 Filters (optional)", expanded=False):
    intersections = sorted(df["intersection"].unique())
    sel_intersections = st.multiselect("Intersection", intersections, default=intersections)

    dates = sorted(df["date"].unique())
    sel_dates = st.multiselect("Date", dates, default=dates)

    if is_monthly:
        sel_hr_range = (int(df["hour"].min()), int(df["hour"].max()))
    else:
        min_hr, max_hr = int(df["hour"].min()), int(df["hour"].max())
        sel_hr_range = st.slider("Time range (hour)", min_hr, max_hr, (min_hr, max_hr))

    incident_filter = st.selectbox("Incident status", ["All", "Only incidents", "Exclude incidents"])

fdf = df[
    df["intersection"].isin(sel_intersections)
    & df["date"].isin(sel_dates)
    & df["hour"].between(sel_hr_range[0], sel_hr_range[1])
]
if incident_filter == "Only incidents":
    fdf = fdf[fdf["incident"] != ""]
elif incident_filter == "Exclude incidents":
    fdf = fdf[fdf["incident"] == ""]

if fdf.empty:
    st.warning("No data matches the current filters — widen your selection in the sidebar.")
    st.stop()

# ---------------- KPI cards (kept to the 4 that matter most) ----------------
worst_row = fdf.groupby("intersection")["congestion_score"].mean().idxmax()
avg_congestion = fdf["congestion_score"].mean()

def _congestion_accent(score):
    if score < 55:
        return "accent-green"
    elif score < 80:
        return "accent-amber"
    return "accent-red"

k1, k2, k3, k4 = st.columns(4)
kpis = [
    (k1, "Avg volume", f"{fdf['vehicles_per_min'].mean():.1f} v/min", ""),
    (k2, "Avg speed", f"{fdf['average_speed_kmh'].mean():.1f} km/h", "accent-green"),
    (k3, "Avg congestion", f"{avg_congestion:.0f}/100", _congestion_accent(avg_congestion)),
    (k4, "Worst congestion at", worst_row, "accent-red"),
]
for col, label, value, accent in kpis:
    col.markdown(f'<div class="fs-card {accent}"><div class="label">{label}</div><div class="value">{value}</div></div>',
                 unsafe_allow_html=True)

st.write("")

tabs = st.tabs(["📊 Overview", "🧭 Congestion Analysis", "🚥 Signal Recommendation",
                 "⚖️ Before vs After", "🗂️ Data", "📐 Assumptions"])

# A single time axis choice, reused across tabs: daily points for monthly
# datasets (keeps the lines clean), raw timestamps for single-day datasets.
time_col = "date" if is_monthly else "timestamp"
time_axis_label = "Day" if is_monthly else "Time"

# =====================================================================
# TAB 1 — OVERVIEW
# =====================================================================
with tabs[0]:
    st.subheader(f"Traffic Volume by {time_axis_label}")
    vol_fig = px.line(
        fdf.groupby([time_col, "intersection"], as_index=False)["vehicles_per_min"].mean(),
        x=time_col, y="vehicles_per_min", color="intersection",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    vol_fig.update_layout(height=360, legend_title="Intersection", xaxis_title=time_axis_label,
                           yaxis_title="Avg vehicles/min")
    st.plotly_chart(vol_fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Average Speed by {time_axis_label}")
        spd_fig = px.line(
            fdf.groupby([time_col, "intersection"], as_index=False)["average_speed_kmh"].mean(),
            x=time_col, y="average_speed_kmh", color="intersection",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        spd_fig.update_layout(height=320, xaxis_title=time_axis_label, yaxis_title="km/h", showlegend=False)
        st.plotly_chart(spd_fig, use_container_width=True)

    with c2:
        st.subheader("Volume by Road")
        by_road = fdf.groupby("road", as_index=False)["vehicles_per_min"].mean().sort_values(
            "vehicles_per_min", ascending=True)
        road_fig = px.bar(by_road, x="vehicles_per_min", y="road", orientation="h",
                           color_discrete_sequence=["#134074"])
        road_fig.update_layout(height=320, xaxis_title="Avg vehicles/min", yaxis_title="")
        st.plotly_chart(road_fig, use_container_width=True)

# =====================================================================
# TAB 2 — CONGESTION ANALYSIS
# =====================================================================
with tabs[1]:
    st.subheader(f"Congestion Score by {time_axis_label}")
    cs_fig = px.line(
        fdf.groupby([time_col, "intersection"], as_index=False)["congestion_score"].mean(),
        x=time_col, y="congestion_score", color="intersection",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    cs_fig.add_hrect(y0=80, y1=100, fillcolor=BAND_COLORS["Severe"], opacity=0.08, line_width=0)
    cs_fig.update_layout(height=340, xaxis_title=time_axis_label, yaxis_title="Score (0-100)")
    st.plotly_chart(cs_fig, use_container_width=True)

    st.subheader("Intersection Comparison")
    comp = fdf.groupby("intersection", as_index=False)["congestion_score"].mean()
    bar_fig = px.bar(comp, x="intersection", y="congestion_score",
                      color="congestion_score",
                      color_continuous_scale=[BAND_COLORS["Low"], BAND_COLORS["Moderate"],
                                               BAND_COLORS["High"], BAND_COLORS["Severe"]],
                      range_color=[0, 100])
    bar_fig.update_layout(height=320, yaxis_title="Avg congestion score")
    st.plotly_chart(bar_fig, use_container_width=True)

    st.subheader("Peak Congestion Hours")
    st.dataframe(detect_peak_hours(fdf), use_container_width=True, hide_index=True)

    st.subheader("Most Congested Approaches / Bottlenecks")
    bottlenecks = detect_bottlenecks(fdf)
    st.dataframe(bottlenecks.style.format({
        "avg_congestion": "{:.1f}", "pct_severe": "{:.0f}%", "avg_vc": "{:.2f}", "avg_speed": "{:.1f}"
    }), use_container_width=True, hide_index=True)

    st.markdown(
        f'<div class="fs-note"><b>Capacity assumption:</b> each lane is assumed to sustain '
        f'~{BASE_CAPACITY_PER_LANE_PER_MIN} vehicles/min comfortably. Congestion score = '
        f'70% weight on volume-to-capacity ratio + 30% weight on speed reduction vs. speed limit, '
        f'clipped to 0–100.</div>', unsafe_allow_html=True)

# =====================================================================
# TAB 3 — SIGNAL RECOMMENDATION
# =====================================================================
with tabs[2]:
    st.subheader("Recommended Green-Time Split")
    rec_intersection = st.selectbox("Choose intersection", intersections, key="rec_int")
    cycle_length = st.number_input("Cycle length (sec)", min_value=40, max_value=180,
                                    value=DEFAULT_CYCLE_LENGTH_SEC, step=5)

    idf = fdf[fdf["intersection"] == rec_intersection]
    if idf.empty:
        st.info("No data for this intersection under current filters.")
    else:
        approaches = idf["direction"].unique()
        n_app = len(approaches)
        default_green = int(cycle_length * 0.9 / n_app)  # leave ~10% for clearance/yellow

        demand_rows = []
        for i, direction in enumerate(approaches):
            sub = idf[idf["direction"] == direction]
            demand_rows.append({
                "direction": direction,
                "road": sub["road"].iloc[0],
                "avg_demand": sub["vehicles_per_min"].mean(),
                "lane_count": sub["lane_count"].iloc[0],
                "current_green": default_green,
            })
        demand_df = pd.DataFrame(demand_rows)

        with st.expander("Adjust current green times (optional)"):
            for i, row in demand_df.iterrows():
                demand_df.at[i, "current_green"] = st.slider(
                    f"Current green — {row['road']} ({row['direction']})",
                    min_value=MIN_GREEN_SEC, max_value=int(cycle_length * MAX_GREEN_FRACTION),
                    value=int(row["current_green"]), key=f"green_{row['direction']}"
                )

        rec_df = recommend_signal_split(demand_df, cycle_length)

        for _, row in rec_df.iterrows():
            c1, c2, c3 = st.columns([1, 1, 2])
            c1.markdown(f"**{row['road']}** ({row['direction']})\n\nCurrent: **{row['current_green']} sec**")
            arrow = "🔼" if row["change_sec"] > 0 else ("🔽" if row["change_sec"] < 0 else "➡️")
            c2.markdown(f"Recommended: **{row['recommended_green']} sec**\n\n{arrow} {row['change_sec']:+.0f} sec")
            c3.info(row["reason"])

        st.subheader("Why this recommendation?")
        top_row = rec_df.sort_values("change_sec", ascending=False).iloc[0]
        peak_hr_row = idf.loc[idf["congestion_score"].idxmax()]
        st.markdown(
            f"**{rec_intersection}** experiences its highest observed congestion around "
            f"**{peak_hr_row['hour_label']}**. **{top_row['road']}** carries approximately "
            f"**{top_row['demand_share']*100:.0f}%** of total approach demand while currently "
            f"receiving **{top_row['current_green_share']*100:.0f}%** of green time — the model "
            f"proposes shifting the split toward measured demand."
        )
        st.caption("Recommendations are analytical suggestions for engineer review — not automatically applied.")

    st.divider()
    st.subheader("📄 PDF Report")
    st.caption("Generates a report covering every intersection currently in view — recommended "
               "green-time changes, reasons, and the top bottlenecks — ready to share with an engineer.")
    if st.button("Generate PDF report", type="primary"):
        pdf_bytes = build_recommendation_report(fdf, cycle_length, st.session_state.data_source)
        st.session_state.pdf_report_bytes = pdf_bytes
        st.success("Report generated — download below.")
    if "pdf_report_bytes" in st.session_state:
        st.download_button(
            "⬇️ Download PDF report", data=st.session_state.pdf_report_bytes,
            file_name="metroflow_ai_recommendations.pdf", mime="application/pdf",
        )

# =====================================================================
# TAB 4 — BEFORE VS AFTER SIMULATION
# =====================================================================
with tabs[3]:
    st.markdown('<span class="fs-sim-tag">SIMULATED ESTIMATE</span>', unsafe_allow_html=True)
    st.subheader("Queueing Simulation — Current vs Recommended Timing")

    sim_intersection = st.selectbox("Intersection", intersections, key="sim_int")
    idf2 = fdf[fdf["intersection"] == sim_intersection]

    if idf2.empty:
        st.info("No data for this intersection under current filters.")
    else:
        approaches2 = idf2["direction"].unique()
        default_green2 = int(cycle_length * 0.9 / len(approaches2))
        sim_choice = st.selectbox("Approach to simulate", approaches2, key="sim_dir")

        sub2 = idf2[idf2["direction"] == sim_choice]
        arrival_rate = sub2["vehicles_per_min"].mean()
        lanes2 = int(sub2["lane_count"].iloc[0])
        current_green = st.session_state.get(f"green_{sim_choice}", default_green2)

        demand_rows2 = []
        for direction in approaches2:
            s = idf2[idf2["direction"] == direction]
            demand_rows2.append({
                "direction": direction, "road": s["road"].iloc[0],
                "avg_demand": s["vehicles_per_min"].mean(), "lane_count": s["lane_count"].iloc[0],
                "current_green": st.session_state.get(f"green_{direction}", default_green2),
            })
        rec_df2 = recommend_signal_split(pd.DataFrame(demand_rows2), cycle_length)
        recommended_green = int(rec_df2.loc[rec_df2["direction"] == sim_choice, "recommended_green"].iloc[0])

        before = simulate_approach(arrival_rate, current_green, cycle_length, lanes2)
        after = simulate_approach(arrival_rate, recommended_green, cycle_length, lanes2)

        baseline_speed_reduction = sub2["speed_reduction"].mean()
        score_before = congestion_score_from_vc(before["vc_effective"], baseline_speed_reduction, before["vc_effective"])
        score_after = congestion_score_from_vc(after["vc_effective"], baseline_speed_reduction, before["vc_effective"])

        def pct_change(b, a):
            if b == 0:
                return "—"
            return f"{(a - b) / b * 100:+.1f}%"

        comp_table = pd.DataFrame({
            "Metric": ["Green time (sec)", "Avg queue (veh)", "Avg waiting time (sec)",
                       "Throughput (veh/hr)", "Congestion score"],
            "Before": [current_green, before["avg_queue"], before["avg_wait_sec"],
                       before["throughput_per_hr"], score_before],
            "Simulated After": [recommended_green, after["avg_queue"], after["avg_wait_sec"],
                                 after["throughput_per_hr"], score_after],
        })
        comp_table["Change"] = [
            f"{recommended_green - current_green:+d} sec",
            pct_change(before["avg_queue"], after["avg_queue"]),
            pct_change(before["avg_wait_sec"], after["avg_wait_sec"]),
            pct_change(before["throughput_per_hr"], after["throughput_per_hr"]),
            pct_change(score_before, score_after),
        ]
        st.dataframe(comp_table, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_q = go.Figure()
            fig_q.add_trace(go.Bar(x=["Before", "After"], y=[before["avg_queue"], after["avg_queue"]],
                                    marker_color=["#D64550", "#2BB673"]))
            fig_q.update_layout(title="Average Queue (veh)", height=300)
            st.plotly_chart(fig_q, use_container_width=True)
        with c2:
            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(x=["Before", "After"], y=[before["avg_wait_sec"], after["avg_wait_sec"]],
                                    marker_color=["#D64550", "#2BB673"]))
            fig_w.update_layout(title="Average Waiting Time (sec)", height=300)
            st.plotly_chart(fig_w, use_container_width=True)

        st.caption(
            "Model: minute-by-minute deterministic queue build/drain using an assumed saturation "
            f"flow of {SATURATION_FLOW_PER_LANE_PER_MIN} veh/min/lane during green. This is a simplified "
            "estimate, not a calibrated microsimulation, and results are not a guarantee of real-world outcomes."
        )

    st.divider()
    st.subheader("Incident Impact")
    if (fdf["incident"] != "").any():
        inc = fdf[fdf["incident"] != ""]
        norm = fdf[fdf["incident"] == ""]
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg speed — incident", f"{inc['average_speed_kmh'].mean():.1f} km/h",
                   f"{inc['average_speed_kmh'].mean() - norm['average_speed_kmh'].mean():.1f} vs normal")
        c2.metric("Avg congestion — incident", f"{inc['congestion_score'].mean():.1f}",
                   f"{inc['congestion_score'].mean() - norm['congestion_score'].mean():+.1f} vs normal")
        c3.metric("Incident periods", f"{inc[['intersection','road','incident']].drop_duplicates().shape[0]}")
        st.dataframe(inc.groupby(["intersection", "road", "incident"], as_index=False).agg(
            avg_speed=("average_speed_kmh", "mean"), avg_congestion=("congestion_score", "mean"),
            records=("incident", "count")
        ), use_container_width=True, hide_index=True)
    else:
        st.info("No incident data available.")

# =====================================================================
# TAB 5 — DATA
# =====================================================================
with tabs[4]:
    st.subheader("Filtered Dataset")
    st.dataframe(fdf.sort_values("timestamp"), use_container_width=True, height=420)
    st.download_button("⬇️ Download filtered data (CSV)", fdf.to_csv(index=False).encode(),
                        "metroflow_ai_filtered_data.csv", "text/csv")

    st.subheader("Expected CSV Schema")
    st.code(", ".join(REQUIRED_COLUMNS), language="text")
    st.caption("Only 'timestamp', 'intersection' and 'vehicles_per_min' are strictly required — "
               "everything else is filled in with a documented default if missing.")

# =====================================================================
# TAB 6 — ASSUMPTIONS & LIMITATIONS
# =====================================================================
with tabs[5]:
    st.subheader("Model Assumptions & Limitations")
    st.markdown("""
- This tool uses **historical / synthetic data only** — it is **not** a real-time system and does **not** connect to or control any live traffic signal.
- Road capacity is **estimated** from lane count using an assumed sustained throughput of
  **{base} veh/min/lane**, and simulated saturation flow of **{sat} veh/min/lane** during green.
- The congestion score is a **transparent weighted blend** (70% volume/capacity, 30% speed reduction), not a calibrated traffic-engineering index.
- Queue and waiting-time calculations use a **simplified deterministic model** (not a stochastic microsimulation like SUMO or VISSIM).
- Signal-timing recommendations are **rule-based analytical suggestions** for a qualified traffic engineer to review — they are not automatically deployed.
- "Before vs After" results are **simulated estimates only** and are not a guarantee of real-world improvement.
- Real-world deployment requires **field validation, traffic-engineering review, and compliance with local signal-timing standards**.
- The system does **not** directly or indirectly control traffic signals in any way.
    """.format(base=BASE_CAPACITY_PER_LANE_PER_MIN, sat=SATURATION_FLOW_PER_LANE_PER_MIN))

    st.markdown(
        '<div class="fs-note">MetroFlow AI is an academic MVP intended to demonstrate an offline '
        'analysis-and-recommendation workflow, not a production traffic-management product.</div>',
        unsafe_allow_html=True
    )
