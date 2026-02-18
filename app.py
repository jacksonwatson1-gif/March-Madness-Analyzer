"""
March Madness Analyzer — app.py  (v2)
======================================
Streamlit application for bracket analysis, team statistics,
historical seed performance, and upset probability modeling.

Dependencies: streamlit, pandas, requests, rapidfuzz

Data sources:
  - Barttorvik T-Rank  → AdjOE, AdjDE, Tempo, Continuity (live)
  - ESPN unofficial API → real bracket seeds + regions (live after Selection Sunday)
  - Synthetic fallback  → used automatically when APIs are unavailable
"""

import streamlit as st
import pandas as pd
import requests
import math
import random
from datetime import datetime

# rapidfuzz is optional — gracefully degrade if not installed
try:
    from rapidfuzz import process as fuzz_process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="March Madness Analyzer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

:root {
    --orange: #FF6B35;
    --navy:   #0B1F3A;
    --cream:  #F5F0E8;
    --gold:   #FFD166;
    --green:  #06D6A0;
    --red:    #EF476F;
    --card:   #0F2847;
}

html, body, [class*="css"] {
    background-color: var(--navy);
    color: var(--cream);
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; }

.main-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    color: var(--orange);
    line-height: 1;
    letter-spacing: 4px;
}

.subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--gold);
    letter-spacing: 3px;
    text-transform: uppercase;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--orange);
    border-radius: 4px;
    padding: 1.2rem;
    text-align: center;
    margin: 0.3rem 0;
}

.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: var(--gold);
}

.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--orange);
    letter-spacing: 2px;
    text-transform: uppercase;
}

.upset-high   { color: #EF476F; font-weight: 600; }
.upset-medium { color: #FFD166; font-weight: 600; }
.upset-low    { color: #06D6A0; font-weight: 600; }

.badge-live { background:#06D6A0; color:#000; padding:2px 8px;
              border-radius:3px; font-size:0.65rem;
              font-family:'IBM Plex Mono',monospace; letter-spacing:2px; }
.badge-demo { background:#FF6B35; color:#000; padding:2px 8px;
              border-radius:3px; font-size:0.65rem;
              font-family:'IBM Plex Mono',monospace; letter-spacing:2px; }

.stSelectbox label, .stSlider label, .stMultiSelect label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--orange) !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

div[data-testid="stSidebar"] {
    background-color: #071429;
    border-right: 2px solid var(--orange);
}

.stDataFrame { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }

section[data-testid="stSidebar"] h2 {
    font-family: 'Bebas Neue', sans-serif;
    color: var(--orange);
    letter-spacing: 3px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
REGIONS = ["East", "West", "South", "Midwest"]
ROUNDS  = ["Round of 64", "Round of 32", "Sweet 16",
           "Elite Eight", "Final Four", "Championship"]

CONFERENCES = [
    "ACC", "Big Ten", "Big 12", "SEC", "Big East",
    "Pac-12", "American", "Mountain West", "WCC", "A-10"
]

HISTORICAL_UPSET_RATES = {
    (1,16): 0.013, (2,15): 0.063, (3,14): 0.152, (4,13): 0.202,
    (5,12): 0.359, (6,11): 0.370, (7,10): 0.392, (8,9):  0.489,
}

# ── Live Data: Barttorvik T-Rank ───────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_barttorvik(year: int = 2025) -> pd.DataFrame:
    """
    Fetch T-Rank efficiency data from barttorvik.com.
    Returns: AdjOE, AdjDE, Tempo, Continuity, AdjEM per team.
    Falls back to empty DataFrame on failure.
    """
    url = f"https://barttorvik.com/{year}_team_results.json"
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        raw = r.json()
        records = []
        for row in raw:
            try:
                records.append({
                    "Team":        str(row[0]).strip(),
                    "Conference":  str(row[1]).strip(),
                    "AdjOE":       float(row[2]),
                    "AdjDE":       float(row[3]),
                    "Tempo":       float(row[4]),
                    "Luck":        float(row[5]),
                    "AdjEM":       round(float(row[2]) - float(row[3]), 2),
                    "OppAdjOE":    float(row[6]),
                    "OppAdjDE":    float(row[7]),
                    "Continuity":  float(row[14]) if len(row) > 14 else None,
                })
            except (IndexError, ValueError, TypeError):
                continue
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


# ── Live Data: ESPN Bracket ────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_espn_bracket() -> pd.DataFrame:
    """
    Pull real bracket seeds + regions from ESPN's unofficial API.
    Only populated after Selection Sunday (~March 16).
    Returns empty DataFrame if unavailable.
    """
    url = (
        "https://site.api.espn.com/apis/v2/sports/basketball/"
        "mens-college-basketball/tournaments/22?groups=50"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        teams = []
        rounds = data.get("bracket", {}).get("rounds", [])
        if not rounds:
            return pd.DataFrame()
        for matchup in rounds[0].get("matchups", []):
            region = matchup.get("region", {}).get("name", "Unknown")
            for competitor in matchup.get("competitors", []):
                teams.append({
                    "Team":   competitor["team"]["displayName"],
                    "Seed":   int(competitor.get("seed", 0)),
                    "Region": region,
                })
        return pd.DataFrame(teams) if teams else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── Fuzzy Team Name Matching ───────────────────────────────────────────────────
def fuzzy_match_team(name: str, choices: list, threshold: int = 82):
    """Match a team name against a list using fuzzy string similarity."""
    if not FUZZY_AVAILABLE or not choices:
        return None
    result = fuzz_process.extractOne(name, choices)
    if result and result[1] >= threshold:
        return result[0]
    return None


# ── Synthetic Demo Data (Fallback) ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def generate_demo_bracket() -> pd.DataFrame:
    """Generate a realistic 68-team bracket with synthetic stats."""
    random.seed(42)
    teams = []
    for region in REGIONS:
        for seed in range(1, 17):
            conference = random.choice(CONFERENCES)
            base_off   = 115 - (seed * 2.5) + random.uniform(-3, 3)
            base_def   = 90  + (seed * 1.8) + random.uniform(-3, 3)
            win_pct    = max(0.40, min(0.97, 1.0 - (seed * 0.035) + random.uniform(-0.05, 0.05)))
            sos        = round(random.uniform(0.45, 0.72), 3)
            kenpom     = round(30 - (seed * 1.9) + random.uniform(-4, 4), 1)
            continuity = round(random.uniform(45, 90), 1)
            tempo      = round(random.uniform(64, 74), 1)

            teams.append({
                "Seed":        seed,
                "Region":      region,
                "Team":        f"Team {chr(64+seed)}-{region[:2]}",
                "Conference":  conference,
                "Record":      f"{int(win_pct*30)}-{30-int(win_pct*30)}",
                "Win%":        round(win_pct, 3),
                "AdjOE":       round(base_off, 1),
                "AdjDE":       round(base_def, 1),
                "AdjEM":       round(base_off - base_def, 1),
                "OffRtg":      round(base_off, 1),
                "DefRtg":      round(base_def, 1),
                "NetRtg":      round(base_off - base_def, 1),
                "Tempo":       tempo,
                "Continuity":  continuity,
                "SOS":         sos,
                "KenPom":      kenpom,
                "3P%":         round(random.uniform(0.31, 0.40), 3),
                "TOV%":        round(random.uniform(14, 22), 1),
                "RebMgn":      round(random.uniform(-3, 8), 1),
            })
    return pd.DataFrame(teams)


# ── Build Full Dataset (Live + Fallback) ───────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def build_full_dataset(year: int = 2025):
    """
    Attempt to build a real bracket by merging ESPN seeds with Barttorvik stats.
    Returns (DataFrame, data_mode) where data_mode is 'LIVE' or 'DEMO'.
    """
    df_espn  = fetch_espn_bracket()
    df_trank = fetch_barttorvik(year)

    if df_espn.empty or df_trank.empty:
        return generate_demo_bracket(), "DEMO"

    trank_names = df_trank["Team"].tolist()

    def match_name(name):
        if name in trank_names:
            return name
        return fuzzy_match_team(name, trank_names)

    df_espn["TrankName"] = df_espn["Team"].apply(match_name)
    df_merged = df_espn.merge(
        df_trank.rename(columns={"Team": "TrankName"}),
        on="TrankName", how="left"
    ).drop(columns=["TrankName"], errors="ignore")

    # Fill any missing stats with seed-correlated estimates
    for idx, row in df_merged.iterrows():
        seed = row.get("Seed", 8)
        if pd.isna(row.get("AdjOE")):
            df_merged.at[idx, "AdjOE"] = round(115 - (seed * 2.5) + random.uniform(-2, 2), 1)
        if pd.isna(row.get("AdjDE")):
            df_merged.at[idx, "AdjDE"] = round(90 + (seed * 1.8) + random.uniform(-2, 2), 1)
        if pd.isna(row.get("AdjEM")):
            df_merged.at[idx, "AdjEM"] = round(
                df_merged.at[idx, "AdjOE"] - df_merged.at[idx, "AdjDE"], 2)
        if pd.isna(row.get("Continuity")):
            df_merged.at[idx, "Continuity"] = round(random.uniform(50, 85), 1)
        if pd.isna(row.get("Tempo")):
            df_merged.at[idx, "Tempo"] = round(random.uniform(64, 74), 1)
        if pd.isna(row.get("Conference")):
            df_merged.at[idx, "Conference"] = "Unknown"

    df_merged["KenPom"] = df_merged["AdjEM"].rank(ascending=False).astype(int)
    df_merged["SOS"]    = df_merged.get("OppAdjOE", pd.Series(dtype=float)).fillna(
        df_merged["Seed"].apply(lambda s: round(random.uniform(0.45, 0.72), 3))
    )
    df_merged["NetRtg"] = df_merged["AdjEM"]
    df_merged["OffRtg"] = df_merged["AdjOE"]
    df_merged["DefRtg"] = df_merged["AdjDE"]

    return df_merged, "LIVE"


# ── Historical Seed Data ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def historical_seed_performance() -> pd.DataFrame:
    data = {
        "Seed": list(range(1, 17)),
        "FF_Appearances": [140,48,32,22,18,14,12,12,3,5,8,9,1,1,0,1],
        "Championships":  [37,18,7,5,3,2,1,1,0,0,1,0,0,0,0,0],
        "R32_Rate":       [0.993,0.937,0.848,0.798,0.641,0.630,0.608,0.511,
                           0.489,0.392,0.370,0.359,0.202,0.152,0.063,0.013],
        "Sweet16_Rate":   [0.874,0.680,0.540,0.436,0.318,0.296,0.280,0.240,
                           0.165,0.155,0.163,0.152,0.074,0.063,0.027,0.006],
    }
    return pd.DataFrame(data)


# ── Upset Probability Engine ───────────────────────────────────────────────────
def upset_probability(
    fav_seed: int,
    dog_seed: int,
    fav_kenpom: float,
    dog_kenpom: float,
    fav_sos: float,
    dog_sos: float,
    fav_continuity: float = 70.0,
    dog_continuity: float = 70.0,
    fav_adjoe: float = 0.0,
    dog_adjoe: float = 0.0,
    fav_adjde: float = 0.0,
    dog_adjde: float = 0.0,
) -> dict:
    """
    Logistic upset probability model incorporating:
      - Historical base rate (seed matchup)
      - KenPom / AdjEM differential
      - Strength-of-schedule differential
      - Roster continuity differential
      - Offensive efficiency differential (AdjOE)
      - Defensive efficiency differential (AdjDE)
    """
    matchup   = (min(fav_seed, dog_seed), max(fav_seed, dog_seed))
    base_rate = HISTORICAL_UPSET_RATES.get(matchup, 0.35)

    kenpom_diff     = fav_kenpom - dog_kenpom
    sos_diff        = fav_sos - dog_sos
    continuity_diff = dog_continuity - fav_continuity  # positive = underdog more experienced
    adjoe_diff      = fav_adjoe - dog_adjoe
    adjde_diff      = dog_adjde - fav_adjde

    logit_base = math.log(max(base_rate, 1e-6) / max(1 - base_rate, 1e-6))
    logit_adj  = (
        logit_base
        - (kenpom_diff      * 0.07)
        - (sos_diff         * 2.2)
        - (continuity_diff  * 0.012)
        - (adjoe_diff       * 0.04)
        - (adjde_diff       * 0.03)
    )

    prob_upset = 1 / (1 + math.exp(-logit_adj))
    prob_upset = max(0.01, min(0.75, prob_upset))
    severity   = "HIGH" if prob_upset > 0.40 else "MEDIUM" if prob_upset > 0.22 else "LOW"

    return {
        "upset_prob":       round(prob_upset, 4),
        "fav_prob":         round(1 - prob_upset, 4),
        "severity":         severity,
        "seed_diff":        dog_seed - fav_seed,
        "base_rate":        base_rate,
        "kenpom_edge":      round(kenpom_diff, 1),
        "continuity_edge":  round(continuity_diff, 1),
        "adjoe_edge":       round(adjoe_diff, 1),
        "adjde_edge":       round(adjde_diff, 1),
    }


# ── Monte Carlo Simulator ──────────────────────────────────────────────────────
def simulate_tournament(df: pd.DataFrame, n_sims: int = 1000) -> pd.DataFrame:
    """Monte Carlo tournament simulation. Returns championship probability per team."""
    results = {row["Team"]: 0 for _, row in df.iterrows()}

    def play_game(t1, t2):
        fav = t1 if t1["KenPom"] >= t2["KenPom"] else t2
        dog = t2 if fav["Team"] == t1["Team"] else t1
        up  = upset_probability(
            fav["Seed"], dog["Seed"],
            fav["KenPom"], dog["KenPom"],
            fav.get("SOS", 0.55),        dog.get("SOS", 0.55),
            fav.get("Continuity", 70),   dog.get("Continuity", 70),
            fav.get("AdjOE", 105),       dog.get("AdjOE", 105),
            fav.get("AdjDE", 100),       dog.get("AdjDE", 100),
        )
        return dog if random.random() < up["upset_prob"] else fav

    for _ in range(n_sims):
        region_winners = []
        for region in REGIONS:
            region_df = df[df["Region"] == region].sort_values("Seed")
            seeded    = {row["Seed"]: row for _, row in region_df.iterrows()}
            order     = [1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]
            pool      = [seeded[s] for s in order if s in seeded]

            while len(pool) > 1:
                next_round = []
                for i in range(0, len(pool), 2):
                    if i + 1 >= len(pool):
                        next_round.append(pool[i])
                    else:
                        next_round.append(play_game(pool[i], pool[i+1]))
                pool = next_round
            region_winners.append(pool[0])

        ff_pool = region_winners[:]
        while len(ff_pool) > 1:
            next_round = []
            for i in range(0, len(ff_pool), 2):
                if i + 1 >= len(ff_pool):
                    next_round.append(ff_pool[i])
                else:
                    next_round.append(play_game(ff_pool[i], ff_pool[i+1]))
            ff_pool = next_round

        results[ff_pool[0]["Team"]] += 1

    df_sim = df.copy()
    df_sim["ChampionshipProb"] = df_sim["Team"].map(
        lambda t: round(results.get(t, 0) / n_sims, 4)
    )
    return df_sim.sort_values("ChampionshipProb", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading bracket data..."):
    df_bracket, data_mode = build_full_dataset(year=2025)
    df_history = historical_seed_performance()

# Ensure all required columns exist regardless of data source
for col, default in [("Continuity", 70.0), ("Tempo", 68.0),
                     ("AdjOE", 105.0), ("AdjDE", 100.0), ("AdjEM", 5.0),
                     ("KenPom", 15.0), ("SOS", 0.55)]:
    if col not in df_bracket.columns:
        df_bracket[col] = default

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ CONTROLS")
    data_badge = "<span class='badge-live'>● LIVE</span>" if data_mode == "LIVE" \
                 else "<span class='badge-demo'>● DEMO</span>"
    st.markdown(f"**Data Mode:** {data_badge}", unsafe_allow_html=True)
    st.markdown("---")

    selected_region = st.selectbox("Region Filter", ["All"] + REGIONS)
    selected_round  = st.selectbox("Round", ROUNDS)
    n_simulations   = st.slider("Monte Carlo Simulations", 100, 5000, 1000, 100)
    seed_range      = st.slider("Seed Range", 1, 16, (1, 16))
    conf_filter     = st.multiselect("Conference Filter", CONFERENCES, default=CONFERENCES)
    run_sim         = st.button("▶ RUN SIMULATION", use_container_width=True)

    if st.button("🔄 Refresh Live Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.68rem;color:#888;font-family:IBM Plex Mono,monospace;'>
    Live data: Barttorvik + ESPN<br>
    Fallback: synthetic (seed-correlated)<br><br>
    Upset model inputs:<br>
    · Historical base rate<br>
    · AdjEM differential<br>
    · Offensive efficiency (AdjOE)<br>
    · Defensive efficiency (AdjDE)<br>
    · Roster continuity<br>
    · Strength of schedule
    </div>""", unsafe_allow_html=True)


# ── Apply Filters ──────────────────────────────────────────────────────────────
df_filtered = df_bracket[
    (df_bracket["Seed"] >= seed_range[0]) &
    (df_bracket["Seed"] <= seed_range[1])
].copy()

if "Conference" in df_filtered.columns and conf_filter != CONFERENCES:
    df_filtered = df_filtered[df_filtered["Conference"].isin(conf_filter)]
if selected_region != "All":
    df_filtered = df_filtered[df_filtered["Region"] == selected_region]


# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_date = st.columns([5, 1])
with col_title:
    st.markdown("<div class='main-title'>MARCH MADNESS<br>ANALYZER</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>● NCAA Tournament Intelligence Platform ●</div>",
                unsafe_allow_html=True)
with col_date:
    mode_color = "#06D6A0" if data_mode == "LIVE" else "#FF6B35"
    st.markdown(f"""
    <div style='text-align:right;font-family:IBM Plex Mono,monospace;
                font-size:0.7rem;color:#FFD166;padding-top:1rem;'>
    {datetime.now().strftime('%Y · %b %d')}<br>
    <span style='color:{mode_color}'>{'● LIVE DATA' if data_mode == "LIVE" else '● DEMO MODE'}</span>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── KPI Row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
for col, (val, label) in zip([k1,k2,k3,k4,k5], [
    ("68", "Field Size"), ("4", "Regions"), ("63", "Games"),
    (str(len(df_filtered)), "Filtered Teams"), ("2025", "Season")
]):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 BRACKET FIELD",
    "⚡ EFFICIENCY",
    "🎯 UPSET ANALYZER",
    "📈 SEED HISTORY",
    "🎲 MONTE CARLO SIM",
])


# ─── Tab 1: Bracket Field ──────────────────────────────────────────────────────
with tab1:
    st.markdown("### FIELD AT A GLANCE")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        avail_cols = [c for c in
            ["Seed","Region","Team","Conference","Record","Win%",
             "AdjOE","AdjDE","AdjEM","Continuity","KenPom","SOS","Tempo"]
            if c in df_filtered.columns]
        fmt = {}
        if "Win%"      in avail_cols: fmt["Win%"]      = "{:.1%}"
        if "SOS"       in avail_cols: fmt["SOS"]       = "{:.3f}"
        if "AdjEM"     in avail_cols: fmt["AdjEM"]     = "{:+.1f}"
        if "Continuity" in avail_cols: fmt["Continuity"] = "{:.1f}%"

        styled = df_filtered[avail_cols].style
        if "KenPom" in avail_cols:
            styled = styled.background_gradient(subset=["KenPom"], cmap="YlOrRd")
        if "AdjEM" in avail_cols:
            styled = styled.background_gradient(subset=["AdjEM"], cmap="RdYlGn")
        if "AdjOE" in avail_cols:
            styled = styled.background_gradient(subset=["AdjOE"], cmap="Greens")
        if "AdjDE" in avail_cols:
            styled = styled.background_gradient(subset=["AdjDE"], cmap="RdYlGn_r")
        styled = styled.format(fmt)
        st.dataframe(styled, use_container_width=True, height=520)

    with col_b:
        st.markdown("#### Avg AdjEM by Seed")
        seed_avg = (df_filtered.groupby("Seed")["AdjEM"]
                    .mean().reset_index().rename(columns={"AdjEM": "Avg AdjEM"}))
        st.bar_chart(seed_avg.set_index("Seed"), use_container_width=True, height=220)

        st.markdown("#### Continuity % by Seed")
        if "Continuity" in df_filtered.columns:
            cont_avg = (df_filtered.groupby("Seed")["Continuity"]
                        .mean().reset_index().rename(columns={"Continuity": "Avg Continuity %"}))
            st.bar_chart(cont_avg.set_index("Seed"), use_container_width=True, height=220)


# ─── Tab 2: Efficiency Dashboard ──────────────────────────────────────────────
with tab2:
    st.markdown("### OFFENSIVE & DEFENSIVE EFFICIENCY")
    st.markdown("All ratings adjusted for opponent strength (per 100 possessions).")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("#### Top 20 — Offensive Efficiency (AdjOE)")
        top_off = (df_bracket.nlargest(20, "AdjOE")
                   [["Seed","Team","Region","AdjOE","Tempo"]]
                   .reset_index(drop=True))
        st.dataframe(
            top_off.style
                .background_gradient(subset=["AdjOE"], cmap="Greens")
                .format({"AdjOE": "{:.1f}", "Tempo": "{:.1f}"}),
            use_container_width=True, height=420
        )

    with col_e2:
        st.markdown("#### Top 20 — Defensive Efficiency (AdjDE — lower is better)")
        top_def = (df_bracket.nsmallest(20, "AdjDE")
                   [["Seed","Team","Region","AdjDE","Tempo"]]
                   .reset_index(drop=True))
        st.dataframe(
            top_def.style
                .background_gradient(subset=["AdjDE"], cmap="RdYlGn_r")
                .format({"AdjDE": "{:.1f}", "Tempo": "{:.1f}"}),
            use_container_width=True, height=420
        )

    st.markdown("---")

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        st.markdown("#### Net Efficiency (AdjEM) — Top 30 Teams")
        chart_data = (df_bracket.sort_values("AdjEM", ascending=False)
                      .head(30).set_index("Team")["AdjEM"])
        st.bar_chart(chart_data, use_container_width=True, height=300)

    with col_f2:
        st.markdown("#### Continuity Leaderboard")
        cont_leaders = (df_bracket.nlargest(15, "Continuity")
                        [["Seed","Team","Continuity","AdjEM"]]
                        .reset_index(drop=True))
        st.dataframe(
            cont_leaders.style
                .background_gradient(subset=["Continuity"], cmap="Blues")
                .format({"Continuity": "{:.1f}%", "AdjEM": "{:+.1f}"}),
            use_container_width=True, height=360
        )

    st.markdown("---")
    st.markdown("#### Efficiency Quadrant Analysis")
    st.caption("Teams in the **Elite** quadrant have above-median offense AND below-median defense.")

    med_off = df_bracket["AdjOE"].median()
    med_def = df_bracket["AdjDE"].median()

    def get_quadrant(row):
        if row["AdjOE"] >= med_off and row["AdjDE"] <= med_def:
            return "Elite (High O + Low D)"
        elif row["AdjOE"] >= med_off:
            return "Offense-Only"
        elif row["AdjDE"] <= med_def:
            return "Defense-Only"
        return "Below Average"

    df_quad = df_bracket.copy()
    df_quad["Quadrant"] = df_quad.apply(get_quadrant, axis=1)

    q1, q2 = st.columns([1, 2])
    with q1:
        quad_counts = df_quad["Quadrant"].value_counts().reset_index()
        quad_counts.columns = ["Quadrant", "Team Count"]
        st.dataframe(quad_counts, use_container_width=True, hide_index=True)
    with q2:
        elite = (df_quad[df_quad["Quadrant"] == "Elite (High O + Low D)"]
                 .sort_values("AdjEM", ascending=False)
                 [["Seed","Team","Region","AdjOE","AdjDE","AdjEM","Continuity"]]
                 .head(16).reset_index(drop=True))
        st.dataframe(
            elite.style
                .background_gradient(subset=["AdjEM"], cmap="YlOrRd")
                .format({"AdjOE": "{:.1f}", "AdjDE": "{:.1f}",
                         "AdjEM": "{:+.1f}", "Continuity": "{:.1f}%"}),
            use_container_width=True
        )


# ─── Tab 3: Upset Analyzer ────────────────────────────────────────────────────
with tab3:
    st.markdown("### UPSET PROBABILITY MATRIX")
    st.markdown("Head-to-head matchup calculator incorporating AdjOE, AdjDE, continuity, and SOS.")

    teams_list = df_bracket["Team"].tolist()

    def team_card(data, label):
        adjoe = data.get("AdjOE", 0.0)
        adjde = data.get("AdjDE", 0.0)
        adjem = data.get("AdjEM", 0.0)
        cont  = data.get("Continuity", 70.0)
        return f"""
        <div class='metric-card'>
        <div class='metric-label'>{label} | Seed · {data['Seed']} | KenPom · {data['KenPom']}</div>
        <div style='font-size:0.85rem;color:#aaa;margin-bottom:0.8rem;'>
            {data.get('Conference','?')} · {data.get('Region','?')}
        </div>
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:0.4rem;'>
            <div>
                <div class='metric-value' style='font-size:1.3rem;color:#06D6A0;'>{adjoe:.1f}</div>
                <div class='metric-label'>AdjOE</div>
            </div>
            <div>
                <div class='metric-value' style='font-size:1.3rem;color:#EF476F;'>{adjde:.1f}</div>
                <div class='metric-label'>AdjDE</div>
            </div>
            <div>
                <div class='metric-value' style='font-size:1.3rem;color:#FFD166;'>{adjem:+.1f}</div>
                <div class='metric-label'>AdjEM</div>
            </div>
            <div>
                <div class='metric-value' style='font-size:1.3rem;color:#aaa;'>{cont:.0f}%</div>
                <div class='metric-label'>Continuity</div>
            </div>
        </div>
        </div>"""

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🏆 FAVORITE")
        fav_team = st.selectbox("Favorite Team", teams_list, key="fav")
        fav_data = df_bracket[df_bracket["Team"] == fav_team].iloc[0]
        st.markdown(team_card(fav_data, "FAV"), unsafe_allow_html=True)

    with c2:
        st.markdown("##### 🐶 UNDERDOG")
        dog_team = st.selectbox("Underdog Team",
                                [t for t in teams_list if t != fav_team], key="dog")
        dog_data = df_bracket[df_bracket["Team"] == dog_team].iloc[0]
        st.markdown(team_card(dog_data, "DOG"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("COMPUTE UPSET PROBABILITY"):
        up = upset_probability(
            fav_data["Seed"],            dog_data["Seed"],
            fav_data["KenPom"],          dog_data["KenPom"],
            fav_data.get("SOS", 0.55),   dog_data.get("SOS", 0.55),
            fav_data.get("Continuity", 70), dog_data.get("Continuity", 70),
            fav_data.get("AdjOE", 105),  dog_data.get("AdjOE", 105),
            fav_data.get("AdjDE", 100),  dog_data.get("AdjDE", 100),
        )
        sev_class = f"upset-{up['severity'].lower()}"
        r1, r2, r3, r4, r5 = st.columns(5)
        for col, (val, lbl) in zip([r1,r2,r3,r4,r5], [
            (f"{up['upset_prob']*100:.1f}%",  "Upset Probability"),
            (f"{up['fav_prob']*100:.1f}%",    "Favorite Win Prob"),
            (f"{up['base_rate']*100:.1f}%",   "Historical Base Rate"),
            (f"{up['continuity_edge']:+.1f}%","Continuity Edge (Dog)"),
            (f"<span class='{sev_class}'>{up['severity']}</span>", "Upset Severity"),
        ]):
            with col:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='font-size:1.6rem;'>{val}</div>
                    <div class='metric-label'>{lbl}</div></div>""",
                    unsafe_allow_html=True)

        st.markdown("##### Model Factor Breakdown")
        factors = pd.DataFrame([
            {"Factor": "Historical Base Rate (seed matchup)", "Value": f"{up['base_rate']*100:.1f}%",  "Favors": "—"},
            {"Factor": "KenPom / AdjEM Differential",         "Value": f"{up['kenpom_edge']:+.1f}",    "Favors": "Favorite" if up['kenpom_edge'] > 0 else "Underdog"},
            {"Factor": "Offensive Efficiency (AdjOE) Δ",      "Value": f"{up['adjoe_edge']:+.1f}",     "Favors": "Favorite" if up['adjoe_edge'] > 0 else "Underdog"},
            {"Factor": "Defensive Efficiency (AdjDE) Δ",      "Value": f"{up['adjde_edge']:+.1f}",     "Favors": "Favorite" if up['adjde_edge'] > 0 else "Underdog"},
            {"Factor": "Roster Continuity Δ (Dog − Fav)",     "Value": f"{up['continuity_edge']:+.1f}%","Favors": "Underdog" if up['continuity_edge'] > 0 else "Favorite"},
        ])
        st.dataframe(factors, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### ALL FIRST-ROUND UPSET PROBABILITIES")
    upset_rows = []
    for region in REGIONS:
        reg_df = df_bracket[df_bracket["Region"] == region].sort_values("Seed")
        pairs  = [(1,16),(2,15),(3,14),(4,13),(5,12),(6,11),(7,10),(8,9)]
        seeded = {row["Seed"]: row for _, row in reg_df.iterrows()}
        for (s1, s2) in pairs:
            if s1 in seeded and s2 in seeded:
                fav = seeded[s1]; dog = seeded[s2]
                up  = upset_probability(
                    fav["Seed"], dog["Seed"],
                    fav["KenPom"], dog["KenPom"],
                    fav.get("SOS", 0.55),      dog.get("SOS", 0.55),
                    fav.get("Continuity", 70), dog.get("Continuity", 70),
                    fav.get("AdjOE", 105),     dog.get("AdjOE", 105),
                    fav.get("AdjDE", 100),     dog.get("AdjDE", 100),
                )
                upset_rows.append({
                    "Region":       region,
                    "Matchup":      f"({s1}) vs ({s2})",
                    "Favorite":     fav["Team"],
                    "Underdog":     dog["Team"],
                    "UpsetProb":    f"{up['upset_prob']*100:.1f}%",
                    "Severity":     up["severity"],
                    "KenPom Δ":     up["kenpom_edge"],
                    "AdjOE Δ":      up["adjoe_edge"],
                    "AdjDE Δ":      up["adjde_edge"],
                    "Continuity Δ": f"{up['continuity_edge']:+.1f}%",
                })
    st.dataframe(pd.DataFrame(upset_rows), use_container_width=True)


# ─── Tab 4: Seed History ───────────────────────────────────────────────────────
with tab4:
    st.markdown("### HISTORICAL SEED PERFORMANCE (1985 – 2024)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Final Four Appearances by Seed")
        st.bar_chart(df_history.set_index("Seed")["FF_Appearances"],
                     use_container_width=True, height=300)
    with c2:
        st.markdown("#### Championships by Seed")
        st.bar_chart(df_history.set_index("Seed")["Championships"],
                     use_container_width=True, height=300)

    st.markdown("#### Round-by-Round Win Rates")
    st.dataframe(
        df_history[["Seed","R32_Rate","Sweet16_Rate","FF_Appearances","Championships"]]
            .style
            .background_gradient(subset=["R32_Rate","Sweet16_Rate"], cmap="YlOrRd")
            .format({"R32_Rate": "{:.1%}", "Sweet16_Rate": "{:.1%}"}),
        use_container_width=True,
    )

    st.markdown("#### Notable Historical Upsets")
    notable = pd.DataFrame([
        {"Year": 2018, "Matchup": "(1) Virginia vs (16) UMBC",                "Winner": "UMBC",             "Seed Diff": 15},
        {"Year": 2023, "Matchup": "(2) Arizona vs (15) Princeton",             "Winner": "Princeton",        "Seed Diff": 13},
        {"Year": 2022, "Matchup": "(2) Kentucky vs (15) Saint Peter's",        "Winner": "Saint Peter's",    "Seed Diff": 13},
        {"Year": 2021, "Matchup": "(1) Illinois vs (15) Oral Roberts",         "Winner": "Oral Roberts",     "Seed Diff": 14},
        {"Year": 2016, "Matchup": "(2) Michigan State vs (15) Mid. Tennessee", "Winner": "Middle Tennessee", "Seed Diff": 13},
        {"Year": 2013, "Matchup": "(2) Georgetown vs (15) FGCU",               "Winner": "FGCU",             "Seed Diff": 13},
    ])
    st.dataframe(notable, use_container_width=True)


# ─── Tab 5: Monte Carlo Simulation ────────────────────────────────────────────
with tab5:
    st.markdown("### MONTE CARLO CHAMPIONSHIP SIMULATION")
    st.markdown("Each game resolved via logistic model: AdjOE · AdjDE · Continuity · SOS · KenPom.")

    if run_sim:
        with st.spinner(f"Running {n_simulations:,} tournament simulations..."):
            df_sim = simulate_tournament(df_bracket, n_sims=n_simulations)

        st.markdown(f"#### Championship Probability — Top 20 Teams ({n_simulations:,} simulations)")

        sim_cols = [c for c in
            ["Seed","Region","Team","Conference","KenPom",
             "AdjOE","AdjDE","AdjEM","Continuity","ChampionshipProb"]
            if c in df_sim.columns]

        fmt_sim = {"ChampionshipProb": "{:.2%}"}
        if "AdjEM"      in sim_cols: fmt_sim["AdjEM"]      = "{:+.1f}"
        if "Continuity" in sim_cols: fmt_sim["Continuity"] = "{:.1f}%"
        if "AdjOE"      in sim_cols: fmt_sim["AdjOE"]      = "{:.1f}"
        if "AdjDE"      in sim_cols: fmt_sim["AdjDE"]      = "{:.1f}"

        top20 = df_sim.head(20)[sim_cols]
        styled_sim = top20.style.background_gradient(subset=["ChampionshipProb"], cmap="YlOrRd")
        if "AdjOE" in sim_cols:
            styled_sim = styled_sim.background_gradient(subset=["AdjOE"], cmap="Greens")
        if "AdjDE" in sim_cols:
            styled_sim = styled_sim.background_gradient(subset=["AdjDE"], cmap="RdYlGn_r")
        styled_sim = styled_sim.format(fmt_sim)
        st.dataframe(styled_sim, use_container_width=True)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Championship Probability by Seed")
            seed_prob = df_sim.groupby("Seed")["ChampionshipProb"].sum().reset_index()
            st.bar_chart(seed_prob.set_index("Seed"), use_container_width=True, height=260)

        with col_s2:
            st.markdown("#### Championship Probability by Conference")
            if "Conference" in df_sim.columns:
                conf_prob = (df_sim.groupby("Conference")["ChampionshipProb"]
                             .sum().sort_values(ascending=False).reset_index())
                st.bar_chart(conf_prob.set_index("Conference"),
                             use_container_width=True, height=260)

        champ = df_sim.iloc[0]
        st.markdown(f"""
        <div style='background:#0F2847;border:2px solid #FFD166;border-radius:6px;
                    padding:1.5rem;text-align:center;margin-top:1rem;'>
            <div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;
                        color:#FF6B35;letter-spacing:3px;'>MOST LIKELY CHAMPION</div>
            <div style='font-family:Bebas Neue,sans-serif;font-size:3rem;
                        color:#FFD166;letter-spacing:4px;'>{champ['Team']}</div>
            <div style='font-family:IBM Plex Mono,monospace;font-size:0.85rem;color:#aaa;'>
                ({champ['Seed']}) Seed · {champ.get('Region','?')} ·
                {champ['ChampionshipProb']*100:.1f}% championship prob ·
                AdjEM {champ.get('AdjEM', 0):+.1f} ·
                Continuity {champ.get('Continuity', 0):.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Configure parameters in the sidebar and press **▶ RUN SIMULATION**.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center;font-family:IBM Plex Mono,monospace;
            font-size:0.65rem;color:#555;letter-spacing:2px;padding:1rem 0;'>
MARCH MADNESS ANALYZER v2 ● DATA MODE: {data_mode} ●
SOURCES: BARTTORVIK T-RANK + ESPN BRACKET API ●
UPSET MODEL: LOGISTIC REGRESSION + ADJOE + ADJDE + CONTINUITY + SOS ●
NOT FOR WAGERING PURPOSES
</div>
""", unsafe_allow_html=True)
