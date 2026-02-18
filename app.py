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

# ── Bracket Visualization Helpers ─────────────────────────────────────────────
def win_prob_color(prob: float) -> tuple:
    """
    Map a win probability (0–1) to a background color and text color.
    >= 0.75 → deep green    (certain favorite)
    0.55–0.75 → light green (moderate favorite)
    0.45–0.55 → gold        (toss-up)
    0.25–0.45 → light red   (moderate underdog)
    <  0.25   → deep red    (heavy underdog)
    """
    if prob >= 0.75:
        return ("#0a5c2e", "#06D6A0", "●●●●")   # bg, text, dots
    elif prob >= 0.55:
        return ("#1a472a", "#4ade80", "●●●○")
    elif prob >= 0.45:
        return ("#3d2e00", "#FFD166", "●●○○")
    elif prob >= 0.25:
        return ("#5c1a1a", "#f87171", "●●○○")
    else:
        return ("#6b0f0f", "#EF476F", "●○○○")


def build_bracket_state(df: pd.DataFrame) -> dict:
    """
    Build a bracket state dict keyed by region.
    Returns matchups for Round of 64, with win probabilities computed per slot.
    Advancement status defaults to None (not yet played).
    """
    bracket = {}
    first_round_pairs = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]

    for region in REGIONS:
        reg_df = df[df["Region"] == region].sort_values("Seed")
        seeded = {int(row["Seed"]): row for _, row in reg_df.iterrows()}
        matchups = []
        for (s1, s2) in first_round_pairs:
            if s1 in seeded and s2 in seeded:
                t1 = seeded[s1]
                t2 = seeded[s2]
                up = upset_probability(
                    t1["Seed"], t2["Seed"],
                    t1["KenPom"], t2["KenPom"],
                    t1.get("SOS", 0.55),        t2.get("SOS", 0.55),
                    t1.get("Continuity", 70),   t2.get("Continuity", 70),
                    t1.get("AdjOE", 105),        t2.get("AdjOE", 105),
                    t1.get("AdjDE", 100),        t2.get("AdjDE", 100),
                )
                matchups.append({
                    "fav":      t1,
                    "dog":      t2,
                    "fav_prob": up["fav_prob"],
                    "dog_prob": up["upset_prob"],
                    "winner":   None,    # None = not played
                })
        bracket[region] = matchups
    return bracket


def team_slot_html(team_row, prob: float, is_winner: bool = False,
                   is_eliminated: bool = False) -> str:
    """Render a single team slot with color-coded win probability."""
    bg, fg, dots = win_prob_color(prob)
    name    = str(team_row.get("Team", "TBD"))
    seed    = int(team_row.get("Seed", 0))
    adjem   = float(team_row.get("AdjEM", 0))
    cont    = float(team_row.get("Continuity", 0))
    pct_str = f"{prob*100:.0f}%"

    # Override color states
    if is_winner:
        bg, fg = "#0a3d1f", "#06D6A0"
    if is_eliminated:
        bg, fg = "#1a0a0a", "#555"

    adv_icon = "✓ ADV" if is_winner else ("✗ ELIM" if is_eliminated else "")
    adv_style = "color:#06D6A0;" if is_winner else "color:#666;"

    return f"""
    <div style="
        background:{bg};
        border-left: 3px solid {fg};
        border-radius: 3px;
        padding: 5px 8px;
        margin: 2px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 36px;
        font-family: 'IBM Plex Mono', monospace;
        position: relative;
    ">
        <div style="display:flex;align-items:center;gap:6px;">
            <span style="color:{fg};font-size:0.65rem;font-weight:700;
                         min-width:18px;text-align:center;">{seed}</span>
            <span style="color:#e0e0e0;font-size:0.72rem;font-weight:500;
                         white-space:nowrap;overflow:hidden;max-width:120px;
                         text-overflow:ellipsis;" title="{name}">{name}</span>
        </div>
        <div style="display:flex;align-items:center;gap:5px;">
            <span style="font-size:0.65rem;{adv_style}font-weight:700;">{adv_icon}</span>
            <span style="
                background:{fg}22;
                color:{fg};
                font-size:0.65rem;
                font-weight:700;
                padding:2px 5px;
                border-radius:2px;
                letter-spacing:1px;
            ">{pct_str}</span>
        </div>
    </div>"""


def matchup_html(matchup: dict) -> str:
    """Render a full matchup (two team slots + connector line)."""
    fav  = matchup["fav"]
    dog  = matchup["dog"]
    fp   = matchup["fav_prob"]
    dp   = matchup["dog_prob"]
    winner = matchup.get("winner")

    fav_won  = winner == str(fav.get("Team", ""))
    dog_won  = winner == str(dog.get("Team", ""))
    fav_elim = dog_won
    dog_elim = fav_won

    slot1 = team_slot_html(fav, fp, is_winner=fav_won, is_eliminated=fav_elim)
    slot2 = team_slot_html(dog, dp, is_winner=dog_won, is_eliminated=dog_elim)

    return f"""
    <div style="
        background:#0a1628;
        border:1px solid #1e3a5f;
        border-radius:4px;
        padding:6px;
        margin-bottom:8px;
    ">
        {slot1}
        <div style="height:1px;background:#1e3a5f;margin:2px 0;"></div>
        {slot2}
    </div>"""


def region_bracket_html(region: str, matchups: list) -> str:
    """Render all first-round matchups for a region."""
    cards = "".join(matchup_html(m) for m in matchups)
    return f"""
    <div style="flex:1;min-width:260px;">
        <div style="
            font-family:'Bebas Neue',sans-serif;
            font-size:1.2rem;
            color:#FF6B35;
            letter-spacing:3px;
            padding:4px 0 8px 0;
            border-bottom:2px solid #FF6B35;
            margin-bottom:10px;
        ">{region.upper()}</div>
        {cards}
    </div>"""


def render_color_legend():
    """Render the win probability color legend using native Streamlit components."""
    entries = [
        ("🟢", "#06D6A0", "≥ 75%",   "Heavy Favorite"),
        ("🟩", "#4ade80", "55 – 74%", "Moderate Favorite"),
        ("🟡", "#FFD166", "45 – 54%", "Toss-Up"),
        ("🟥", "#f87171", "25 – 44%", "Moderate Underdog"),
        ("🔴", "#EF476F", "< 25%",    "Heavy Underdog"),
    ]
    st.markdown("**WIN PROBABILITY COLOR KEY**")
    cols = st.columns(5)
    for col, (icon, color, pct, label) in zip(cols, entries):
        with col:
            st.markdown(
                f"<div style='background:{color}22;border-left:4px solid {color};"
                f"border-radius:3px;padding:8px 10px;text-align:center;'>"
                f"<div style='font-size:1.2rem;'>{icon}</div>"
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;"
                f"color:{color};font-weight:700;'>{pct}</div>"
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.62rem;"
                f"color:#aaa;margin-top:2px;'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    st.caption("✓ ADV = Advanced after game played · ✗ ELIM = Eliminated · Reset button clears all results")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 BRACKET FIELD",
    "🗂 LIVE BRACKET",
    "🏆 BRACKET PICKER",
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


# ─── Tab 2: Live Bracket ──────────────────────────────────────────────────────
with tab2:

    # ── March Madness Header Banner ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1f3c 0%,#1a3a6b 60%,#0d2a4a 100%);
                border:2px solid #FF6B35;border-radius:10px;padding:22px 30px;
                text-align:center;margin-bottom:18px;position:relative;
                box-shadow:0 4px 24px #FF6B3544;">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:3rem;
                    color:#FF6B35;letter-spacing:8px;line-height:1.05;
                    text-shadow:0 2px 12px #FF6B3588;">
            MARCH MADNESS
        </div>
        <div style="font-family:'Bebas Neue',sans-serif;font-size:1.5rem;
                    color:#FFD166;letter-spacing:5px;margin-top:2px;">
            LIVE BRACKET TRACKER
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                    color:#aaa;letter-spacing:3px;margin-top:8px;text-transform:uppercase;">
            2025 NCAA Division I Men's Basketball Tournament
        </div>
        <div style="display:flex;justify-content:center;gap:24px;margin-top:12px;
                    flex-wrap:wrap;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                          color:#06D6A0;border:1px solid #06D6A0;padding:3px 10px;
                          border-radius:3px;letter-spacing:2px;">R1: MAR 20-21</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                          color:#FFD166;border:1px solid #FFD166;padding:3px 10px;
                          border-radius:3px;letter-spacing:2px;">R2: MAR 22-23</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                          color:#FF6B35;border:1px solid #FF6B35;padding:3px 10px;
                          border-radius:3px;letter-spacing:2px;">S16: MAR 27-28</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                          color:#EF476F;border:1px solid #EF476F;padding:3px 10px;
                          border-radius:3px;letter-spacing:2px;">E8: MAR 29-30</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                          color:#c084fc;border:1px solid #c084fc;padding:3px 10px;
                          border-radius:3px;letter-spacing:2px;">F4 + CHAMP: APR 5/7</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "bracket_state" not in st.session_state:
        st.session_state.bracket_state = build_bracket_state(df_bracket)

    ctrl_col1, ctrl_col2 = st.columns([1, 4])
    with ctrl_col1:
        if st.button("🔄 Reset All Results", key="reset_bracket",
                     use_container_width=True):
            st.session_state.bracket_state = build_bracket_state(df_bracket)

    with ctrl_col2:
        render_color_legend()

    bracket_state = st.session_state.bracket_state

    # ── Styled region tabs ──
    reg_tabs = st.tabs([f"◈ {r.upper()}" for r in REGIONS])

    for reg_tab, region in zip(reg_tabs, REGIONS):
        with reg_tab:
            matchups = bracket_state.get(region, [])
            if not matchups:
                st.warning(f"No matchup data for {region}.")
                continue

            # Region sub-header
            st.markdown(f"""
            <div style="background:linear-gradient(90deg,#FF6B3522,transparent);
                        border-left:4px solid #FF6B35;padding:10px 16px;
                        margin-bottom:16px;border-radius:0 4px 4px 0;">
                <span style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;
                             color:#FF6B35;letter-spacing:4px;">{region.upper()} REGION</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                             color:#aaa;margin-left:12px;letter-spacing:2px;">ROUND OF 64</span>
            </div>""", unsafe_allow_html=True)

            # Two-column layout: controls left, visual cards right
            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.markdown("""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                            color:#FFD166;letter-spacing:2px;margin-bottom:10px;">
                    ▶ SELECT GAME WINNERS
                </div>""", unsafe_allow_html=True)

                for i, m in enumerate(matchups):
                    fav_name = str(m["fav"].get("Team","Team A"))
                    dog_name = str(m["dog"].get("Team","Team B"))
                    fav_seed = int(m["fav"].get("Seed",0))
                    dog_seed = int(m["dog"].get("Seed",0))
                    current  = m.get("winner")
                    options  = ["— Not Played —", fav_name, dog_name]
                    def_idx  = 0
                    if current == fav_name: def_idx = 1
                    elif current == dog_name: def_idx = 2

                    st.markdown(f"""
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                                color:#888;letter-spacing:1px;margin-top:8px;">
                        GAME {i+1} · ({fav_seed}) vs ({dog_seed})
                    </div>""", unsafe_allow_html=True)

                    choice = st.selectbox(
                        f"G{i+1}", options, index=def_idx,
                        key=f"w_{region}_{i}", label_visibility="collapsed"
                    )
                    bracket_state[region][i]["winner"] = (
                        None if "Not Played" in choice else choice
                    )

            with right_col:
                st.markdown("""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                            color:#FFD166;letter-spacing:2px;margin-bottom:10px;">
                    ◈ MATCHUP PROBABILITY CARDS
                </div>""", unsafe_allow_html=True)
                for m in bracket_state[region]:
                    st.markdown(matchup_html(m), unsafe_allow_html=True)

            # Full region visual bracket card
            st.markdown("---")
            adv_count = sum(1 for m in bracket_state[region] if m.get("winner"))
            upsets    = sum(1 for m in bracket_state[region]
                           if m.get("winner") and
                           m.get("winner") == str(m["dog"].get("Team","")))
            kc1, kc2, kc3 = st.columns(3)
            for col, (val, label) in zip([kc1,kc2,kc3], [
                (f"{adv_count}/8",  "Games Played"),
                (f"{8-adv_count}",  "Games Remaining"),
                (f"{upsets}",       "Upsets So Far"),
            ]):
                with col:
                    st.markdown(f"""
                    <div class='metric-card' style='padding:8px;'>
                        <div class='metric-value' style='font-size:1.6rem;'>{val}</div>
                        <div class='metric-label'>{label}</div>
                    </div>""", unsafe_allow_html=True)

    # ── Full 4-region bracket visual ──
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Bebas Neue',sans-serif;font-size:1.8rem;
                color:#FFD166;letter-spacing:5px;margin-bottom:12px;">
        ◈ COMPLETE BRACKET — ALL REGIONS
    </div>""", unsafe_allow_html=True)

    all_html = "".join(
        region_bracket_html(r, bracket_state.get(r,[])) for r in REGIONS
    )
    st.markdown(f"""
    <div style="display:flex;gap:14px;flex-wrap:wrap;
                background:linear-gradient(135deg,#071429,#0d1f3c);
                border:2px solid #FF6B35;border-radius:8px;
                padding:20px;overflow-x:auto;
                box-shadow:0 4px 24px #00000055;">
        {all_html}
    </div>""", unsafe_allow_html=True)

    # ── Probability summary table ──
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;
                color:#FF6B35;letter-spacing:4px;margin-bottom:10px;">
        WIN PROBABILITY SUMMARY
    </div>""", unsafe_allow_html=True)

    prob_rows = []
    for region in REGIONS:
        for m in bracket_state.get(region,[]):
            fav = m["fav"]; dog = m["dog"]
            prob_rows.append({
                "Region":   region,
                "Matchup":  f"({int(fav['Seed'])}) vs ({int(dog['Seed'])})",
                "Favorite": fav.get("Team",""),
                "Fav Win%": m["fav_prob"],
                "Underdog": dog.get("Team",""),
                "Dog Win%": m["dog_prob"],
                "Result":   m.get("winner") or "—",
                "Upset?":   "✓ UPSET" if (
                    m.get("winner") and
                    m.get("winner") == dog.get("Team","")
                ) else "—",
            })
    df_probs = pd.DataFrame(prob_rows)
    if not df_probs.empty:
        st.dataframe(
            df_probs.style
                .background_gradient(subset=["Fav Win%"], cmap="Greens")
                .background_gradient(subset=["Dog Win%"], cmap="Reds")
                .format({"Fav Win%": "{:.1%}", "Dog Win%": "{:.1%}"}),
            use_container_width=True, height=420,
        )


# ─── Tab 3: Bracket Picker ────────────────────────────────────────────────────
with tab3:

    # ── Header ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1f3c 0%,#1a3a6b 60%,#0d2a4a 100%);
                border:2px solid #FFD166;border-radius:10px;padding:22px 30px;
                text-align:center;margin-bottom:20px;
                box-shadow:0 4px 24px #FFD16633;">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:2.8rem;
                    color:#FFD166;letter-spacing:8px;line-height:1.05;">
            BRACKET PICKER
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                    color:#aaa;letter-spacing:3px;margin-top:6px;">
            BUILD · SAVE · COMPARE MULTIPLE BRACKETS
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Multi-bracket management ──
    if "saved_brackets" not in st.session_state:
        st.session_state.saved_brackets = {}   # name → picks dict
    if "active_picker" not in st.session_state:
        st.session_state.active_picker = {}    # current working picks

    mgmt_col1, mgmt_col2, mgmt_col3, mgmt_col4 = st.columns([2,1,1,1])

    with mgmt_col1:
        bracket_name = st.text_input(
            "Bracket Name", value="My Bracket",
            placeholder="Enter a name for this bracket...",
            key="bracket_name_input"
        )

    with mgmt_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save Bracket", use_container_width=True):
            if bracket_name.strip():
                st.session_state.saved_brackets[bracket_name.strip()] = \
                    dict(st.session_state.active_picker)
                st.success(f"Saved: {bracket_name.strip()}")

    with mgmt_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 Clear Picks", use_container_width=True):
            st.session_state.active_picker = {}
            st.rerun()

    with mgmt_col4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.saved_brackets:
            load_name = st.selectbox(
                "Load Bracket",
                ["— Select —"] + list(st.session_state.saved_brackets.keys()),
                key="load_bracket_select",
                label_visibility="collapsed"
            )
            if load_name != "— Select —":
                st.session_state.active_picker = \
                    dict(st.session_state.saved_brackets[load_name])

    # ── Saved brackets summary ──
    if st.session_state.saved_brackets:
        st.markdown("---")
        st.markdown("""
        <div style="font-family:'Bebas Neue',sans-serif;font-size:1.2rem;
                    color:#FF6B35;letter-spacing:3px;margin-bottom:8px;">
            SAVED BRACKETS
        </div>""", unsafe_allow_html=True)

        sb_cols = st.columns(min(len(st.session_state.saved_brackets), 4))
        for col, (bname, bpicks) in zip(
            sb_cols, st.session_state.saved_brackets.items()
        ):
            total_picks = len(bpicks)
            with col:
                st.markdown(f"""
                <div style="background:#0F2847;border:1px solid #FFD166;
                            border-radius:4px;padding:10px;text-align:center;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                                color:#FFD166;font-weight:700;white-space:nowrap;
                                overflow:hidden;text-overflow:ellipsis;">{bname}</div>
                    <div style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;
                                color:#FF6B35;">{total_picks}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                                color:#aaa;">GAMES PICKED</div>
                </div>""", unsafe_allow_html=True)

        # Delete a bracket
        del_name = st.selectbox(
            "Delete a saved bracket:",
            ["— Select to delete —"] + list(st.session_state.saved_brackets.keys()),
            key="del_bracket"
        )
        if del_name != "— Select to delete —":
            if st.button(f"🗑 Delete '{del_name}'", key="confirm_delete"):
                del st.session_state.saved_brackets[del_name]
                st.rerun()

    st.markdown("---")

    # ── Visual bracket picker — region by region ──
    # Build seed lookup from bracket data
    def get_region_seeds(region: str) -> dict:
        reg_df = df_bracket[df_bracket["Region"] == region].sort_values("Seed")
        return {int(row["Seed"]): row for _, row in reg_df.iterrows()}

    ROUND_PAIRS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]

    def picker_slot_html(seed: int, name: str, prob: float,
                         picked: bool, eliminated: bool) -> str:
        bg, fg, _ = win_prob_color(prob)
        if picked:
            bg = "#0a3d1f"; fg = "#06D6A0"; border = "2px solid #06D6A0"
        elif eliminated:
            bg = "#111"; fg = "#444"; border = "1px solid #333"
        else:
            border = f"1px solid {fg}"

        short_name = name[:18] + "…" if len(name) > 18 else name
        pct = f"{prob*100:.0f}%"
        return f"""<div style="background:{bg};border:{border};border-radius:3px;
            padding:5px 8px;margin:1px 0;display:flex;align-items:center;
            justify-content:space-between;min-height:34px;cursor:pointer;
            font-family:'IBM Plex Mono',monospace;transition:all 0.15s;">
            <div style="display:flex;align-items:center;gap:5px;">
                <span style="color:{fg};font-size:0.62rem;font-weight:700;
                             min-width:16px;">{seed}</span>
                <span style="color:{'#06D6A0' if picked else ('#555' if eliminated else '#ddd')};
                             font-size:0.68rem;" title="{name}">{short_name}</span>
            </div>
            <span style="color:{fg};font-size:0.6rem;font-weight:700;
                         background:{fg}22;padding:1px 5px;border-radius:2px;">{pct}</span>
        </div>"""

    def render_picker_region(region: str, seeded: dict):
        """Render one region's bracket picker with clickable team slots."""
        st.markdown(f"""
        <div style="background:linear-gradient(90deg,#FFD16622,transparent);
                    border-left:4px solid #FFD166;padding:8px 14px;
                    margin-bottom:12px;border-radius:0 4px 4px 0;">
            <span style="font-family:'Bebas Neue',sans-serif;font-size:1.5rem;
                         color:#FFD166;letter-spacing:4px;">{region.upper()}</span>
        </div>""", unsafe_allow_html=True)

        for pair_idx, (s1, s2) in enumerate(ROUND_PAIRS):
            if s1 not in seeded or s2 not in seeded:
                continue
            t1 = seeded[s1]; t2 = seeded[s2]

            # Compute probs
            up = upset_probability(
                t1["Seed"], t2["Seed"],
                t1["KenPom"], t2["KenPom"],
                t1.get("SOS",0.55), t2.get("SOS",0.55),
                t1.get("Continuity",70), t2.get("Continuity",70),
                t1.get("AdjOE",105), t2.get("AdjOE",105),
                t1.get("AdjDE",100), t2.get("AdjDE",100),
            )

            game_key  = f"{region}_R1_G{pair_idx}"
            picked    = st.session_state.active_picker.get(game_key)
            t1_name   = str(t1.get("Team",""))
            t2_name   = str(t2.get("Team",""))

            t1_picked = picked == t1_name
            t2_picked = picked == t2_name
            t1_elim   = t2_picked
            t2_elim   = t1_picked

            st.markdown(f"""
            <div style="background:#0a1628;border:1px solid #1e3a5f;
                        border-radius:4px;padding:5px;margin-bottom:6px;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
                            color:#555;letter-spacing:1px;padding:2px 4px;">
                    GAME {pair_idx+1} · {region[:1].upper()} REGION
                </div>
                {picker_slot_html(s1,t1_name,up['fav_prob'],t1_picked,t1_elim)}
                <div style="height:1px;background:#1e3a5f;margin:2px 0;"></div>
                {picker_slot_html(s2,t2_name,up['upset_prob'],t2_picked,t2_elim)}
            </div>""", unsafe_allow_html=True)

            # Picker controls
            pick_col1, pick_col2 = st.columns(2)
            with pick_col1:
                if st.button(
                    f"✓ ({s1}) {t1_name[:14]}",
                    key=f"pick_{region}_{pair_idx}_t1",
                    use_container_width=True,
                    type="primary" if t1_picked else "secondary"
                ):
                    st.session_state.active_picker[game_key] = t1_name
                    st.rerun()
            with pick_col2:
                if st.button(
                    f"✓ ({s2}) {t2_name[:14]}",
                    key=f"pick_{region}_{pair_idx}_t2",
                    use_container_width=True,
                    type="primary" if t2_picked else "secondary"
                ):
                    st.session_state.active_picker[game_key] = t2_name
                    st.rerun()

    # ── Render all 4 regions in a 2x2 grid ──
    st.markdown("""
    <div style="font-family:'Bebas Neue',sans-serif;font-size:1.3rem;
                color:#FF6B35;letter-spacing:4px;margin-bottom:12px;">
        ◈ ROUND OF 64 — PICK YOUR WINNERS
    </div>""", unsafe_allow_html=True)

    # Progress indicator
    total_r1_games = len(ROUND_PAIRS) * 4
    picked_count   = len(st.session_state.active_picker)
    progress_pct   = picked_count / total_r1_games if total_r1_games > 0 else 0
    st.progress(progress_pct,
                text=f"Round of 64 Progress: {picked_count}/{total_r1_games} games picked")

    # Region columns — left pair and right pair like a real bracket
    left_regions  = [REGIONS[0], REGIONS[2]]  # East, South
    right_regions = [REGIONS[1], REGIONS[3]]  # West, Midwest

    picker_left, picker_divider, picker_right = st.columns([5, 0.3, 5])

    with picker_left:
        for region in left_regions:
            seeded = get_region_seeds(region)
            render_picker_region(region, seeded)

    with picker_divider:
        st.markdown("""
        <div style="width:2px;background:linear-gradient(to bottom,
              transparent,#FF6B35,#FFD166,#FF6B35,transparent);
              min-height:600px;margin:auto;"></div>""",
        unsafe_allow_html=True)

    with picker_right:
        for region in right_regions:
            seeded = get_region_seeds(region)
            render_picker_region(region, seeded)

    # ── Picked winners summary ──
    if st.session_state.active_picker:
        st.markdown("---")
        st.markdown("""
        <div style="font-family:'Bebas Neue',sans-serif;font-size:1.3rem;
                    color:#06D6A0;letter-spacing:4px;margin-bottom:10px;">
            YOUR ROUND OF 64 PICKS
        </div>""", unsafe_allow_html=True)

        pick_summary = []
        for game_key, winner_name in st.session_state.active_picker.items():
            parts  = game_key.split("_")
            region = parts[0]
            g_idx  = int(parts[-1].replace("G",""))
            s1, s2 = ROUND_PAIRS[g_idx]
            seeded = get_region_seeds(region)
            if s1 in seeded and s2 in seeded:
                t1 = seeded[s1]; t2 = seeded[s2]
                is_upset = winner_name == str(t2.get("Team",""))
                up = upset_probability(
                    t1["Seed"], t2["Seed"],
                    t1["KenPom"], t2["KenPom"],
                    t1.get("SOS",0.55), t2.get("SOS",0.55),
                    t1.get("Continuity",70), t2.get("Continuity",70),
                    t1.get("AdjOE",105), t2.get("AdjOE",105),
                    t1.get("AdjDE",100), t2.get("AdjDE",100),
                )
                win_prob = up["upset_prob"] if is_upset else up["fav_prob"]
                pick_summary.append({
                    "Region":   region,
                    "Matchup":  f"({s1}) vs ({s2})",
                    "Your Pick": winner_name,
                    "Win Prob": win_prob,
                    "Upset Pick?": "🔥 YES" if is_upset else "—",
                })

        if pick_summary:
            df_picks = pd.DataFrame(pick_summary).sort_values("Region")
            st.dataframe(
                df_picks.style
                    .background_gradient(subset=["Win Prob"], cmap="RdYlGn")
                    .format({"Win Prob": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )
            upset_picks = sum(1 for p in pick_summary if "YES" in p["Upset Pick?"])
            avg_prob    = sum(p["Win Prob"] for p in pick_summary) / len(pick_summary)
            s1, s2, s3  = st.columns(3)
            for col, (val, label) in zip([s1,s2,s3],[
                (f"{len(pick_summary)}", "Games Picked"),
                (f"{upset_picks}",       "Upset Picks"),
                (f"{avg_prob*100:.1f}%", "Avg Win Prob"),
            ]):
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{val}</div>
                        <div class='metric-label'>{label}</div>
                    </div>""", unsafe_allow_html=True)


# ─── Tab 4: Efficiency Dashboard ──────────────────────────────────────────────
with tab4:
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


# ─── Tab 5: Upset Analyzer ────────────────────────────────────────────────────
with tab5:
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


# ─── Tab 6: Seed History ───────────────────────────────────────────────────────
with tab6:
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


# ─── Tab 7: Monte Carlo Simulation ────────────────────────────────────────────
with tab7:
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
