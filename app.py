"""
March Madness Analyzer — app.py
Streamlit application for bracket analysis, team statistics,
historical seed performance, and upset probability modeling.

Dependencies: streamlit, pandas, requests
Data source: Public sports-reference / college basketball APIs
             (falls back to synthetic demo data if offline)
"""

import streamlit as st
import pandas as pd
import requests
import json
import math
import random
from datetime import datetime

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

# Historical upset rates by matchup seed (higher_seed vs lower_seed)
HISTORICAL_UPSET_RATES = {
    (1,16): 0.013, (2,15): 0.063, (3,14): 0.152, (4,13): 0.202,
    (5,12): 0.359, (6,11): 0.370, (7,10): 0.392, (8,9):  0.489,
}

# ── Synthetic Demo Data ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def generate_demo_bracket() -> pd.DataFrame:
    """Generate a realistic 68-team bracket with synthetic stats."""
    random.seed(42)
    teams = []
    play_in_pairs = [(16, "East"), (16, "West"), (11, "Midwest"), (11, "South")]

    seed_slot = 1
    for region in REGIONS:
        for seed in range(1, 17):
            conference = random.choice(CONFERENCES)
            # Simulate plausible stats correlated with seed
            base_off  = 115 - (seed * 2.5) + random.uniform(-3, 3)
            base_def  = 90  + (seed * 1.8) + random.uniform(-3, 3)
            win_pct   = max(0.40, min(0.97, 1.0 - (seed * 0.035) + random.uniform(-0.05, 0.05)))
            sos       = round(random.uniform(0.45, 0.72), 3)
            kenpom    = round(30 - (seed * 1.9) + random.uniform(-4, 4), 1)

            teams.append({
                "Seed":       seed,
                "Region":     region,
                "Team":       f"Team {chr(64+seed)}-{region[:2]}",
                "Conference": conference,
                "Record":     f"{int(win_pct*30)}-{30-int(win_pct*30)}",
                "Win%":       round(win_pct, 3),
                "OffRtg":     round(base_off, 1),
                "DefRtg":     round(base_def, 1),
                "NetRtg":     round(base_off - base_def, 1),
                "SOS":        sos,
                "KenPom":     kenpom,
                "3P%":        round(random.uniform(0.31, 0.40), 3),
                "TOV%":       round(random.uniform(14, 22), 1),
                "RebMgn":     round(random.uniform(-3, 8), 1),
            })
    return pd.DataFrame(teams)


@st.cache_data(ttl=3600)
def historical_seed_performance() -> pd.DataFrame:
    """Returns historical Final Four appearances and championship wins by seed (1985–2024)."""
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
) -> dict:
    """
    Computes upset probability using a logistic model blending:
      - Historical base rate for the seed matchup
      - KenPom efficiency differential
      - Strength-of-schedule differential
    """
    matchup = (min(fav_seed, dog_seed), max(fav_seed, dog_seed))
    base_rate = HISTORICAL_UPSET_RATES.get(matchup, 0.35)

    kenpom_diff = fav_kenpom - dog_kenpom  # positive = favorite better
    sos_diff    = fav_sos - dog_sos

    # Logit adjustment
    logit_base = math.log(base_rate / (1 - base_rate))
    logit_adj  = logit_base - (kenpom_diff * 0.08) - (sos_diff * 2.5)
    prob_upset  = 1 / (1 + math.exp(-logit_adj))
    prob_upset  = max(0.01, min(0.75, prob_upset))

    seed_diff = dog_seed - fav_seed
    severity  = "HIGH" if prob_upset > 0.40 else "MEDIUM" if prob_upset > 0.22 else "LOW"

    return {
        "upset_prob":   round(prob_upset, 4),
        "fav_prob":     round(1 - prob_upset, 4),
        "severity":     severity,
        "seed_diff":    seed_diff,
        "base_rate":    base_rate,
        "kenpom_edge":  round(kenpom_diff, 1),
    }


# ── Bracket Simulator ──────────────────────────────────────────────────────────
def simulate_tournament(df: pd.DataFrame, n_sims: int = 1000) -> pd.DataFrame:
    """Monte Carlo simulation of the tournament. Returns win probability per team."""
    results = {row["Team"]: 0 for _, row in df.iterrows()}

    for _ in range(n_sims):
        # Per region: seeds 1–16, single elimination
        champion = None
        region_winners = []

        for region in REGIONS:
            region_teams = df[df["Region"] == region].sort_values("Seed")
            bracket_order = [1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]
            seeded = {row["Seed"]: row for _, row in region_teams.iterrows()}
            pool = [seeded[s] for s in bracket_order if s in seeded]

            while len(pool) > 1:
                next_round = []
                for i in range(0, len(pool), 2):
                    if i+1 >= len(pool):
                        next_round.append(pool[i])
                        continue
                    t1, t2 = pool[i], pool[i+1]
                    fav = t1 if t1["KenPom"] >= t2["KenPom"] else t2
                    dog = t2 if fav["Team"] == t1["Team"] else t1
                    up  = upset_probability(
                        fav["Seed"], dog["Seed"],
                        fav["KenPom"], dog["KenPom"],
                        fav["SOS"], dog["SOS"]
                    )
                    winner = dog if random.random() < up["upset_prob"] else fav
                    next_round.append(winner)
                pool = next_round
            region_winners.append(pool[0])

        # Final Four
        ff_pool = region_winners[:]
        while len(ff_pool) > 1:
            next_round = []
            for i in range(0, len(ff_pool), 2):
                if i+1 >= len(ff_pool):
                    next_round.append(ff_pool[i])
                    continue
                t1, t2 = ff_pool[i], ff_pool[i+1]
                fav = t1 if t1["KenPom"] >= t2["KenPom"] else t2
                dog = t2 if fav["Team"] == t1["Team"] else t1
                up  = upset_probability(
                    fav["Seed"], dog["Seed"],
                    fav["KenPom"], dog["KenPom"],
                    fav["SOS"], dog["SOS"]
                )
                winner = dog if random.random() < up["upset_prob"] else fav
                next_round.append(winner)
            ff_pool = next_round

        champion = ff_pool[0]["Team"]
        results[champion] += 1

    df_sim = df.copy()
    df_sim["ChampionshipProb"] = df_sim["Team"].map(
        lambda t: round(results.get(t, 0) / n_sims, 4)
    )
    return df_sim.sort_values("ChampionshipProb", ascending=False)


# ── UI: Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ CONTROLS")
    selected_region  = st.selectbox("Region Filter", ["All"] + REGIONS)
    selected_round   = st.selectbox("Round", ROUNDS)
    n_simulations    = st.slider("Monte Carlo Simulations", 100, 5000, 1000, 100)
    seed_range       = st.slider("Seed Range", 1, 16, (1, 16))
    conf_filter      = st.multiselect("Conference Filter", CONFERENCES, default=CONFERENCES)
    run_sim          = st.button("▶ RUN SIMULATION", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div class='subtitle'>Data Notes</div>
    <div style='font-size:0.7rem; color:#aaa; font-family: IBM Plex Mono, monospace; margin-top:0.5rem;'>
    Demo mode: synthetic data<br>
    correlated with historical<br>
    seed distributions.<br><br>
    Upset model: logistic regression<br>
    + KenPom differential.
    </div>
    """, unsafe_allow_html=True)


# ── Load Data ──────────────────────────────────────────────────────────────────
df_bracket = generate_demo_bracket()
df_history = historical_seed_performance()

# Apply filters
df_filtered = df_bracket[
    (df_bracket["Seed"] >= seed_range[0]) &
    (df_bracket["Seed"] <= seed_range[1]) &
    (df_bracket["Conference"].isin(conf_filter))
]
if selected_region != "All":
    df_filtered = df_filtered[df_filtered["Region"] == selected_region]


# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_date = st.columns([5, 1])
with col_title:
    st.markdown("<div class='main-title'>MARCH MADNESS<br>ANALYZER</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>● NCAA Tournament Intelligence Platform ●</div>", unsafe_allow_html=True)
with col_date:
    st.markdown(f"""
    <div style='text-align:right; font-family: IBM Plex Mono, monospace;
                font-size:0.7rem; color:#FFD166; padding-top:1rem;'>
    {datetime.now().strftime('%Y · %b %d')}<br>
    <span style='color:#FF6B35'>DEMO MODE</span>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── KPI Row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
metrics = [
    ("68",   "Field Size"),
    ("4",    "Regions"),
    ("63",   "Games"),
    (f"{len(df_filtered)}", "Filtered Teams"),
    ("2025", "Season"),
]
for col, (val, label) in zip([k1,k2,k3,k4,k5], metrics):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 BRACKET FIELD",
    "🎯 UPSET ANALYZER",
    "📈 SEED HISTORY",
    "🎲 MONTE CARLO SIM",
])

# ─── Tab 1: Bracket Field ──────────────────────────────────────────────────────
with tab1:
    st.markdown("### FIELD AT A GLANCE")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        display_cols = ["Seed","Region","Team","Conference","Record",
                        "Win%","OffRtg","DefRtg","NetRtg","KenPom","SOS"]
        st.dataframe(
            df_filtered[display_cols].style
                .background_gradient(subset=["KenPom"], cmap="YlOrRd")
                .background_gradient(subset=["NetRtg"], cmap="RdYlGn")
                .format({"Win%": "{:.1%}", "SOS": "{:.3f}"}),
            use_container_width=True,
            height=500,
        )

    with col_b:
        st.markdown("#### KenPom Rank by Seed (Filtered)")
        seed_avg = (
            df_filtered.groupby("Seed")["KenPom"]
            .mean().reset_index()
            .rename(columns={"KenPom": "Avg KenPom"})
        )
        st.bar_chart(seed_avg.set_index("Seed"), use_container_width=True, height=220)

        st.markdown("#### Net Rating Distribution")
        st.bar_chart(
            df_filtered.set_index("Team")["NetRtg"].head(20),
            use_container_width=True, height=220
        )

# ─── Tab 2: Upset Analyzer ────────────────────────────────────────────────────
with tab2:
    st.markdown("### UPSET PROBABILITY MATRIX")
    st.markdown("Configure a head-to-head matchup to compute upset probability.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🏆 FAVORITE")
        teams_list = df_bracket["Team"].tolist()
        fav_team   = st.selectbox("Favorite Team", teams_list, key="fav")
        fav_data   = df_bracket[df_bracket["Team"] == fav_team].iloc[0]
        st.markdown(f"""
        <div class='metric-card'>
        <div class='metric-label'>Seed · {fav_data['Seed']} | KenPom · {fav_data['KenPom']}</div>
        <div style='font-size:0.9rem;color:#aaa;'>{fav_data['Conference']} | {fav_data['Record']}</div>
        <div class='metric-value' style='font-size:1.6rem;'>{fav_data['NetRtg']:+.1f}</div>
        <div class='metric-label'>NET RATING</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("##### 🐶 UNDERDOG")
        dog_team = st.selectbox("Underdog Team",
                                [t for t in teams_list if t != fav_team], key="dog")
        dog_data = df_bracket[df_bracket["Team"] == dog_team].iloc[0]
        st.markdown(f"""
        <div class='metric-card'>
        <div class='metric-label'>Seed · {dog_data['Seed']} | KenPom · {dog_data['KenPom']}</div>
        <div style='font-size:0.9rem;color:#aaa;'>{dog_data['Conference']} | {dog_data['Record']}</div>
        <div class='metric-value' style='font-size:1.6rem;'>{dog_data['NetRtg']:+.1f}</div>
        <div class='metric-label'>NET RATING</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("COMPUTE UPSET PROBABILITY"):
        up = upset_probability(
            fav_data["Seed"], dog_data["Seed"],
            fav_data["KenPom"], dog_data["KenPom"],
            fav_data["SOS"], dog_data["SOS"],
        )
        sev_class = f"upset-{up['severity'].lower()}"
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{up['upset_prob']*100:.1f}%</div>
                <div class='metric-label'>Upset Probability</div></div>""",
                unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{up['fav_prob']*100:.1f}%</div>
                <div class='metric-label'>Favorite Win Prob</div></div>""",
                unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{up['base_rate']*100:.1f}%</div>
                <div class='metric-label'>Historical Base Rate</div></div>""",
                unsafe_allow_html=True)
        with r4:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value {sev_class}'>{up['severity']}</div>
                <div class='metric-label'>Upset Severity</div></div>""",
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ALL FIRST-ROUND UPSET PROBABILITIES")
    upset_rows = []
    for region in REGIONS:
        reg_df = df_bracket[df_bracket["Region"] == region].sort_values("Seed")
        # Standard first round pairs: 1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9
        pairs = [(1,16),(2,15),(3,14),(4,13),(5,12),(6,11),(7,10),(8,9)]
        seeded = {row["Seed"]: row for _, row in reg_df.iterrows()}
        for (s1, s2) in pairs:
            if s1 in seeded and s2 in seeded:
                fav = seeded[s1]; dog = seeded[s2]
                up  = upset_probability(
                    fav["Seed"], dog["Seed"],
                    fav["KenPom"], dog["KenPom"],
                    fav["SOS"], dog["SOS"]
                )
                upset_rows.append({
                    "Region":    region,
                    "Matchup":   f"({s1}) vs ({s2})",
                    "Favorite":  fav["Team"],
                    "Underdog":  dog["Team"],
                    "UpsetProb": f"{up['upset_prob']*100:.1f}%",
                    "Severity":  up["severity"],
                    "KenPom Δ":  up["kenpom_edge"],
                })
    df_upsets = pd.DataFrame(upset_rows)
    st.dataframe(df_upsets, use_container_width=True)

# ─── Tab 3: Seed History ───────────────────────────────────────────────────────
with tab3:
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

    st.markdown("#### Round-by-Round Win Rates by Seed")
    st.dataframe(
        df_history[["Seed","R32_Rate","Sweet16_Rate","FF_Appearances","Championships"]]
            .style
            .background_gradient(subset=["R32_Rate","Sweet16_Rate"], cmap="YlOrRd")
            .format({"R32_Rate": "{:.1%}", "Sweet16_Rate": "{:.1%}"}),
        use_container_width=True,
    )

    st.markdown("#### Notable Historical Upsets")
    notable = pd.DataFrame([
        {"Year": 2018, "Matchup": "(1) Virginia vs (16) UMBC",    "Winner": "UMBC",        "Seed Diff": 15},
        {"Year": 2023, "Matchup": "(2) Arizona vs (15) Princeton", "Winner": "Princeton",   "Seed Diff": 13},
        {"Year": 2022, "Matchup": "(2) Kentucky vs (15) Saint Peter's", "Winner": "Saint Peter's", "Seed Diff": 13},
        {"Year": 2021, "Matchup": "(1) Illinois vs (15) Oral Roberts", "Winner": "Oral Roberts", "Seed Diff": 14},
        {"Year": 2016, "Matchup": "(2) Michigan State vs (15) Middle Tennessee", "Winner": "Middle Tennessee", "Seed Diff": 13},
    ])
    st.dataframe(notable, use_container_width=True)

# ─── Tab 4: Monte Carlo Simulation ────────────────────────────────────────────
with tab4:
    st.markdown("### MONTE CARLO CHAMPIONSHIP SIMULATION")

    if run_sim:
        with st.spinner(f"Running {n_simulations:,} tournament simulations..."):
            df_sim = simulate_tournament(df_bracket, n_sims=n_simulations)

        st.markdown(f"#### Championship Probability — Top 20 Teams ({n_simulations:,} sims)")

        top20 = df_sim.head(20)[["Seed","Region","Team","Conference",
                                  "KenPom","NetRtg","ChampionshipProb"]]

        st.dataframe(
            top20.style
                .background_gradient(subset=["ChampionshipProb"], cmap="YlOrRd")
                .background_gradient(subset=["KenPom"], cmap="Blues")
                .format({"ChampionshipProb": "{:.2%}", "NetRtg": "{:+.1f}"}),
            use_container_width=True,
        )

        st.markdown("#### Championship Probability by Seed (Aggregated)")
        seed_prob = (
            df_sim.groupby("Seed")["ChampionshipProb"]
            .sum().reset_index()
        )
        st.bar_chart(seed_prob.set_index("Seed"), use_container_width=True, height=280)

        # Most likely champion callout
        champ = df_sim.iloc[0]
        st.markdown(f"""
        <div style='background:#0F2847; border:2px solid #FFD166; border-radius:6px;
                    padding:1.5rem; text-align:center; margin-top:1rem;'>
            <div style='font-family:IBM Plex Mono,monospace; font-size:0.7rem;
                        color:#FF6B35; letter-spacing:3px;'>MOST LIKELY CHAMPION</div>
            <div style='font-family:Bebas Neue,sans-serif; font-size:3rem;
                        color:#FFD166; letter-spacing:4px;'>{champ['Team']}</div>
            <div style='font-family:IBM Plex Mono,monospace; font-size:0.85rem; color:#aaa;'>
                ({champ['Seed']}) Seed · {champ['Region']} Region · {champ['ChampionshipProb']*100:.1f}% championship probability
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Configure simulation parameters in the sidebar and press **▶ RUN SIMULATION**.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-family:IBM Plex Mono,monospace;
            font-size:0.65rem; color:#555; letter-spacing:2px; padding:1rem 0;'>
MARCH MADNESS ANALYZER ● DEMO BUILD ● DATA IS SYNTHETIC ●
UPSET MODEL: LOGISTIC REGRESSION + KENPOM DIFFERENTIAL ●
NOT FOR WAGERING PURPOSES
</div>
""", unsafe_allow_html=True)
