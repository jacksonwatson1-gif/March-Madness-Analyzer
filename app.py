"""
March Madness Analyzer — app.py  (v3)
======================================
Streamlit application for bracket analysis, team statistics,
historical seed performance, upset probability modeling,
EV-optimized bracket generation, and Monte Carlo simulation.

Architecture: Modular (config, data/, models/, ui/)
Model: Fitted logistic regression with 9 features (no collinearity)
"""

import streamlit as st
import pandas as pd
import math
import random
from datetime import datetime

# ── Local modules ──
from config import (
    REGIONS, ROUNDS, CONFERENCES, HISTORICAL_UPSET_RATES,
    BRACKET_MATCHUP_SEEDS, ROUND_POINTS, SEED_ROUND_WIN_RATES,
)
from fetch import build_full_dataset, generate_demo_bracket
from upset import (
    fit_model, upset_probability, get_model_info,
    compute_all_first_round, _matchup_from_rows,
)
from simulator import simulate_tournament
from optimizer import optimize_bracket
from styles import CSS
from components import (
    metric_card, style_df, team_slot_html, matchup_html,
    region_bracket_html, render_color_legend, team_card_html,
    win_prob_color,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="March Madness Analyzer v3",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CSS, unsafe_allow_html=True)


# ── Fit Model at Startup ──────────────────────────────────────────────────────
if "model_info" not in st.session_state:
    st.session_state.model_info = fit_model()


# ── Load Data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading bracket data..."):
    df_bracket, data_mode = build_full_dataset(year=2025)

# Ensure all required columns
for col, default in [
    ("Continuity", 70.0), ("Tempo", 68.0), ("AdjOE", 105.0),
    ("AdjDE", 100.0), ("AdjEM", 5.0), ("SOS", 0.55),
    ("TOV%", 16.0), ("3P%", 0.35), ("3P_Var", 0.04),
    ("FTRate", 0.30), ("RebMgn", 2.0),
]:
    if col not in df_bracket.columns:
        df_bracket[col] = default

# KenPom rank (ordinal rank of AdjEM)
df_bracket["KenPom"] = df_bracket["AdjEM"].rank(ascending=False).astype(int)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ CONTROLS")

    data_badge = "<span class='badge-live'>● LIVE</span>" if data_mode == "LIVE" \
                 else "<span class='badge-demo'>● DEMO</span>"
    model_info = st.session_state.model_info
    model_badge = "<span class='badge-fitted'>● FITTED</span>" if model_info.get("fitted") \
                  else "<span class='badge-default'>● DEFAULT</span>"

    st.markdown(f"**Data:** {data_badge} &nbsp; **Model:** {model_badge}", unsafe_allow_html=True)

    if model_info.get("brier_score"):
        st.caption(f"Brier Score: {model_info['brier_score']:.4f} · "
                   f"Accuracy: {model_info['accuracy']:.1%}")
    st.markdown("---")

    selected_region = st.selectbox("Region Filter", ["All"] + REGIONS)
    n_simulations = st.slider("Monte Carlo Sims", 500, 10000, 2000, 500)
    seed_range = st.slider("Seed Range", 1, 16, (1, 16))
    conf_filter = st.multiselect("Conference Filter", CONFERENCES[:10], default=CONFERENCES[:10])
    run_sim = st.button("▶ RUN SIMULATION", use_container_width=True)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.65rem;color:#666;font-family:IBM Plex Mono,monospace;'>
    <b>v3 Model Features:</b><br>
    · AdjOE / AdjDE (no collinearity)<br>
    · Tempo mismatch<br>
    · Roster continuity<br>
    · Strength of schedule<br>
    · Turnover rate<br>
    · 3-point variance<br>
    · Free throw rate<br>
    · Round adjustment<br>
    · Historical base rate anchor
    </div>""", unsafe_allow_html=True)


# ── Apply Filters ──────────────────────────────────────────────────────────────
df_filtered = df_bracket[
    (df_bracket["Seed"] >= seed_range[0]) &
    (df_bracket["Seed"] <= seed_range[1])
].copy()

if "Conference" in df_filtered.columns and conf_filter:
    df_filtered = df_filtered[df_filtered["Conference"].isin(conf_filter)]
if selected_region != "All":
    df_filtered = df_filtered[df_filtered["Region"] == selected_region]


# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_date = st.columns([5, 1])
with col_title:
    st.markdown("<div class='main-title'>MARCH MADNESS<br>ANALYZER</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>● NCAA Tournament Intelligence Platform · v3 ●</div>",
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
for col, (val, label) in zip([k1, k2, k3, k4, k5], [
    ("68", "Field Size"), ("4", "Regions"), ("63", "Games"),
    (str(len(df_filtered)), "Filtered"), ("2025", "Season"),
]):
    with col:
        st.markdown(metric_card(val, label), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_field, tab_bracket, tab_picker, tab_ev, tab_eff, tab_upset, tab_hist, tab_mc, tab_model = st.tabs([
    "📊 FIELD",
    "🗂 LIVE BRACKET",
    "🏆 PICKER",
    "💰 EV OPTIMIZER",
    "⚡ EFFICIENCY",
    "🎯 UPSET ANALYZER",
    "📈 SEED HISTORY",
    "🎲 MONTE CARLO",
    "🔬 MODEL",
])


# ─── Tab 1: Bracket Field ──────────────────────────────────────────────────────
with tab_field:
    st.markdown("### FIELD AT A GLANCE")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        show_cols = [c for c in
            ["Seed", "Region", "Team", "Conference", "Record", "Win%",
             "AdjOE", "AdjDE", "AdjEM", "Tempo", "Continuity", "TOV%", "SOS"]
            if c in df_filtered.columns]
        fmt = {}
        if "Win%" in show_cols: fmt["Win%"] = "{:.1%}"
        if "SOS" in show_cols: fmt["SOS"] = "{:.3f}"
        if "AdjEM" in show_cols: fmt["AdjEM"] = "{:+.1f}"
        if "Continuity" in show_cols: fmt["Continuity"] = "{:.1f}%"
        if "TOV%" in show_cols: fmt["TOV%"] = "{:.1f}"

        st.dataframe(
            style_df(df_filtered[show_cols], fmt=fmt),
            use_container_width=True, height=520,
        )

    with col_b:
        st.markdown("#### Avg AdjEM by Seed")
        seed_avg = (df_filtered.groupby("Seed")["AdjEM"]
                    .mean().reset_index().rename(columns={"AdjEM": "Avg AdjEM"}))
        st.bar_chart(seed_avg.set_index("Seed"), use_container_width=True, height=220)

        if "Continuity" in df_filtered.columns:
            st.markdown("#### Continuity % by Seed")
            cont_avg = (df_filtered.groupby("Seed")["Continuity"]
                        .mean().reset_index().rename(columns={"Continuity": "Avg %"}))
            st.bar_chart(cont_avg.set_index("Seed"), use_container_width=True, height=220)


# ─── Tab 2: Live Bracket ──────────────────────────────────────────────────────
with tab_bracket:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1f3c 0%,#1a3a6b 60%,#0d2a4a 100%);
                border:2px solid #FF6B35;border-radius:10px;padding:22px 30px;
                text-align:center;margin-bottom:18px;
                box-shadow:0 4px 24px #FF6B3544;">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:3rem;
                    color:#FF6B35;letter-spacing:8px;">MARCH MADNESS</div>
        <div style="font-family:'Bebas Neue',sans-serif;font-size:1.5rem;
                    color:#FFD166;letter-spacing:5px;">LIVE BRACKET TRACKER</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                    color:#aaa;letter-spacing:3px;margin-top:8px;">
            2025 NCAA Division I Men's Basketball Tournament</div>
    </div>""", unsafe_allow_html=True)

    # Build bracket state
    def build_bracket_state(df):
        bracket = {}
        for region in REGIONS:
            reg_df = df[df["Region"] == region].sort_values("Seed")
            seeded = {int(row["Seed"]): row for _, row in reg_df.iterrows()}
            matchups = []
            for s1, s2 in BRACKET_MATCHUP_SEEDS:
                if s1 in seeded and s2 in seeded:
                    t1, t2 = seeded[s1], seeded[s2]
                    up = _matchup_from_rows(t1, t2)
                    matchups.append({
                        "fav": t1, "dog": t2,
                        "fav_prob": up["fav_prob"],
                        "dog_prob": up["upset_prob"],
                        "winner": None,
                    })
            bracket[region] = matchups
        return bracket

    if "bracket_state" not in st.session_state:
        st.session_state.bracket_state = build_bracket_state(df_bracket)

    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("🔄 Reset Results", key="reset_bracket", use_container_width=True):
            st.session_state.bracket_state = build_bracket_state(df_bracket)
    with c2:
        render_color_legend()

    bracket_state = st.session_state.bracket_state

    reg_tabs = st.tabs([f"◈ {r.upper()}" for r in REGIONS])
    for reg_tab, region in zip(reg_tabs, REGIONS):
        with reg_tab:
            matchups = bracket_state.get(region, [])
            if not matchups:
                st.warning(f"No data for {region}.")
                continue

            left_col, right_col = st.columns([1, 1])
            with left_col:
                st.markdown(f"**{region.upper()} — Select Winners**")
                for i, m in enumerate(matchups):
                    fav_name = str(m["fav"].get("Team", "A"))
                    dog_name = str(m["dog"].get("Team", "B"))
                    fav_seed = int(m["fav"].get("Seed", 0))
                    dog_seed = int(m["dog"].get("Seed", 0))
                    current = m.get("winner")
                    options = ["— Not Played —", fav_name, dog_name]
                    def_idx = 0
                    if current == fav_name: def_idx = 1
                    elif current == dog_name: def_idx = 2

                    choice = st.selectbox(
                        f"G{i+1} ({fav_seed})v({dog_seed})", options, index=def_idx,
                        key=f"w_{region}_{i}", label_visibility="collapsed",
                    )
                    bracket_state[region][i]["winner"] = (
                        None if "Not Played" in choice else choice
                    )

            with right_col:
                st.markdown(f"**{region.upper()} — Probability Cards**")
                for m in matchups:
                    st.markdown(matchup_html(m), unsafe_allow_html=True)

    # Full bracket visual
    st.markdown("---")
    all_html = "".join(
        region_bracket_html(r, bracket_state.get(r, [])) for r in REGIONS
    )
    st.markdown(f"""
    <div style="display:flex;gap:14px;flex-wrap:wrap;
                background:linear-gradient(135deg,#071429,#0d1f3c);
                border:2px solid #FF6B35;border-radius:8px;
                padding:20px;overflow-x:auto;">
        {all_html}
    </div>""", unsafe_allow_html=True)


# ─── Tab 3: Bracket Picker ────────────────────────────────────────────────────
with tab_picker:
    st.markdown("### BRACKET PICKER — BUILD YOUR BRACKET")

    if "active_picker" not in st.session_state:
        st.session_state.active_picker = {}
    if "saved_brackets" not in st.session_state:
        st.session_state.saved_brackets = {}

    mc1, mc2, mc3 = st.columns([2, 1, 1])
    with mc1:
        bracket_name = st.text_input("Bracket Name", value="My Bracket", key="bp_name")
    with mc2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save", use_container_width=True):
            if bracket_name.strip():
                st.session_state.saved_brackets[bracket_name.strip()] = dict(st.session_state.active_picker)
                st.success(f"Saved: {bracket_name.strip()}")
    with mc3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 Clear Picks", use_container_width=True):
            st.session_state.active_picker = {}
            st.rerun()

    # Progress
    total_r1 = len(BRACKET_MATCHUP_SEEDS) * 4
    picked = len(st.session_state.active_picker)
    st.progress(picked / max(total_r1, 1), text=f"R64: {picked}/{total_r1} picked")

    # Region picker grid
    for region in REGIONS:
        reg_df = df_bracket[df_bracket["Region"] == region].sort_values("Seed")
        seeded = {int(row["Seed"]): row for _, row in reg_df.iterrows()}

        st.markdown(f"""
        <div style="background:linear-gradient(90deg,#FFD16622,transparent);
                    border-left:4px solid #FFD166;padding:8px 14px;
                    margin:12px 0;border-radius:0 4px 4px 0;">
            <span style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;
                         color:#FFD166;letter-spacing:4px;">{region.upper()}</span>
        </div>""", unsafe_allow_html=True)

        cols_per_row = 4
        pairs = BRACKET_MATCHUP_SEEDS
        for chunk_start in range(0, len(pairs), cols_per_row):
            chunk = pairs[chunk_start:chunk_start + cols_per_row]
            columns = st.columns(len(chunk))
            for col, (s1, s2) in zip(columns, chunk):
                if s1 not in seeded or s2 not in seeded:
                    continue
                t1, t2 = seeded[s1], seeded[s2]
                game_key = f"{region}_R1_{s1}v{s2}"
                current_pick = st.session_state.active_picker.get(game_key)

                with col:
                    up = _matchup_from_rows(t1, t2)
                    t1_name = str(t1.get("Team", ""))
                    t2_name = str(t2.get("Team", ""))
                    t1_picked = current_pick == t1_name
                    t2_picked = current_pick == t2_name

                    st.caption(f"({s1}) vs ({s2})")
                    if st.button(
                        f"{'✓ ' if t1_picked else ''}({s1}) {t1_name[:16]}",
                        key=f"pk_{region}_{s1}_{s2}_1",
                        use_container_width=True,
                        type="primary" if t1_picked else "secondary",
                    ):
                        st.session_state.active_picker[game_key] = t1_name
                        st.rerun()
                    if st.button(
                        f"{'✓ ' if t2_picked else ''}({s2}) {t2_name[:16]}",
                        key=f"pk_{region}_{s1}_{s2}_2",
                        use_container_width=True,
                        type="primary" if t2_picked else "secondary",
                    ):
                        st.session_state.active_picker[game_key] = t2_name
                        st.rerun()


# ─── Tab 4: EV Optimizer (NEW) ────────────────────────────────────────────────
with tab_ev:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1f3c,#1a3a6b);
                border:2px solid #FFD166;border-radius:10px;padding:22px 30px;
                text-align:center;margin-bottom:20px;">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:2.8rem;
                    color:#FFD166;letter-spacing:6px;">EV BRACKET OPTIMIZER</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                    color:#aaa;letter-spacing:3px;margin-top:4px;">
            MAXIMIZE EXPECTED POINTS · SCORING-SYSTEM AWARE</div>
    </div>""", unsafe_allow_html=True)

    ev_c1, ev_c2, ev_c3, ev_c4 = st.columns(4)
    with ev_c1:
        scoring_sys = st.selectbox("Scoring System", list(ROUND_POINTS.keys()),
                                   format_func=lambda x: x.upper(), key="ev_scoring")
    with ev_c2:
        strategy = st.selectbox("Strategy",
                                ["max_ev", "chalk", "contrarian"],
                                format_func=lambda x: x.replace("_", " ").title(),
                                key="ev_strategy")
    with ev_c3:
        pool_size = st.number_input("Pool Size", min_value=2, max_value=1000,
                                     value=5, key="ev_pool")
    with ev_c4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_ev = st.button("⚡ OPTIMIZE", use_container_width=True, type="primary")

    if run_ev:
        with st.spinner("Computing EV-optimal bracket..."):
            result = optimize_bracket(
                df_bracket, scoring=scoring_sys,
                strategy=strategy, pool_size=pool_size,
            )

        # Summary
        total_ev = result["total_ev"]
        picks_df = pd.DataFrame(result["picks"])

        ev1, ev2, ev3, ev4 = st.columns(4)
        for col, (val, label) in zip([ev1, ev2, ev3, ev4], [
            (f"{total_ev:.1f}", "Total Expected Points"),
            (f"{len(picks_df)}", "Games Picked"),
            (strategy.replace('_', ' ').title(), "Strategy"),
            (scoring_sys.upper(), "Scoring"),
        ]):
            with col:
                st.markdown(metric_card(val, label, "1.8rem"), unsafe_allow_html=True)

        # Champion
        champ_pick = picks_df[picks_df["Round"] == "Champ"]
        if not champ_pick.empty:
            champ = champ_pick.iloc[0]
            st.markdown(f"""
            <div style='background:#0F2847;border:2px solid #FFD166;border-radius:6px;
                        padding:1.5rem;text-align:center;margin:1rem 0;'>
                <div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;
                            color:#FF6B35;letter-spacing:3px;'>EV-OPTIMAL CHAMPION</div>
                <div style='font-family:Bebas Neue,sans-serif;font-size:3rem;
                            color:#FFD166;letter-spacing:4px;'>{champ['Pick']}</div>
                <div style='font-family:IBM Plex Mono,monospace;font-size:0.85rem;color:#aaa;'>
                    ({champ['Seed']}) Seed · Win Prob {champ['WinProb']:.1%} · EV {champ['EV']:.1f} pts
                </div>
            </div>""", unsafe_allow_html=True)

        # Round-by-round picks
        for rnd in ["R64", "R32", "S16", "E8", "F4", "Champ"]:
            rnd_df = picks_df[picks_df["Round"] == rnd]
            if rnd_df.empty:
                continue
            round_labels = {"R64": "ROUND OF 64", "R32": "ROUND OF 32", "S16": "SWEET 16",
                           "E8": "ELITE EIGHT", "F4": "FINAL FOUR", "Champ": "CHAMPIONSHIP"}
            st.markdown(f"#### {round_labels.get(rnd, rnd)}")
            st.dataframe(
                style_df(rnd_df, fmt={"WinProb": "{:.1%}", "EV": "{:.2f}"}),
                use_container_width=True, hide_index=True,
            )


# ─── Tab 5: Efficiency Dashboard ──────────────────────────────────────────────
with tab_eff:
    st.markdown("### OFFENSIVE & DEFENSIVE EFFICIENCY")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### Top 20 — AdjOE (Offense)")
        top_off = (df_bracket.nlargest(20, "AdjOE")
                   [["Seed", "Team", "Region", "AdjOE", "Tempo"]]
                   .reset_index(drop=True))
        st.dataframe(
            style_df(top_off, fmt={"AdjOE": "{:.1f}", "Tempo": "{:.1f}"}),
            use_container_width=True, height=420,
        )
    with col_e2:
        st.markdown("#### Top 20 — AdjDE (Defense, lower = better)")
        top_def = (df_bracket.nsmallest(20, "AdjDE")
                   [["Seed", "Team", "Region", "AdjDE", "Tempo"]]
                   .reset_index(drop=True))
        st.dataframe(
            style_df(top_def, fmt={"AdjDE": "{:.1f}", "Tempo": "{:.1f}"}),
            use_container_width=True, height=420,
        )

    st.markdown("---")
    st.markdown("#### Net Efficiency (AdjEM) — Top 30")
    chart_data = (df_bracket.sort_values("AdjEM", ascending=False)
                  .head(30).set_index("Team")["AdjEM"])
    st.bar_chart(chart_data, use_container_width=True, height=300)

    # Quadrant analysis
    st.markdown("---")
    st.markdown("#### Efficiency Quadrant Analysis")
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
        quad_counts.columns = ["Quadrant", "Count"]
        st.dataframe(quad_counts, use_container_width=True, hide_index=True)
    with q2:
        elite = (df_quad[df_quad["Quadrant"] == "Elite (High O + Low D)"]
                 .sort_values("AdjEM", ascending=False)
                 [["Seed", "Team", "Region", "AdjOE", "AdjDE", "AdjEM", "Continuity"]]
                 .head(16).reset_index(drop=True))
        st.dataframe(
            style_df(elite, fmt={"AdjOE": "{:.1f}", "AdjDE": "{:.1f}",
                                  "AdjEM": "{:+.1f}", "Continuity": "{:.1f}%"}),
            use_container_width=True,
        )


# ─── Tab 6: Upset Analyzer ────────────────────────────────────────────────────
with tab_upset:
    st.markdown("### UPSET PROBABILITY MATRIX")
    st.markdown("9-feature logistic model: AdjOE, AdjDE, Tempo, SOS, Continuity, TOV%, 3P Var, FT Rate, Round.")

    teams_list = df_bracket["Team"].tolist()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🏆 FAVORITE")
        fav_team = st.selectbox("Favorite", teams_list, key="fav_t")
        fav_data = df_bracket[df_bracket["Team"] == fav_team].iloc[0]
        st.markdown(team_card_html(fav_data, "FAV"), unsafe_allow_html=True)
    with c2:
        st.markdown("##### 🐶 UNDERDOG")
        dog_team = st.selectbox("Underdog",
                                [t for t in teams_list if t != fav_team], key="dog_t")
        dog_data = df_bracket[df_bracket["Team"] == dog_team].iloc[0]
        st.markdown(team_card_html(dog_data, "DOG"), unsafe_allow_html=True)

    if st.button("COMPUTE UPSET PROBABILITY", type="primary"):
        up = _matchup_from_rows(fav_data, dog_data)
        sev_class = f"upset-{up['severity'].lower()}"

        r1, r2, r3, r4, r5 = st.columns(5)
        for col, (val, lbl) in zip([r1, r2, r3, r4, r5], [
            (f"{up['upset_prob']*100:.1f}%", "Upset Prob"),
            (f"{up['fav_prob']*100:.1f}%", "Favorite Win"),
            (f"{up['base_rate']*100:.1f}%", "Base Rate"),
            (f"{up['tempo_mismatch']:.1f}", "Tempo Δ"),
            (f"<span class='{sev_class}'>{up['severity']}</span>", "Severity"),
        ]):
            with col:
                st.markdown(metric_card(val, lbl, "1.6rem"), unsafe_allow_html=True)

        # Factor breakdown
        st.markdown("##### Factor Breakdown")
        factors = pd.DataFrame([
            {"Factor": "Historical Base Rate", "Value": f"{up['base_rate']*100:.1f}%", "Favors": "—"},
            {"Factor": "AdjOE Differential", "Value": f"{up['adj_oe_edge']:+.1f}", "Favors": "Fav" if up['adj_oe_edge'] > 0 else "Dog"},
            {"Factor": "AdjDE Differential", "Value": f"{up['adj_de_edge']:+.1f}", "Favors": "Fav" if up['adj_de_edge'] > 0 else "Dog"},
            {"Factor": "Tempo Mismatch", "Value": f"{up['tempo_mismatch']:.1f}", "Favors": "Dog" if up['tempo_mismatch'] > 3 else "Neutral"},
            {"Factor": "SOS Edge", "Value": f"{up['sos_edge']:+.3f}", "Favors": "Fav" if up['sos_edge'] > 0 else "Dog"},
            {"Factor": "Continuity Δ (Dog−Fav)", "Value": f"{up['continuity_edge']:+.1f}%", "Favors": "Dog" if up['continuity_edge'] > 0 else "Fav"},
            {"Factor": "TOV Rate Δ", "Value": f"{up['tov_edge']:+.1f}", "Favors": "Fav" if up['tov_edge'] > 0 else "Dog"},
            {"Factor": "3P% Variance (Dog)", "Value": f"{up['three_pt_var']:.4f}", "Favors": "Dog" if up['three_pt_var'] > 0.05 else "Neutral"},
            {"Factor": "FT Rate Δ", "Value": f"{up['ft_rate_edge']:+.3f}", "Favors": "Fav" if up['ft_rate_edge'] > 0 else "Dog"},
        ])
        st.dataframe(factors, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### ALL FIRST-ROUND UPSET PROBABILITIES")
    all_upsets = compute_all_first_round(df_bracket)
    if all_upsets:
        upset_df = pd.DataFrame(all_upsets)
        show_cols = ["Region", "Matchup", "Favorite", "Underdog",
                     "upset_prob", "severity", "adj_oe_edge", "adj_de_edge",
                     "tempo_mismatch", "continuity_edge"]
        display_cols = [c for c in show_cols if c in upset_df.columns]
        upset_display = upset_df[display_cols].copy()
        upset_display = upset_display.rename(columns={
            "upset_prob": "Upset%", "severity": "Risk",
            "adj_oe_edge": "AdjOE Δ", "adj_de_edge": "AdjDE Δ",
            "tempo_mismatch": "Tempo Δ", "continuity_edge": "Cont. Δ",
        })
        upset_display["Upset%"] = upset_display["Upset%"].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(upset_display, use_container_width=True)


# ─── Tab 7: Seed History ───────────────────────────────────────────────────────
with tab_hist:
    st.markdown("### HISTORICAL SEED PERFORMANCE (1985 – 2024)")

    seed_data = {
        "Seed": list(range(1, 17)),
        "FF_Apps": [140, 48, 32, 22, 18, 14, 12, 12, 3, 5, 8, 9, 1, 1, 0, 1],
        "Titles":  [37, 18, 7, 5, 3, 2, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        "R32_Rate": [0.993, 0.937, 0.848, 0.798, 0.641, 0.630, 0.608, 0.511,
                     0.489, 0.392, 0.370, 0.359, 0.202, 0.152, 0.063, 0.013],
        "S16_Rate": [0.874, 0.680, 0.540, 0.436, 0.318, 0.296, 0.280, 0.240,
                     0.165, 0.155, 0.163, 0.152, 0.074, 0.063, 0.027, 0.006],
    }
    df_history = pd.DataFrame(seed_data)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Final Four Appearances")
        st.bar_chart(df_history.set_index("Seed")["FF_Apps"], use_container_width=True, height=300)
    with c2:
        st.markdown("#### Championships")
        st.bar_chart(df_history.set_index("Seed")["Titles"], use_container_width=True, height=300)

    st.markdown("#### Round-by-Round Win Rates")
    st.dataframe(
        style_df(df_history, fmt={"R32_Rate": "{:.1%}", "S16_Rate": "{:.1%}"}),
        use_container_width=True,
    )

    st.markdown("#### Notable Historical Upsets")
    notable = pd.DataFrame([
        {"Year": 2023, "Matchup": "(1) Purdue vs (16) FDU",              "Winner": "FDU",              "Gap": 15},
        {"Year": 2018, "Matchup": "(1) Virginia vs (16) UMBC",           "Winner": "UMBC",             "Gap": 15},
        {"Year": 2023, "Matchup": "(2) Arizona vs (15) Princeton",       "Winner": "Princeton",        "Gap": 13},
        {"Year": 2022, "Matchup": "(2) Kentucky vs (15) Saint Peter's",  "Winner": "Saint Peter's",    "Gap": 13},
        {"Year": 2021, "Matchup": "(2) Ohio St vs (15) Oral Roberts",    "Winner": "Oral Roberts",     "Gap": 13},
        {"Year": 2016, "Matchup": "(2) Mich. St vs (15) MTSU",           "Winner": "Middle Tennessee", "Gap": 13},
        {"Year": 2013, "Matchup": "(2) Georgetown vs (15) FGCU",         "Winner": "FGCU",             "Gap": 13},
    ])
    st.dataframe(notable, use_container_width=True)


# ─── Tab 8: Monte Carlo ───────────────────────────────────────────────────────
with tab_mc:
    st.markdown("### MONTE CARLO CHAMPIONSHIP SIMULATION")
    st.markdown("9-feature logistic model resolved per game · round-specific adjustments · 95% confidence intervals.")

    if run_sim:
        with st.spinner(f"Running {n_simulations:,} simulations..."):
            sim_result = simulate_tournament(df_bracket, n_sims=n_simulations)
            df_sim = sim_result["results"]

        # Top 20
        st.markdown(f"#### Championship Probability — Top 20 ({n_simulations:,} sims)")

        sim_cols = [c for c in
            ["Seed", "Region", "Team", "Conference",
             "AdjOE", "AdjDE", "AdjEM", "Continuity",
             "R64_Prob", "R32_Prob", "S16_Prob", "E8_Prob", "F4_Prob",
             "ChampionshipProb", "CI_Lower", "CI_Upper"]
            if c in df_sim.columns]

        fmt_sim = {
            "ChampionshipProb": "{:.2%}", "CI_Lower": "{:.2%}", "CI_Upper": "{:.2%}",
            "R64_Prob": "{:.1%}", "R32_Prob": "{:.1%}", "S16_Prob": "{:.1%}",
            "E8_Prob": "{:.1%}", "F4_Prob": "{:.1%}",
            "AdjEM": "{:+.1f}", "Continuity": "{:.1f}%",
            "AdjOE": "{:.1f}", "AdjDE": "{:.1f}",
        }

        st.dataframe(
            style_df(df_sim.head(20)[sim_cols], fmt=fmt_sim),
            use_container_width=True,
        )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Champ Prob by Seed")
            seed_prob = df_sim.groupby("Seed")["ChampionshipProb"].sum().reset_index()
            st.bar_chart(seed_prob.set_index("Seed"), use_container_width=True, height=260)
        with col_s2:
            st.markdown("#### Champ Prob by Conference")
            if "Conference" in df_sim.columns:
                conf_prob = (df_sim.groupby("Conference")["ChampionshipProb"]
                             .sum().sort_values(ascending=False).head(10).reset_index())
                st.bar_chart(conf_prob.set_index("Conference"), use_container_width=True, height=260)

        # Champion card
        champ = df_sim.iloc[0]
        ci = sim_result["confidence"].get(champ["Team"], (0, 0))
        st.markdown(f"""
        <div style='background:#0F2847;border:2px solid #FFD166;border-radius:6px;
                    padding:1.5rem;text-align:center;margin-top:1rem;'>
            <div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;
                        color:#FF6B35;letter-spacing:3px;'>MOST LIKELY CHAMPION</div>
            <div style='font-family:Bebas Neue,sans-serif;font-size:3rem;
                        color:#FFD166;letter-spacing:4px;'>{champ['Team']}</div>
            <div style='font-family:IBM Plex Mono,monospace;font-size:0.85rem;color:#aaa;'>
                ({champ['Seed']}) Seed · {champ.get('Region','?')} ·
                {champ['ChampionshipProb']*100:.1f}% [95% CI: {ci[0]*100:.1f}%–{ci[1]*100:.1f}%]
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Configure simulations in the sidebar and press **▶ RUN SIMULATION**.")


# ─── Tab 9: Model Dashboard (NEW) ─────────────────────────────────────────────
with tab_model:
    st.markdown("### MODEL TRANSPARENCY DASHBOARD")

    info = st.session_state.model_info

    # Status
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, (val, label) in zip([mc1, mc2, mc3, mc4], [
        ("FITTED" if info.get("fitted") else "DEFAULT", "Model Status"),
        (f"{info.get('brier_score', 'N/A')}", "Brier Score"),
        (f"{info.get('n_samples', 0):,}", "Training Samples"),
        (f"{info.get('accuracy', 0):.1%}" if info.get('accuracy') else "N/A", "Accuracy"),
    ]):
        with col:
            st.markdown(metric_card(str(val), label, "1.6rem"), unsafe_allow_html=True)

    st.markdown("---")

    # Coefficient table
    st.markdown("#### Fitted Coefficients")
    coefs = info.get("coefficients", {})
    coef_df = pd.DataFrame([
        {"Feature": k.replace("_", " ").title(), "Coefficient": round(v, 5), "Direction": "↑ Upset" if v > 0 else "↓ Upset"}
        for k, v in coefs.items()
    ])
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Feature importance (absolute coefficient magnitude)
    st.markdown("#### Feature Importance (|Coefficient|)")
    importance = {k: abs(v) for k, v in coefs.items() if k != "intercept"}
    imp_df = pd.DataFrame([
        {"Feature": k.replace("_", " ").title(), "Importance": round(v, 4)}
        for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)
    ])
    st.bar_chart(imp_df.set_index("Feature"), use_container_width=True, height=300)

    st.markdown("---")
    st.markdown("""
    **Model Architecture:**
    - Logistic regression with L2 regularization (C=1.0)
    - 9 features: AdjOE diff, AdjDE diff, tempo mismatch, SOS diff,
      continuity diff, TOV rate diff, 3P% variance, FT rate diff, round number
    - Historical base rate anchor (40% weight on seed-pair logit)
    - No AdjOE/AdjDE/AdjEM collinearity — uses AdjOE and AdjDE independently
    - Probability clamp: [0.5%, 95%] (expanded from prior 1%–75%)
    - Brier score: lower is better (0.0 = perfect, 0.25 = random)
    """)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center;font-family:IBM Plex Mono,monospace;
            font-size:0.6rem;color:#444;letter-spacing:2px;padding:1rem 0;'>
MARCH MADNESS ANALYZER v3 · {data_mode} DATA ·
9-FEATURE LOGISTIC MODEL {'(FITTED)' if info.get('fitted') else '(DEFAULT)'} ·
BARTTORVIK + ESPN · NOT FOR WAGERING PURPOSES
</div>""", unsafe_allow_html=True)
