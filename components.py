"""
ui/components.py — Reusable HTML card builders and bracket visualization helpers.
"""

import streamlit as st
import pandas as pd


def metric_card(value: str, label: str, size: str = "2.4rem") -> str:
    return f"""
    <div class='metric-card'>
        <div class='metric-value' style='font-size:{size};'>{value}</div>
        <div class='metric-label'>{label}</div>
    </div>"""


def win_prob_color(prob: float) -> tuple:
    if prob >= 0.75:
        return ("#0a5c2e", "#06D6A0", "@@@@")
    elif prob >= 0.55:
        return ("#1a472a", "#4ade80", "@@@o")
    elif prob >= 0.45:
        return ("#3d2e00", "#FFD166", "@@oo")
    elif prob >= 0.25:
        return ("#5c1a1a", "#f87171", "@@oo")
    else:
        return ("#6b0f0f", "#EF476F", "@ooo")


def team_slot_html(
    team_row, prob: float,
    is_winner: bool = False,
    is_eliminated: bool = False,
) -> str:
    bg, fg, dots = win_prob_color(prob)
    name = str(team_row.get("Team", "TBD"))
    seed = int(team_row.get("Seed", 0))
    pct_str = f"{prob*100:.0f}%"

    if is_winner:
        bg, fg = "#0a3d1f", "#06D6A0"
    if is_eliminated:
        bg, fg = "#1a0a0a", "#555"

    adv_icon = "ADV" if is_winner else ("ELIM" if is_eliminated else "")
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
    fav = matchup["fav"]
    dog = matchup["dog"]
    fp = matchup["fav_prob"]
    dp = matchup["dog_prob"]
    winner = matchup.get("winner")

    fav_won = winner == str(fav.get("Team", ""))
    dog_won = winner == str(dog.get("Team", ""))

    slot1 = team_slot_html(fav, fp, is_winner=fav_won, is_eliminated=dog_won)
    slot2 = team_slot_html(dog, dp, is_winner=dog_won, is_eliminated=fav_won)

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
    entries = [
        ("G", "#06D6A0", ">= 75%",   "Heavy Fav"),
        ("g", "#4ade80", "55-74%",  "Mod. Fav"),
        ("Y", "#FFD166", "45-54%",  "Toss-Up"),
        ("r", "#f87171", "25-44%",  "Mod. Dog"),
        ("R", "#EF476F", "< 25%",   "Heavy Dog"),
    ]
    st.markdown("**WIN PROBABILITY COLOR KEY**")
    cols = st.columns(5)
    for col, (icon, color, pct, label) in zip(cols, entries):
        with col:
            st.markdown(
                f"<div style='background:{color}22;border-left:4px solid {color};"
                f"border-radius:3px;padding:8px 10px;text-align:center;'>"
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;"
                f"color:{color};font-weight:700;'>{pct}</div>"
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.62rem;"
                f"color:#aaa;margin-top:2px;'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def style_df(df: pd.DataFrame, fmt: dict = None):
    display_df = df.copy()

    for col in ["Win Prob", "Win%", "ChampionshipProb", "Champ_Prob",
                "R64_Prob", "R32_Prob", "S16_Prob", "E8_Prob", "F4_Prob"]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(
                display_df[col].astype(str).str.replace('%', ''), errors='coerce'
            )

    styled = display_df.style

    for col in ["Win Prob", "Win%", "ChampionshipProb", "Champ_Prob", "WinProb"]:
        if col in display_df.columns:
            styled = styled.background_gradient(subset=[col], cmap='RdYlGn', vmin=0, vmax=1)

    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#FFD166'),
            ('color', '#0B1F3A'),
            ('font-weight', 'bold'),
        ]}
    ])

    if fmt:
        styled = styled.format(fmt, na_rep="--")

    return styled


def team_card_html(data, label: str) -> str:
    adjoe = float(data.get("AdjOE", 0))
    adjde = float(data.get("AdjDE", 0))
    adjem = float(data.get("AdjEM", 0))
    cont = float(data.get("Continuity", 70))
    tempo = float(data.get("Tempo", 68))
    tov = float(data.get("TOV%", 16))

    return f"""
    <div class='metric-card'>
    <div class='metric-label'>{label} | Seed: {data['Seed']} | Region: {data.get('Region','?')}</div>
    <div style='font-size:0.85rem;color:#aaa;margin-bottom:0.8rem;'>
        {data.get('Conference','?')}
    </div>
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;'>
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
    </div>
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;margin-top:0.5rem;'>
        <div>
            <div class='metric-value' style='font-size:1.1rem;color:#aaa;'>{cont:.0f}%</div>
            <div class='metric-label'>Continuity</div>
        </div>
        <div>
            <div class='metric-value' style='font-size:1.1rem;color:#60a5fa;'>{tempo:.1f}</div>
            <div class='metric-label'>Tempo</div>
        </div>
        <div>
            <div class='metric-value' style='font-size:1.1rem;color:#c084fc;'>{tov:.1f}</div>
            <div class='metric-label'>TOV%</div>
        </div>
    </div>
    </div>"""
