"""
components.py — March Madness Analyzer v3
==========================================
Reusable HTML/Streamlit component helpers.
"""


def metric_card(value, label, font_size="2.2rem"):
    """Render a styled metric card."""
    return f"""
    <div class='metric-card'>
        <div class='value' style='font-size:{font_size};'>{value}</div>
        <div class='label'>{label}</div>
    </div>"""


def style_df(df, fmt=None):
    """Apply formatting to a DataFrame for display."""
    if fmt:
        styled = df.style.format(fmt, na_rep="—")
    else:
        styled = df.style
    return styled


def win_prob_color(prob):
    """Return hex color based on win probability."""
    if prob >= 0.75:
        return "#06D6A0"
    elif prob >= 0.55:
        return "#FFD166"
    elif prob >= 0.40:
        return "#FF6B35"
    else:
        return "#EF476F"


def team_slot_html(row, prob=None, is_fav=True):
    """Render a team slot in bracket view."""
    seed = int(row.get("Seed", 0))
    team = str(row.get("Team", ""))
    css_class = "fav" if is_fav else "dog"
    prob_html = ""
    if prob is not None:
        color = win_prob_color(prob)
        prob_html = f"<span style='color:{color};font-weight:600;'>{prob*100:.0f}%</span>"

    return f"""
    <div class='team-slot {css_class}'>
        <span>({seed}) {team}</span>
        {prob_html}
    </div>"""


def matchup_html(matchup):
    """Render a matchup probability card."""
    fav = matchup.get("fav", {})
    dog = matchup.get("dog", {})
    fav_prob = matchup.get("fav_prob", 0.5)
    dog_prob = matchup.get("dog_prob", 0.5)
    winner = matchup.get("winner")

    fav_name = str(fav.get("Team", "Team A"))
    dog_name = str(dog.get("Team", "Team B"))
    fav_seed = int(fav.get("Seed", 1))
    dog_seed = int(dog.get("Seed", 16))

    fav_color = win_prob_color(fav_prob)
    dog_color = win_prob_color(dog_prob)

    winner_marker_fav = " ✓" if winner == fav_name else ""
    winner_marker_dog = " ✓" if winner == dog_name else ""

    fav_bg = "#06D6A015" if winner == fav_name else "transparent"
    dog_bg = "#FF6B3515" if winner == dog_name else "transparent"

    return f"""
    <div class='matchup-card'>
        <div style='display:flex;justify-content:space-between;align-items:center;
                    padding:3px 0;background:{fav_bg};border-radius:3px;'>
            <span style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#ddd;'>
                ({fav_seed}) {fav_name}{winner_marker_fav}
            </span>
            <span style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;
                         color:{fav_color};font-weight:600;'>
                {fav_prob*100:.1f}%
            </span>
        </div>
        <div style='display:flex;justify-content:space-between;align-items:center;
                    padding:3px 0;background:{dog_bg};border-radius:3px;'>
            <span style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#ddd;'>
                ({dog_seed}) {dog_name}{winner_marker_dog}
            </span>
            <span style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;
                         color:{dog_color};font-weight:600;'>
                {dog_prob*100:.1f}%
            </span>
        </div>
    </div>"""


def region_bracket_html(region, matchups):
    """Render a mini region bracket visualization."""
    games_html = ""
    for m in matchups:
        fav = m.get("fav", {})
        dog = m.get("dog", {})
        fav_seed = int(fav.get("Seed", 0))
        dog_seed = int(dog.get("Seed", 0))
        fav_name = str(fav.get("Team", ""))[:14]
        dog_name = str(dog.get("Team", ""))[:14]
        fav_prob = m.get("fav_prob", 0.5)
        dog_prob = m.get("dog_prob", 0.5)
        winner = m.get("winner")

        fav_w = "font-weight:700;color:#06D6A0;" if winner == str(fav.get("Team", "")) else ""
        dog_w = "font-weight:700;color:#FF6B35;" if winner == str(dog.get("Team", "")) else ""

        games_html += f"""
        <div style='margin:3px 0;padding:4px 6px;background:#07142955;border-radius:3px;
                    font-family:IBM Plex Mono,monospace;font-size:0.58rem;'>
            <div style='{fav_w}'>({fav_seed}) {fav_name} <span style='float:right;'>{fav_prob*100:.0f}%</span></div>
            <div style='{dog_w}'>({dog_seed}) {dog_name} <span style='float:right;'>{dog_prob*100:.0f}%</span></div>
        </div>"""

    return f"""
    <div class='region-bracket'>
        <div class='region-title'>{region.upper()}</div>
        {games_html}
    </div>"""


def render_color_legend():
    """Render the win probability color legend."""
    import streamlit as st
    st.markdown("""
    <div class='color-legend'>
        <span><span class='dot' style='background:#06D6A0;'></span> ≥75% (Strong)</span>
        <span><span class='dot' style='background:#FFD166;'></span> 55–75% (Lean)</span>
        <span><span class='dot' style='background:#FF6B35;'></span> 40–55% (Toss-up)</span>
        <span><span class='dot' style='background:#EF476F;'></span> <40% (Upset Risk)</span>
    </div>""", unsafe_allow_html=True)


def team_card_html(row, role="FAV"):
    """Render a detailed team card for upset analysis."""
    team = str(row.get("Team", ""))
    seed = int(row.get("Seed", 0))
    region = str(row.get("Region", ""))
    conf = str(row.get("Conference", ""))
    record = str(row.get("Record", ""))
    adj_oe = float(row.get("AdjOE", 100))
    adj_de = float(row.get("AdjDE", 100))
    adj_em = float(row.get("AdjEM", 0))
    tempo = float(row.get("Tempo", 68))
    cont = float(row.get("Continuity", 70))
    sos = float(row.get("SOS", 0.50))
    tov = float(row.get("TOV%", 16))

    border_color = "#06D6A0" if role == "FAV" else "#FF6B35"
    role_label = "FAVORITE" if role == "FAV" else "UNDERDOG"

    return f"""
    <div class='team-card' style='border-color:{border_color};'>
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;
                    color:{border_color};letter-spacing:2px;'>{role_label}</div>
        <div class='team-name'>{team}</div>
        <div class='team-meta'>
            ({seed}) Seed · {region} · {conf} · {record}
        </div>
        <div class='stat-row'>
            <div class='stat-item'>AdjOE <span class='stat-val'>{adj_oe:.1f}</span></div>
            <div class='stat-item'>AdjDE <span class='stat-val'>{adj_de:.1f}</span></div>
            <div class='stat-item'>AdjEM <span class='stat-val'>{adj_em:+.1f}</span></div>
            <div class='stat-item'>Tempo <span class='stat-val'>{tempo:.1f}</span></div>
            <div class='stat-item'>Cont <span class='stat-val'>{cont:.0f}%</span></div>
            <div class='stat-item'>SOS <span class='stat-val'>{sos:.3f}</span></div>
            <div class='stat-item'>TOV% <span class='stat-val'>{tov:.1f}</span></div>
        </div>
    </div>"""
