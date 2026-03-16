"""
styles.py — March Madness Analyzer v3
=======================================
CSS styles for Streamlit interface.
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');

.main-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    color: #FF6B35;
    letter-spacing: 12px;
    line-height: 1.0;
    text-transform: uppercase;
}
.subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #FFD166;
    letter-spacing: 5px;
    margin-top: 4px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0F2847, #1a3a6b);
    border: 1px solid #FF6B3555;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-card .value {
    font-family: 'Bebas Neue', sans-serif;
    color: #FFD166;
    letter-spacing: 3px;
}
.metric-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #aaa;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Badges */
.badge-live {
    background: #06D6A0; color: #000; padding: 2px 8px;
    border-radius: 4px; font-size: 0.65rem; font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-demo {
    background: #FF6B35; color: #fff; padding: 2px 8px;
    border-radius: 4px; font-size: 0.65rem; font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-fitted {
    background: #06D6A0; color: #000; padding: 2px 8px;
    border-radius: 4px; font-size: 0.65rem; font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-default {
    background: #FFD166; color: #000; padding: 2px 8px;
    border-radius: 4px; font-size: 0.65rem; font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}

/* Upset severity */
.upset-high   { color: #EF476F; font-weight: 700; }
.upset-medium { color: #FFD166; font-weight: 600; }
.upset-low    { color: #06D6A0; font-weight: 500; }
.upset-minimal{ color: #aaa;    font-weight: 400; }

/* Team slots */
.team-slot {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 4px 8px;
    border-radius: 4px;
    margin: 2px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.team-slot.fav { background: #0F284788; border-left: 3px solid #06D6A0; }
.team-slot.dog { background: #0F284788; border-left: 3px solid #FF6B35; }

/* Matchup card */
.matchup-card {
    background: linear-gradient(135deg, #0d1f3c, #132d54);
    border: 1px solid #FF6B3533;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
}

/* Region bracket */
.region-bracket {
    flex: 1;
    min-width: 200px;
    background: #0d1f3c88;
    border: 1px solid #FF6B3522;
    border-radius: 6px;
    padding: 10px;
}
.region-bracket .region-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    color: #FFD166;
    letter-spacing: 4px;
    text-align: center;
    margin-bottom: 8px;
    border-bottom: 1px solid #FF6B3533;
    padding-bottom: 4px;
}

/* Team card */
.team-card {
    background: linear-gradient(135deg, #0d1f3c, #132d54);
    border: 1px solid #FF6B3544;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
}
.team-card .team-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    color: #FFD166;
    letter-spacing: 3px;
}
.team-card .team-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #aaa;
}
.team-card .stat-row {
    display: flex; gap: 12px; margin-top: 8px; flex-wrap: wrap;
}
.team-card .stat-item {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #ddd;
}
.team-card .stat-item .stat-val {
    color: #FFD166;
    font-weight: 600;
}

/* Color legend */
.color-legend {
    display: flex; gap: 16px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem; color: #aaa; flex-wrap: wrap;
}
.color-legend span {
    display: inline-flex; align-items: center; gap: 4px;
}
.color-legend .dot {
    width: 8px; height: 8px; border-radius: 50%; display: inline-block;
}

/* Global dark tweaks */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #071429 0%, #0d1f3c 100%) !important;
}
[data-testid="stSidebar"] {
    background: #0a1628 !important;
}
</style>
"""
