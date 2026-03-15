"""
ui/styles.py — All CSS for the March Madness Analyzer.
"""

CSS = """
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
    --purple: #c084fc;
    --blue:   #60a5fa;
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

.badge-fitted { background:#06D6A0; color:#000; padding:2px 8px;
                border-radius:3px; font-size:0.65rem;
                font-family:'IBM Plex Mono',monospace; letter-spacing:2px; }
.badge-default { background:#FFD166; color:#000; padding:2px 8px;
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

button[data-baseweb="tab"] div p {
    color: #FFD166 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
}

button[data-baseweb="tab"]:hover div p {
    color: #FFFFFF !important;
    text-shadow: 0 0 10px rgba(255, 209, 102, 0.8) !important;
}

button[aria-selected="true"] div p {
    color: #FF6B35 !important;
    border-bottom: 2px solid #FF6B35 !important;
}

button[data-baseweb="tab"]:hover {
    background-color: rgba(255, 209, 102, 0.1) !important;
}

.stDataFrame th, .stTable th, div[data-testid="stTable"] th {
    background-color: #FFD166 !important;
    border: 1px solid #FF6B35 !important;
}

.stDataFrame th div, .stTable th div, div[data-testid="stTable"] th div {
    color: #0B1F3A !important;
    font-weight: 800 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.stDataFrame td {
    color: #F5F0E8 !important;
}
</style>
"""
