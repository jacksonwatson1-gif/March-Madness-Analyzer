"""
config.py — Constants, historical data, scoring systems, and team name aliases.
"""

REGIONS = ["East", "West", "South", "Midwest"]

ROUNDS = [
    "Round of 64", "Round of 32", "Sweet 16",
    "Elite Eight", "Final Four", "Championship",
]

ROUND_POINTS = {
    # ESPN Tournament Challenge scoring
    "espn":   [10, 20, 40, 80, 160, 320],
    # Standard progressive
    "standard": [1, 2, 4, 8, 16, 32],
    # Upset bonus (seed x round multiplier)
    "upset_bonus": [1, 2, 4, 8, 16, 32],
}

CONFERENCES = [
    "ACC", "Big Ten", "Big 12", "SEC", "Big East",
    "Pac-12", "American", "Mountain West", "WCC", "A-10",
    "Missouri Valley", "CAA", "WAC", "Horizon", "MAAC",
    "Big Sky", "Southland", "Sun Belt", "CUSA", "Ivy",
    "Patriot", "Summit", "Big South", "Atlantic Sun",
    "Ohio Valley", "NEC", "MEAC", "SWAC", "America East",
]

# Historical first-round upset rates (1985-2024, ~40 tournaments)
HISTORICAL_UPSET_RATES = {
    (1, 16): 0.013,
    (2, 15): 0.063,
    (3, 14): 0.152,
    (4, 13): 0.202,
    (5, 12): 0.359,
    (6, 11): 0.370,
    (7, 10): 0.392,
    (8,  9): 0.489,
}

# Historical win rates by seed advancing through each round (1985-2024)
SEED_ROUND_WIN_RATES = {
    1:  [0.993, 0.874, 0.650, 0.500, 0.380, 0.280],
    2:  [0.937, 0.680, 0.480, 0.310, 0.200, 0.130],
    3:  [0.848, 0.540, 0.340, 0.190, 0.100, 0.055],
    4:  [0.798, 0.436, 0.260, 0.140, 0.065, 0.035],
    5:  [0.641, 0.318, 0.170, 0.075, 0.035, 0.015],
    6:  [0.630, 0.296, 0.145, 0.060, 0.025, 0.012],
    7:  [0.608, 0.280, 0.120, 0.045, 0.018, 0.008],
    8:  [0.511, 0.240, 0.095, 0.035, 0.012, 0.005],
    9:  [0.489, 0.165, 0.065, 0.020, 0.008, 0.003],
    10: [0.392, 0.155, 0.060, 0.018, 0.006, 0.002],
    11: [0.370, 0.163, 0.068, 0.022, 0.008, 0.003],
    12: [0.359, 0.152, 0.040, 0.012, 0.004, 0.001],
    13: [0.202, 0.074, 0.015, 0.003, 0.001, 0.000],
    14: [0.152, 0.063, 0.012, 0.002, 0.000, 0.000],
    15: [0.063, 0.027, 0.008, 0.002, 0.001, 0.000],
    16: [0.013, 0.006, 0.002, 0.000, 0.000, 0.000],
}

# Round-of-64 bracket pairing order (matches standard NCAA bracket tree)
BRACKET_MATCHUP_SEEDS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]

# Team Name Normalization Aliases
# Maps common ESPN / AP / media names to Barttorvik canonical names.
TEAM_ALIASES = {
    "UConn":                   "Connecticut",
    "UCONN":                   "Connecticut",
    "Connecticut Huskies":     "Connecticut",
    "Pitt":                    "Pittsburgh",
    "Ole Miss":                "Mississippi",
    "SMU":                     "Southern Methodist",
    "USC":                     "Southern California",
    "UCF":                     "Central Florida",
    "UNC":                     "North Carolina",
    "UNLV":                    "Nevada Las Vegas",
    "VCU":                     "Virginia Commonwealth",
    "BYU":                     "Brigham Young",
    "LSU":                     "Louisiana State",
    "TCU":                     "Texas Christian",
    "UAB":                     "Alabama Birmingham",
    "UTEP":                    "Texas El Paso",
    "UMBC":                    "Maryland Baltimore County",
    "FDU":                     "Fairleigh Dickinson",
    "FGCU":                    "Florida Gulf Coast",
    "LIU":                     "Long Island University",
    "SIU":                     "Southern Illinois",
    "NIU":                     "Northern Illinois",
    "SIUE":                    "SIU Edwardsville",
    "UMass":                   "Massachusetts",
    "UNI":                     "Northern Iowa",
    "ETSU":                    "East Tennessee St.",
    "MTSU":                    "Middle Tennessee",
    "Mid. Tennessee":          "Middle Tennessee",
    "Middle Tennessee State":  "Middle Tennessee",
    "St. Mary's":              "Saint Mary's",
    "St. Mary's (CA)":         "Saint Mary's",
    "St. John's":              "Saint John's",
    "St. Peter's":             "Saint Peter's",
    "St. Bonaventure":         "Saint Bonaventure",
    "St. Joseph's":            "Saint Joseph's",
    "NC State":                "North Carolina St.",
    "Miami (FL)":              "Miami FL",
    "Miami":                   "Miami FL",
    "Loyola Chicago":          "Loyola-Chicago",
    "Loyola (MD)":             "Loyola MD",
    "Texas A&M":               "Texas A&M",
}

# Abbreviation expansions for normalization
NAME_EXPANSIONS = {
    "St.":    "Saint",
    "Univ.":  "University",
    "N.":     "North",
    "S.":     "South",
    "E.":     "East",
    "W.":     "West",
    "C.":     "Central",
}
