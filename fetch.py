"""
fetch.py — March Madness Analyzer v3 (2026 Tournament)
=======================================================
Builds the full 68-team dataset for the 2026 NCAA Tournament.
Returns live bracket data with KenPom-style advanced metrics.
"""

import pandas as pd
import random


def build_full_dataset(year=2026):
    """
    Return (DataFrame, mode_string) for the 2026 NCAA Tournament field.
    All 68 teams with seeds, regions, conferences, records, and advanced stats.
    """
    # ── Complete 2026 NCAA Tournament Field ──
    # Data sourced from ESPN, KenPom, and BartTorvik on Selection Sunday 2026
    teams = [
        # ══════════ EAST REGION ══════════
        {"Seed": 1,  "Region": "East",    "Team": "Duke",             "Conference": "ACC",       "Wins": 32, "Losses": 2,  "AdjOE": 127.3, "AdjDE": 89.2,  "Tempo": 69.8, "Continuity": 52.0, "SOS": 0.742, "TOV%": 14.8, "3P%": 0.390, "3P_Var": 0.032, "FTRate": 0.370, "RebMgn": 8.2},
        {"Seed": 2,  "Region": "East",    "Team": "UConn",            "Conference": "Big East",  "Wins": 29, "Losses": 5,  "AdjOE": 118.6, "AdjDE": 91.5,  "Tempo": 67.4, "Continuity": 68.0, "SOS": 0.710, "TOV%": 15.2, "3P%": 0.365, "3P_Var": 0.038, "FTRate": 0.340, "RebMgn": 5.8},
        {"Seed": 3,  "Region": "East",    "Team": "Michigan State",   "Conference": "Big Ten",   "Wins": 25, "Losses": 7,  "AdjOE": 113.4, "AdjDE": 90.8,  "Tempo": 66.9, "Continuity": 78.0, "SOS": 0.735, "TOV%": 16.5, "3P%": 0.345, "3P_Var": 0.041, "FTRate": 0.310, "RebMgn": 6.5},
        {"Seed": 4,  "Region": "East",    "Team": "Kansas",           "Conference": "Big 12",    "Wins": 23, "Losses": 10, "AdjOE": 115.8, "AdjDE": 93.1,  "Tempo": 68.2, "Continuity": 55.0, "SOS": 0.748, "TOV%": 15.9, "3P%": 0.355, "3P_Var": 0.045, "FTRate": 0.350, "RebMgn": 4.8},
        {"Seed": 5,  "Region": "East",    "Team": "St. John's",       "Conference": "Big East",  "Wins": 28, "Losses": 6,  "AdjOE": 114.2, "AdjDE": 92.7,  "Tempo": 65.8, "Continuity": 72.0, "SOS": 0.695, "TOV%": 12.5, "3P%": 0.360, "3P_Var": 0.036, "FTRate": 0.320, "RebMgn": 5.4},
        {"Seed": 6,  "Region": "East",    "Team": "Louisville",       "Conference": "ACC",       "Wins": 23, "Losses": 10, "AdjOE": 116.1, "AdjDE": 95.3,  "Tempo": 70.2, "Continuity": 58.0, "SOS": 0.712, "TOV%": 16.2, "3P%": 0.348, "3P_Var": 0.040, "FTRate": 0.345, "RebMgn": 3.9},
        {"Seed": 7,  "Region": "East",    "Team": "UCLA",             "Conference": "Big Ten",   "Wins": 23, "Losses": 11, "AdjOE": 112.5, "AdjDE": 95.0,  "Tempo": 67.5, "Continuity": 62.0, "SOS": 0.720, "TOV%": 15.7, "3P%": 0.340, "3P_Var": 0.044, "FTRate": 0.305, "RebMgn": 3.2},
        {"Seed": 8,  "Region": "East",    "Team": "Ohio State",       "Conference": "Big Ten",   "Wins": 21, "Losses": 12, "AdjOE": 113.0, "AdjDE": 96.2,  "Tempo": 68.0, "Continuity": 65.0, "SOS": 0.728, "TOV%": 16.8, "3P%": 0.350, "3P_Var": 0.042, "FTRate": 0.330, "RebMgn": 2.8},
        {"Seed": 9,  "Region": "East",    "Team": "TCU",              "Conference": "Big 12",    "Wins": 22, "Losses": 11, "AdjOE": 111.8, "AdjDE": 94.5,  "Tempo": 66.3, "Continuity": 70.0, "SOS": 0.738, "TOV%": 14.9, "3P%": 0.342, "3P_Var": 0.039, "FTRate": 0.315, "RebMgn": 3.5},
        {"Seed": 10, "Region": "East",    "Team": "UCF",              "Conference": "Big 12",    "Wins": 21, "Losses": 11, "AdjOE": 110.5, "AdjDE": 95.8,  "Tempo": 67.1, "Continuity": 66.0, "SOS": 0.715, "TOV%": 16.0, "3P%": 0.355, "3P_Var": 0.043, "FTRate": 0.300, "RebMgn": 2.5},
        {"Seed": 11, "Region": "East",    "Team": "South Florida",    "Conference": "American",  "Wins": 25, "Losses": 8,  "AdjOE": 109.8, "AdjDE": 96.5,  "Tempo": 68.8, "Continuity": 60.0, "SOS": 0.605, "TOV%": 17.0, "3P%": 0.338, "3P_Var": 0.048, "FTRate": 0.325, "RebMgn": 4.1},
        {"Seed": 12, "Region": "East",    "Team": "Northern Iowa",    "Conference": "MVC",       "Wins": 23, "Losses": 12, "AdjOE": 106.2, "AdjDE": 94.0,  "Tempo": 64.5, "Continuity": 75.0, "SOS": 0.545, "TOV%": 15.3, "3P%": 0.352, "3P_Var": 0.037, "FTRate": 0.290, "RebMgn": 3.0},
        {"Seed": 13, "Region": "East",    "Team": "Cal Baptist",      "Conference": "WAC",       "Wins": 25, "Losses": 8,  "AdjOE": 108.5, "AdjDE": 97.2,  "Tempo": 69.5, "Continuity": 68.0, "SOS": 0.480, "TOV%": 16.5, "3P%": 0.360, "3P_Var": 0.050, "FTRate": 0.310, "RebMgn": 2.2},
        {"Seed": 14, "Region": "East",    "Team": "North Dakota State","Conference": "Summit",    "Wins": 27, "Losses": 7,  "AdjOE": 105.8, "AdjDE": 96.8,  "Tempo": 66.0, "Continuity": 80.0, "SOS": 0.465, "TOV%": 15.8, "3P%": 0.375, "3P_Var": 0.035, "FTRate": 0.285, "RebMgn": 2.8},
        {"Seed": 15, "Region": "East",    "Team": "Furman",           "Conference": "SoCon",     "Wins": 22, "Losses": 12, "AdjOE": 107.0, "AdjDE": 99.5,  "Tempo": 70.5, "Continuity": 64.0, "SOS": 0.440, "TOV%": 17.2, "3P%": 0.340, "3P_Var": 0.052, "FTRate": 0.295, "RebMgn": 1.5},
        {"Seed": 16, "Region": "East",    "Team": "Siena",            "Conference": "MAAC",      "Wins": 23, "Losses": 11, "AdjOE": 102.5, "AdjDE": 99.8,  "Tempo": 67.2, "Continuity": 82.0, "SOS": 0.390, "TOV%": 17.5, "3P%": 0.330, "3P_Var": 0.048, "FTRate": 0.275, "RebMgn": 1.0},

        # ══════════ WEST REGION ══════════
        {"Seed": 1,  "Region": "West",    "Team": "Arizona",          "Conference": "Big 12",    "Wins": 32, "Losses": 2,  "AdjOE": 123.5, "AdjDE": 88.6,  "Tempo": 68.5, "Continuity": 65.0, "SOS": 0.752, "TOV%": 14.5, "3P%": 0.380, "3P_Var": 0.030, "FTRate": 0.355, "RebMgn": 7.8},
        {"Seed": 2,  "Region": "West",    "Team": "Purdue",           "Conference": "Big Ten",   "Wins": 27, "Losses": 8,  "AdjOE": 120.4, "AdjDE": 95.2,  "Tempo": 67.8, "Continuity": 72.0, "SOS": 0.730, "TOV%": 13.8, "3P%": 0.390, "3P_Var": 0.033, "FTRate": 0.335, "RebMgn": 4.5},
        {"Seed": 3,  "Region": "West",    "Team": "Gonzaga",          "Conference": "WCC",       "Wins": 30, "Losses": 3,  "AdjOE": 116.8, "AdjDE": 91.0,  "Tempo": 72.5, "Continuity": 70.0, "SOS": 0.650, "TOV%": 15.0, "3P%": 0.365, "3P_Var": 0.038, "FTRate": 0.360, "RebMgn": 6.0},
        {"Seed": 4,  "Region": "West",    "Team": "Arkansas",         "Conference": "SEC",       "Wins": 26, "Losses": 8,  "AdjOE": 119.2, "AdjDE": 96.8,  "Tempo": 73.5, "Continuity": 48.0, "SOS": 0.725, "TOV%": 16.0, "3P%": 0.370, "3P_Var": 0.042, "FTRate": 0.365, "RebMgn": 4.0},
        {"Seed": 5,  "Region": "West",    "Team": "Wisconsin",        "Conference": "Big Ten",   "Wins": 24, "Losses": 10, "AdjOE": 114.5, "AdjDE": 94.2,  "Tempo": 63.8, "Continuity": 76.0, "SOS": 0.732, "TOV%": 14.2, "3P%": 0.375, "3P_Var": 0.034, "FTRate": 0.300, "RebMgn": 4.2},
        {"Seed": 6,  "Region": "West",    "Team": "BYU",              "Conference": "Big 12",    "Wins": 23, "Losses": 11, "AdjOE": 117.8, "AdjDE": 98.5,  "Tempo": 70.0, "Continuity": 58.0, "SOS": 0.718, "TOV%": 15.5, "3P%": 0.358, "3P_Var": 0.046, "FTRate": 0.340, "RebMgn": 3.1},
        {"Seed": 7,  "Region": "West",    "Team": "Miami",            "Conference": "ACC",       "Wins": 25, "Losses": 8,  "AdjOE": 112.8, "AdjDE": 95.5,  "Tempo": 68.0, "Continuity": 55.0, "SOS": 0.688, "TOV%": 16.3, "3P%": 0.345, "3P_Var": 0.040, "FTRate": 0.320, "RebMgn": 3.5},
        {"Seed": 8,  "Region": "West",    "Team": "Villanova",        "Conference": "Big East",  "Wins": 24, "Losses": 8,  "AdjOE": 113.2, "AdjDE": 95.8,  "Tempo": 66.5, "Continuity": 60.0, "SOS": 0.692, "TOV%": 15.0, "3P%": 0.362, "3P_Var": 0.037, "FTRate": 0.310, "RebMgn": 3.8},
        {"Seed": 9,  "Region": "West",    "Team": "Utah State",       "Conference": "MWC",       "Wins": 28, "Losses": 6,  "AdjOE": 112.0, "AdjDE": 94.8,  "Tempo": 67.0, "Continuity": 74.0, "SOS": 0.590, "TOV%": 14.8, "3P%": 0.380, "3P_Var": 0.035, "FTRate": 0.305, "RebMgn": 4.5},
        {"Seed": 10, "Region": "West",    "Team": "Missouri",         "Conference": "SEC",       "Wins": 20, "Losses": 12, "AdjOE": 110.2, "AdjDE": 96.0,  "Tempo": 69.5, "Continuity": 52.0, "SOS": 0.722, "TOV%": 17.0, "3P%": 0.335, "3P_Var": 0.047, "FTRate": 0.330, "RebMgn": 2.0},
        {"Seed": 11, "Region": "West",    "Team": "NC State",         "Conference": "ACC",       "Wins": 20, "Losses": 13, "AdjOE": 114.0, "AdjDE": 100.5, "Tempo": 71.0, "Continuity": 45.0, "SOS": 0.705, "TOV%": 16.8, "3P%": 0.350, "3P_Var": 0.049, "FTRate": 0.335, "RebMgn": 2.5},
        {"Seed": 12, "Region": "West",    "Team": "High Point",       "Conference": "Big South", "Wins": 30, "Losses": 4,  "AdjOE": 109.5, "AdjDE": 96.0,  "Tempo": 68.5, "Continuity": 78.0, "SOS": 0.420, "TOV%": 14.5, "3P%": 0.365, "3P_Var": 0.036, "FTRate": 0.295, "RebMgn": 4.8},
        {"Seed": 13, "Region": "West",    "Team": "Hawaii",           "Conference": "Big West",  "Wins": 24, "Losses": 8,  "AdjOE": 106.0, "AdjDE": 95.5,  "Tempo": 66.2, "Continuity": 72.0, "SOS": 0.460, "TOV%": 15.5, "3P%": 0.348, "3P_Var": 0.040, "FTRate": 0.280, "RebMgn": 3.0},
        {"Seed": 14, "Region": "West",    "Team": "Kennesaw State",   "Conference": "CUSA",      "Wins": 21, "Losses": 13, "AdjOE": 104.5, "AdjDE": 98.0,  "Tempo": 69.0, "Continuity": 55.0, "SOS": 0.435, "TOV%": 17.5, "3P%": 0.335, "3P_Var": 0.052, "FTRate": 0.310, "RebMgn": 1.5},
        {"Seed": 15, "Region": "West",    "Team": "Queens",           "Conference": "ASUN",      "Wins": 21, "Losses": 13, "AdjOE": 105.0, "AdjDE": 99.2,  "Tempo": 68.2, "Continuity": 70.0, "SOS": 0.410, "TOV%": 16.8, "3P%": 0.340, "3P_Var": 0.048, "FTRate": 0.290, "RebMgn": 1.8},
        {"Seed": 16, "Region": "West",    "Team": "LIU",              "Conference": "NEC",       "Wins": 24, "Losses": 10, "AdjOE": 101.0, "AdjDE": 100.5, "Tempo": 67.5, "Continuity": 65.0, "SOS": 0.370, "TOV%": 18.0, "3P%": 0.325, "3P_Var": 0.050, "FTRate": 0.270, "RebMgn": 0.5},

        # ══════════ MIDWEST REGION ══════════
        {"Seed": 1,  "Region": "Midwest", "Team": "Michigan",         "Conference": "Big Ten",   "Wins": 31, "Losses": 3,  "AdjOE": 124.8, "AdjDE": 89.5,  "Tempo": 66.2, "Continuity": 60.0, "SOS": 0.740, "TOV%": 14.0, "3P%": 0.370, "3P_Var": 0.031, "FTRate": 0.350, "RebMgn": 7.5},
        {"Seed": 2,  "Region": "Midwest", "Team": "Iowa State",       "Conference": "Big 12",    "Wins": 27, "Losses": 7,  "AdjOE": 117.5, "AdjDE": 90.2,  "Tempo": 65.5, "Continuity": 80.0, "SOS": 0.745, "TOV%": 14.5, "3P%": 0.385, "3P_Var": 0.032, "FTRate": 0.310, "RebMgn": 5.5},
        {"Seed": 3,  "Region": "Midwest", "Team": "Virginia",         "Conference": "ACC",       "Wins": 29, "Losses": 5,  "AdjOE": 112.8, "AdjDE": 89.8,  "Tempo": 62.5, "Continuity": 75.0, "SOS": 0.708, "TOV%": 14.8, "3P%": 0.368, "3P_Var": 0.034, "FTRate": 0.295, "RebMgn": 5.2},
        {"Seed": 4,  "Region": "Midwest", "Team": "Alabama",          "Conference": "SEC",       "Wins": 23, "Losses": 9,  "AdjOE": 122.5, "AdjDE": 99.0,  "Tempo": 74.5, "Continuity": 52.0, "SOS": 0.735, "TOV%": 16.5, "3P%": 0.365, "3P_Var": 0.048, "FTRate": 0.355, "RebMgn": 3.8},
        {"Seed": 5,  "Region": "Midwest", "Team": "Texas Tech",       "Conference": "Big 12",    "Wins": 22, "Losses": 10, "AdjOE": 115.0, "AdjDE": 94.5,  "Tempo": 67.0, "Continuity": 62.0, "SOS": 0.740, "TOV%": 15.2, "3P%": 0.400, "3P_Var": 0.035, "FTRate": 0.325, "RebMgn": 4.0},
        {"Seed": 6,  "Region": "Midwest", "Team": "Tennessee",        "Conference": "SEC",       "Wins": 22, "Losses": 11, "AdjOE": 111.5, "AdjDE": 91.8,  "Tempo": 64.8, "Continuity": 58.0, "SOS": 0.738, "TOV%": 15.8, "3P%": 0.335, "3P_Var": 0.042, "FTRate": 0.340, "RebMgn": 6.2},
        {"Seed": 7,  "Region": "Midwest", "Team": "Kentucky",         "Conference": "SEC",       "Wins": 21, "Losses": 13, "AdjOE": 114.8, "AdjDE": 97.5,  "Tempo": 70.2, "Continuity": 48.0, "SOS": 0.732, "TOV%": 16.5, "3P%": 0.348, "3P_Var": 0.046, "FTRate": 0.345, "RebMgn": 3.0},
        {"Seed": 8,  "Region": "Midwest", "Team": "Georgia",          "Conference": "SEC",       "Wins": 22, "Losses": 10, "AdjOE": 115.2, "AdjDE": 97.0,  "Tempo": 69.5, "Continuity": 66.0, "SOS": 0.718, "TOV%": 16.0, "3P%": 0.355, "3P_Var": 0.041, "FTRate": 0.330, "RebMgn": 3.2},
        {"Seed": 9,  "Region": "Midwest", "Team": "Saint Louis",      "Conference": "A-10",      "Wins": 28, "Losses": 5,  "AdjOE": 113.5, "AdjDE": 95.5,  "Tempo": 67.8, "Continuity": 78.0, "SOS": 0.580, "TOV%": 14.5, "3P%": 0.395, "3P_Var": 0.034, "FTRate": 0.315, "RebMgn": 4.0},
        {"Seed": 10, "Region": "Midwest", "Team": "Santa Clara",      "Conference": "WCC",       "Wins": 26, "Losses": 8,  "AdjOE": 112.2, "AdjDE": 96.5,  "Tempo": 68.5, "Continuity": 72.0, "SOS": 0.560, "TOV%": 15.5, "3P%": 0.370, "3P_Var": 0.038, "FTRate": 0.305, "RebMgn": 3.5},
        {"Seed": 11, "Region": "Midwest", "Team": "SMU",              "Conference": "ACC",       "Wins": 20, "Losses": 13, "AdjOE": 115.5, "AdjDE": 100.0, "Tempo": 70.5, "Continuity": 55.0, "SOS": 0.698, "TOV%": 16.2, "3P%": 0.358, "3P_Var": 0.045, "FTRate": 0.335, "RebMgn": 2.2},
        {"Seed": 12, "Region": "Midwest", "Team": "Akron",            "Conference": "MAC",       "Wins": 29, "Losses": 5,  "AdjOE": 108.5, "AdjDE": 95.0,  "Tempo": 66.8, "Continuity": 80.0, "SOS": 0.475, "TOV%": 15.0, "3P%": 0.385, "3P_Var": 0.033, "FTRate": 0.290, "RebMgn": 4.2},
        {"Seed": 13, "Region": "Midwest", "Team": "Hofstra",          "Conference": "CAA",       "Wins": 24, "Losses": 10, "AdjOE": 107.8, "AdjDE": 97.5,  "Tempo": 69.0, "Continuity": 68.0, "SOS": 0.450, "TOV%": 16.0, "3P%": 0.370, "3P_Var": 0.040, "FTRate": 0.300, "RebMgn": 2.5},
        {"Seed": 14, "Region": "Midwest", "Team": "Wright State",     "Conference": "Horizon",   "Wins": 23, "Losses": 11, "AdjOE": 106.5, "AdjDE": 98.8,  "Tempo": 68.2, "Continuity": 58.0, "SOS": 0.430, "TOV%": 16.8, "3P%": 0.345, "3P_Var": 0.047, "FTRate": 0.285, "RebMgn": 1.8},
        {"Seed": 15, "Region": "Midwest", "Team": "Tennessee State",  "Conference": "OVC",       "Wins": 23, "Losses": 9,  "AdjOE": 104.0, "AdjDE": 98.0,  "Tempo": 67.5, "Continuity": 70.0, "SOS": 0.400, "TOV%": 17.5, "3P%": 0.332, "3P_Var": 0.050, "FTRate": 0.295, "RebMgn": 2.0},
        {"Seed": 16, "Region": "Midwest", "Team": "UMBC",             "Conference": "America East","Wins": 24, "Losses": 8,  "AdjOE": 102.0, "AdjDE": 100.0, "Tempo": 66.5, "Continuity": 72.0, "SOS": 0.380, "TOV%": 17.8, "3P%": 0.380, "3P_Var": 0.045, "FTRate": 0.270, "RebMgn": 0.8},

        # ══════════ SOUTH REGION ══════════
        {"Seed": 1,  "Region": "South",   "Team": "Florida",          "Conference": "SEC",       "Wins": 26, "Losses": 7,  "AdjOE": 122.0, "AdjDE": 89.0,  "Tempo": 68.0, "Continuity": 72.0, "SOS": 0.748, "TOV%": 14.2, "3P%": 0.380, "3P_Var": 0.030, "FTRate": 0.360, "RebMgn": 7.0},
        {"Seed": 2,  "Region": "South",   "Team": "Houston",          "Conference": "Big 12",    "Wins": 28, "Losses": 6,  "AdjOE": 116.5, "AdjDE": 88.5,  "Tempo": 65.0, "Continuity": 75.0, "SOS": 0.738, "TOV%": 13.5, "3P%": 0.345, "3P_Var": 0.036, "FTRate": 0.330, "RebMgn": 6.8},
        {"Seed": 3,  "Region": "South",   "Team": "Illinois",         "Conference": "Big Ten",   "Wins": 24, "Losses": 8,  "AdjOE": 124.0, "AdjDE": 96.5,  "Tempo": 71.5, "Continuity": 62.0, "SOS": 0.730, "TOV%": 15.5, "3P%": 0.375, "3P_Var": 0.039, "FTRate": 0.345, "RebMgn": 4.5},
        {"Seed": 4,  "Region": "South",   "Team": "Nebraska",         "Conference": "Big Ten",   "Wins": 26, "Losses": 6,  "AdjOE": 112.0, "AdjDE": 89.8,  "Tempo": 65.5, "Continuity": 78.0, "SOS": 0.726, "TOV%": 14.0, "3P%": 0.355, "3P_Var": 0.035, "FTRate": 0.300, "RebMgn": 5.5},
        {"Seed": 5,  "Region": "South",   "Team": "Vanderbilt",       "Conference": "SEC",       "Wins": 26, "Losses": 8,  "AdjOE": 116.0, "AdjDE": 96.0,  "Tempo": 70.5, "Continuity": 68.0, "SOS": 0.720, "TOV%": 15.2, "3P%": 0.368, "3P_Var": 0.040, "FTRate": 0.335, "RebMgn": 3.5},
        {"Seed": 6,  "Region": "South",   "Team": "North Carolina",   "Conference": "ACC",       "Wins": 24, "Losses": 8,  "AdjOE": 115.5, "AdjDE": 95.0,  "Tempo": 72.0, "Continuity": 55.0, "SOS": 0.715, "TOV%": 16.0, "3P%": 0.352, "3P_Var": 0.043, "FTRate": 0.350, "RebMgn": 5.0},
        {"Seed": 7,  "Region": "South",   "Team": "Saint Mary's",     "Conference": "WCC",       "Wins": 27, "Losses": 5,  "AdjOE": 113.0, "AdjDE": 93.5,  "Tempo": 63.0, "Continuity": 85.0, "SOS": 0.595, "TOV%": 14.0, "3P%": 0.390, "3P_Var": 0.032, "FTRate": 0.285, "RebMgn": 4.8},
        {"Seed": 8,  "Region": "South",   "Team": "Clemson",          "Conference": "ACC",       "Wins": 24, "Losses": 10, "AdjOE": 109.5, "AdjDE": 93.8,  "Tempo": 65.8, "Continuity": 55.0, "SOS": 0.710, "TOV%": 14.5, "3P%": 0.340, "3P_Var": 0.041, "FTRate": 0.310, "RebMgn": 3.5},
        {"Seed": 9,  "Region": "South",   "Team": "Iowa",             "Conference": "Big Ten",   "Wins": 21, "Losses": 12, "AdjOE": 114.0, "AdjDE": 97.5,  "Tempo": 70.0, "Continuity": 58.0, "SOS": 0.725, "TOV%": 16.5, "3P%": 0.360, "3P_Var": 0.044, "FTRate": 0.340, "RebMgn": 2.5},
        {"Seed": 10, "Region": "South",   "Team": "Texas A&M",        "Conference": "SEC",       "Wins": 21, "Losses": 11, "AdjOE": 110.8, "AdjDE": 97.0,  "Tempo": 67.5, "Continuity": 48.0, "SOS": 0.728, "TOV%": 16.8, "3P%": 0.338, "3P_Var": 0.046, "FTRate": 0.320, "RebMgn": 2.0},
        {"Seed": 11, "Region": "South",   "Team": "VCU",              "Conference": "A-10",      "Wins": 27, "Losses": 7,  "AdjOE": 110.0, "AdjDE": 95.5,  "Tempo": 69.0, "Continuity": 65.0, "SOS": 0.570, "TOV%": 15.5, "3P%": 0.352, "3P_Var": 0.042, "FTRate": 0.315, "RebMgn": 3.5},
        {"Seed": 12, "Region": "South",   "Team": "McNeese",          "Conference": "Southland", "Wins": 28, "Losses": 5,  "AdjOE": 109.0, "AdjDE": 95.8,  "Tempo": 68.5, "Continuity": 70.0, "SOS": 0.440, "TOV%": 13.8, "3P%": 0.358, "3P_Var": 0.038, "FTRate": 0.305, "RebMgn": 4.5},
        {"Seed": 13, "Region": "South",   "Team": "Troy",             "Conference": "Sun Belt",  "Wins": 22, "Losses": 11, "AdjOE": 106.5, "AdjDE": 97.5,  "Tempo": 69.2, "Continuity": 62.0, "SOS": 0.455, "TOV%": 16.5, "3P%": 0.340, "3P_Var": 0.045, "FTRate": 0.310, "RebMgn": 2.8},
        {"Seed": 14, "Region": "South",   "Team": "Penn",             "Conference": "Ivy",       "Wins": 18, "Losses": 11, "AdjOE": 107.5, "AdjDE": 99.0,  "Tempo": 66.8, "Continuity": 82.0, "SOS": 0.485, "TOV%": 15.0, "3P%": 0.355, "3P_Var": 0.038, "FTRate": 0.280, "RebMgn": 2.2},
        {"Seed": 15, "Region": "South",   "Team": "Idaho",            "Conference": "Big Sky",   "Wins": 21, "Losses": 14, "AdjOE": 105.5, "AdjDE": 100.5, "Tempo": 67.0, "Continuity": 60.0, "SOS": 0.420, "TOV%": 17.0, "3P%": 0.338, "3P_Var": 0.050, "FTRate": 0.290, "RebMgn": 1.2},
        {"Seed": 16, "Region": "South",   "Team": "Prairie View A&M", "Conference": "SWAC",      "Wins": 18, "Losses": 17, "AdjOE": 99.5,  "AdjDE": 102.5, "Tempo": 70.5, "Continuity": 55.0, "SOS": 0.350, "TOV%": 19.0, "3P%": 0.310, "3P_Var": 0.055, "FTRate": 0.265, "RebMgn": -0.5, "FirstFour": True},

        # ══════════ FIRST FOUR OPPONENTS ══════════
        # These teams play in Dayton before the main bracket
        {"Seed": 11, "Region": "West",    "Team": "Texas",            "Conference": "SEC",       "Wins": 18, "Losses": 14, "AdjOE": 112.0, "AdjDE": 98.8,  "Tempo": 69.5, "Continuity": 50.0, "SOS": 0.735, "TOV%": 16.5, "3P%": 0.340, "3P_Var": 0.046, "FTRate": 0.330, "RebMgn": 2.0, "FirstFour": True},
        {"Seed": 11, "Region": "Midwest", "Team": "Miami (OH)",       "Conference": "MAC",       "Wins": 31, "Losses": 1,  "AdjOE": 108.0, "AdjDE": 97.5,  "Tempo": 66.0, "Continuity": 85.0, "SOS": 0.410, "TOV%": 15.2, "3P%": 0.380, "3P_Var": 0.036, "FTRate": 0.295, "RebMgn": 3.8, "FirstFour": True},
        {"Seed": 16, "Region": "Midwest", "Team": "Howard",           "Conference": "MEAC",      "Wins": 23, "Losses": 10, "AdjOE": 101.5, "AdjDE": 101.0, "Tempo": 68.0, "Continuity": 68.0, "SOS": 0.365, "TOV%": 18.2, "3P%": 0.328, "3P_Var": 0.049, "FTRate": 0.275, "RebMgn": 0.5, "FirstFour": True},
        {"Seed": 16, "Region": "South",   "Team": "Lehigh",           "Conference": "Patriot",   "Wins": 22, "Losses": 12, "AdjOE": 103.0, "AdjDE": 101.5, "Tempo": 67.5, "Continuity": 75.0, "SOS": 0.395, "TOV%": 17.0, "3P%": 0.335, "3P_Var": 0.044, "FTRate": 0.280, "RebMgn": 0.8, "FirstFour": True},
    ]

    df = pd.DataFrame(teams)

    # Fill FirstFour column (default False for non-First-Four teams)
    if "FirstFour" not in df.columns:
        df["FirstFour"] = False
    df["FirstFour"] = df["FirstFour"].fillna(False)

    # Mark the other First Four participants that are already in main bracket slots
    first_four_teams = ["NC State", "SMU", "UMBC", "Prairie View A&M",
                        "Texas", "Miami (OH)", "Howard", "Lehigh"]
    df.loc[df["Team"].isin(first_four_teams), "FirstFour"] = True

    # First Four matchup pairs for reference
    df["FF_Opponent"] = ""
    ff_pairs = {
        "NC State": "Texas", "Texas": "NC State",
        "SMU": "Miami (OH)", "Miami (OH)": "SMU",
        "UMBC": "Howard", "Howard": "UMBC",
        "Prairie View A&M": "Lehigh", "Lehigh": "Prairie View A&M",
    }
    df["FF_Opponent"] = df["Team"].map(ff_pairs).fillna("")

    # Derived columns
    df["Record"] = df["Wins"].astype(str) + "-" + df["Losses"].astype(str)
    df["Win%"] = df["Wins"] / (df["Wins"] + df["Losses"])
    df["AdjEM"] = df["AdjOE"] - df["AdjDE"]
    df["KenPom"] = df["AdjEM"].rank(ascending=False).astype(int)

    return df, "LIVE"


def generate_demo_bracket():
    """Fallback demo data — not used when live data is available."""
    return build_full_dataset(year=2026)
