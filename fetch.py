"""
data/fetch.py — Live data fetchers for Barttorvik T-Rank and ESPN bracket API.
Includes team name normalization and fuzzy matching pipeline.
"""

import streamlit as st
import pandas as pd
import requests
import random
import re

try:
    from rapidfuzz import process as fuzz_process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

from config import (
    REGIONS, CONFERENCES, TEAM_ALIASES, NAME_EXPANSIONS,
    BRACKET_MATCHUP_SEEDS,
)


def normalize_team_name(name: str) -> str:
    name = name.strip()
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    name = re.sub(r'\(([^)]+)\)', r'\1', name)
    for abbr, full in NAME_EXPANSIONS.items():
        name = name.replace(abbr, full)
    name = re.sub(r'\s+', ' ', name).strip()
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    return name


def fuzzy_match_team(name: str, choices: list, threshold: int = 78) -> str | None:
    if not FUZZY_AVAILABLE or not choices:
        return None
    norm = normalize_team_name(name)
    if norm in choices:
        return norm
    result = fuzz_process.extractOne(norm, choices)
    if result and result[1] >= threshold:
        return result[0]
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_barttorvik(year: int = 2025) -> pd.DataFrame:
    url = f"https://barttorvik.com/{year}_team_results.json"
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        raw = r.json()
        if not raw or not isinstance(raw, list) or len(raw[0]) < 10:
            return pd.DataFrame()
        records = []
        for row in raw:
            try:
                team = str(row[0]).strip()
                conf = str(row[1]).strip()
                adj_oe = float(row[2])
                adj_de = float(row[3])
                tempo = float(row[4])
                luck = float(row[5])
                opp_adj_oe = float(row[6])
                opp_adj_de = float(row[7])
                if not (70 < adj_oe < 140 and 70 < adj_de < 120):
                    continue
                continuity = float(row[14]) if len(row) > 14 else None
                wins = int(row[8]) if len(row) > 8 else None
                losses = int(row[9]) if len(row) > 9 else None
                records.append({
                    "Team":        normalize_team_name(team),
                    "Conference":  conf,
                    "AdjOE":       round(adj_oe, 2),
                    "AdjDE":       round(adj_de, 2),
                    "Tempo":       round(tempo, 2),
                    "Luck":        round(luck, 3),
                    "AdjEM":       round(adj_oe - adj_de, 2),
                    "OppAdjOE":    round(opp_adj_oe, 2),
                    "OppAdjDE":    round(opp_adj_de, 2),
                    "Continuity":  round(continuity, 1) if continuity is not None else None,
                    "Wins":        wins,
                    "Losses":      losses,
                })
            except (IndexError, ValueError, TypeError):
                continue
        df = pd.DataFrame(records)
        if not df.empty:
            df["SOS"] = (df["OppAdjOE"] - df["OppAdjDE"]).rank(pct=True).round(3)
            df["Record"] = df.apply(
                lambda r: f"{r['Wins']}-{r['Losses']}" if pd.notna(r.get("Wins")) else "?",
                axis=1
            )
            df["Win%"] = df.apply(
                lambda r: round(r["Wins"] / max(r["Wins"] + r["Losses"], 1), 3)
                if pd.notna(r.get("Wins")) else None,
                axis=1
            )
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_espn_bracket() -> pd.DataFrame:
    urls = [
        "https://site.api.espn.com/apis/v2/sports/basketball/mens-college-basketball/tournaments/22?groups=50",
        "https://site.api.espn.com/apis/v2/sports/basketball/mens-college-basketball/tournaments/100?groups=50",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            teams = []
            rounds_data = data.get("bracket", {}).get("rounds", [])
            if not rounds_data:
                continue
            for matchup in rounds_data[0].get("matchups", []):
                region = matchup.get("region", {}).get("name", "Unknown")
                for competitor in matchup.get("competitors", []):
                    raw_name = competitor["team"]["displayName"]
                    teams.append({
                        "Team":    normalize_team_name(raw_name),
                        "RawName": raw_name,
                        "Seed":    int(competitor.get("seed", 0)),
                        "Region":  region,
                    })
            if teams:
                return pd.DataFrame(teams)
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def generate_demo_bracket() -> pd.DataFrame:
    random.seed(42)
    teams = []
    demo_names = {
        1: ["Houston", "UConn", "Purdue", "North Carolina"],
        2: ["Tennessee", "Marquette", "Iowa State", "Arizona"],
        3: ["Kentucky", "Baylor", "Illinois", "Gonzaga"],
        4: ["Duke", "Auburn", "Kansas", "Creighton"],
        5: ["Michigan State", "San Diego State", "Wisconsin", "Clemson"],
        6: ["BYU", "Texas Tech", "Florida", "South Carolina"],
        7: ["Dayton", "Washington State", "Nevada", "Texas"],
        8: ["Mississippi State", "Nebraska", "Memphis", "Utah State"],
        9: ["FAU", "Michigan", "Northwestern", "TCU"],
        10: ["Colorado State", "New Mexico", "Drake", "Colorado"],
        11: ["NC State", "Oregon", "Duquesne", "Pittsburgh"],
        12: ["Grand Canyon", "James Madison", "McNeese State", "UAB"],
        13: ["Vermont", "Oakland", "Yale", "Samford"],
        14: ["Morehead State", "Colgate", "Kent State", "Troy"],
        15: ["Montana State", "UNC Wilmington", "Stony Brook", "South Dakota State"],
        16: ["Stetson", "Wagner", "Howard", "Long Island"],
    }
    for r_idx, region in enumerate(REGIONS):
        for seed in range(1, 17):
            name = demo_names.get(seed, ["Team"])[r_idx % len(demo_names.get(seed, ["Team"]))]
            conference = random.choice(CONFERENCES[:10])
            adj_oe = round(120 - (seed * 2.2) + random.gauss(0, 2.5), 2)
            adj_de = round(90 + (seed * 1.6) + random.gauss(0, 2.0), 2)
            tempo = round(random.gauss(68, 3), 1)
            continuity = round(max(30, min(98, 65 + random.gauss(0, 12))), 1)
            sos = round(max(0.2, min(0.95, 0.70 - (seed * 0.025) + random.gauss(0, 0.08))), 3)
            win_pct = round(max(0.40, min(0.97, 1.0 - (seed * 0.035) + random.gauss(0, 0.04))), 3)
            wins = int(win_pct * 32)
            losses = 32 - wins
            tov_pct = round(max(10, min(25, 16 + (seed * 0.3) + random.gauss(0, 2))), 1)
            three_pct = round(max(0.28, min(0.42, 0.37 - (seed * 0.003) + random.gauss(0, 0.02))), 3)
            three_var = round(max(0.01, random.gauss(0.04, 0.012)), 4)
            ft_rate = round(max(0.15, min(0.45, 0.32 - (seed * 0.005) + random.gauss(0, 0.04))), 3)
            reb_margin = round(6 - (seed * 0.5) + random.gauss(0, 2), 1)
            teams.append({
                "Seed": seed, "Region": region, "Team": name,
                "Conference": conference, "Record": f"{wins}-{losses}",
                "Win%": win_pct, "AdjOE": adj_oe, "AdjDE": adj_de,
                "AdjEM": round(adj_oe - adj_de, 2), "Tempo": tempo,
                "Continuity": continuity, "SOS": sos, "TOV%": tov_pct,
                "3P%": three_pct, "3P_Var": three_var, "FTRate": ft_rate,
                "RebMgn": reb_margin,
            })
    return pd.DataFrame(teams)


@st.cache_data(ttl=1800, show_spinner=False)
def build_full_dataset(year: int = 2025) -> tuple[pd.DataFrame, str]:
    df_espn = fetch_espn_bracket()
    df_trank = fetch_barttorvik(year)
    if df_espn.empty or df_trank.empty:
        return generate_demo_bracket(), "DEMO"
    trank_names = df_trank["Team"].tolist()
    def match_name(name):
        norm = normalize_team_name(name)
        if norm in trank_names:
            return norm
        return fuzzy_match_team(name, trank_names)
    df_espn["TrankName"] = df_espn["Team"].apply(match_name)
    df_merged = df_espn.merge(
        df_trank.rename(columns={"Team": "TrankName"}),
        on="TrankName", how="left"
    ).drop(columns=["TrankName", "RawName"], errors="ignore")
    for col, gen_fn in [
        ("AdjOE",      lambda s: 120 - (s * 2.2) + random.gauss(0, 2)),
        ("AdjDE",      lambda s: 90  + (s * 1.6) + random.gauss(0, 2)),
        ("Continuity", lambda _: round(random.gauss(65, 12), 1)),
        ("Tempo",      lambda _: round(random.gauss(68, 3), 1)),
        ("Conference", lambda _: "Unknown"),
    ]:
        if col not in df_merged.columns:
            df_merged[col] = df_merged["Seed"].apply(gen_fn)
        else:
            mask = df_merged[col].isna()
            df_merged.loc[mask, col] = df_merged.loc[mask, "Seed"].apply(gen_fn)
    df_merged["AdjEM"] = (df_merged["AdjOE"] - df_merged["AdjDE"]).round(2)
    df_merged["SOS"] = df_merged.get("SOS", df_merged["Seed"].apply(
        lambda s: round(0.70 - (s * 0.025) + random.gauss(0, 0.08), 3)
    ))
    for col, gen_fn in [
        ("TOV%",    lambda s: round(16 + (s * 0.3) + random.gauss(0, 2), 1)),
        ("3P%",     lambda s: round(0.37 - (s * 0.003) + random.gauss(0, 0.02), 3)),
        ("3P_Var",  lambda _: round(max(0.01, random.gauss(0.04, 0.012)), 4)),
        ("FTRate",  lambda s: round(0.32 - (s * 0.005) + random.gauss(0, 0.04), 3)),
        ("RebMgn",  lambda s: round(6 - (s * 0.5) + random.gauss(0, 2), 1)),
    ]:
        if col not in df_merged.columns:
            df_merged[col] = df_merged["Seed"].apply(gen_fn)
    df_merged["Win%"] = df_merged.get("Win%", df_merged["Seed"].apply(
        lambda s: round(1.0 - (s * 0.035), 3)
    ))
    return df_merged, "LIVE"
