"""
models/optimizer.py — Expected Value bracket optimizer.

Given Monte Carlo simulation results and a scoring system, computes the
EV-maximizing bracket. Works backward from the championship to R64,
computing each pick's expected point contribution.

Strategies:
  - "max_ev":     Pure expected value maximization (best for large pools)
  - "chalk":      Always pick the favorite (baseline)
  - "contrarian": Weight EV by inverse of ownership % (best for large pools
                  where differentiation matters)
"""

import pandas as pd
import numpy as np
import random
import math

from config import REGIONS, BRACKET_MATCHUP_SEEDS, ROUND_POINTS
from upset import upset_probability


def _get_metric(team, col, default):
    val = team.get(col, default)
    try:
        return float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else default
    except (ValueError, TypeError):
        return default


def _compute_win_prob(t1: dict, t2: dict, round_num: int = 1) -> float:
    """Return probability that t1 beats t2."""
    t1_em = _get_metric(t1, "AdjEM", 0)
    t2_em = _get_metric(t2, "AdjEM", 0)

    if t1_em >= t2_em:
        fav, dog = t1, t2
        is_fav = True
    else:
        fav, dog = t2, t1
        is_fav = False

    up = upset_probability(
        fav_seed=int(_get_metric(fav, "Seed", 1)),
        dog_seed=int(_get_metric(dog, "Seed", 16)),
        fav_adj_oe=_get_metric(fav, "AdjOE", 110),
        dog_adj_oe=_get_metric(dog, "AdjOE", 100),
        fav_adj_de=_get_metric(fav, "AdjDE", 95),
        dog_adj_de=_get_metric(dog, "AdjDE", 105),
        fav_sos=_get_metric(fav, "SOS", 0.55),
        dog_sos=_get_metric(dog, "SOS", 0.45),
        fav_continuity=_get_metric(fav, "Continuity", 70),
        dog_continuity=_get_metric(dog, "Continuity", 70),
        fav_tempo=_get_metric(fav, "Tempo", 68),
        dog_tempo=_get_metric(dog, "Tempo", 68),
        fav_tov=_get_metric(fav, "TOV%", 16),
        dog_tov=_get_metric(dog, "TOV%", 16),
        dog_three_var=_get_metric(dog, "3P_Var", 0.04),
        fav_ft_rate=_get_metric(fav, "FTRate", 0.30),
        dog_ft_rate=_get_metric(dog, "FTRate", 0.30),
        round_num=round_num,
    )

    if is_fav:
        return up["fav_prob"]
    else:
        return up["upset_prob"]


def optimize_bracket(
    df: pd.DataFrame,
    scoring: str = "espn",
    strategy: str = "max_ev",
    pool_size: int = 5,
) -> dict:
    """
    Compute the EV-maximizing bracket.

    Args:
        df:        Full bracket DataFrame with all team stats.
        scoring:   Scoring system key from ROUND_POINTS.
        strategy:  "max_ev", "chalk", or "contrarian".
        pool_size: Number of participants (affects contrarian weighting).

    Returns dict with:
        - "picks": list of dicts {round, region, matchup, pick, ev, win_prob}
        - "total_ev": total expected points for the bracket
        - "strategy": strategy used
    """
    points = ROUND_POINTS.get(scoring, ROUND_POINTS["espn"])
    all_picks = []

    region_winners = {}

    for region in REGIONS:
        reg_df = df[df["Region"] == region].sort_values("Seed")
        seeded = {int(row["Seed"]): row.to_dict() for _, row in reg_df.iterrows()}

        # Round of 64
        r64_winners = []
        for pair_idx, (s1, s2) in enumerate(BRACKET_MATCHUP_SEEDS):
            if s1 not in seeded or s2 not in seeded:
                continue

            t1 = seeded[s1]
            t2 = seeded[s2]
            p1 = _compute_win_prob(t1, t2, round_num=1)
            p2 = 1 - p1

            ev1 = p1 * points[0]
            ev2 = p2 * points[0]

            if strategy == "chalk":
                pick = t1 if s1 < s2 else t2
                pick_prob = p1 if s1 < s2 else p2
                pick_ev = ev1 if s1 < s2 else ev2
            elif strategy == "contrarian":
                public_pick_1 = p1 ** 0.7
                public_pick_2 = 1 - public_pick_1
                leverage_1 = ev1 * (1 - public_pick_1) ** (1 / max(pool_size - 1, 1))
                leverage_2 = ev2 * (1 - public_pick_2) ** (1 / max(pool_size - 1, 1))
                if leverage_1 >= leverage_2:
                    pick, pick_prob, pick_ev = t1, p1, ev1
                else:
                    pick, pick_prob, pick_ev = t2, p2, ev2
            else:  # max_ev
                if ev1 >= ev2:
                    pick, pick_prob, pick_ev = t1, p1, ev1
                else:
                    pick, pick_prob, pick_ev = t2, p2, ev2

            r64_winners.append(pick)
            all_picks.append({
                "Round":    "R64",
                "Region":   region,
                "Matchup":  f"({s1}) vs ({s2})",
                "Pick":     pick["Team"],
                "Seed":     int(_get_metric(pick, "Seed", 0)),
                "WinProb":  round(pick_prob, 3),
                "EV":       round(pick_ev, 2),
            })

        # Later rounds within region (R32, S16, E8)
        pool = r64_winners
        for round_idx in range(1, 4):
            round_name = ["R32", "S16", "E8"][round_idx - 1]
            next_pool = []
            for i in range(0, len(pool), 2):
                if i + 1 >= len(pool):
                    next_pool.append(pool[i])
                    continue

                t1 = pool[i]
                t2 = pool[i + 1]
                p1 = _compute_win_prob(t1, t2, round_num=round_idx + 1)
                p2 = 1 - p1
                ev1 = p1 * points[round_idx]
                ev2 = p2 * points[round_idx]

                if strategy == "chalk":
                    s1 = int(_get_metric(t1, "Seed", 16))
                    s2 = int(_get_metric(t2, "Seed", 16))
                    if s1 <= s2:
                        pick, pick_prob, pick_ev = t1, p1, ev1
                    else:
                        pick, pick_prob, pick_ev = t2, p2, ev2
                elif strategy == "contrarian":
                    public_pick_1 = p1 ** 0.7
                    public_pick_2 = 1 - public_pick_1
                    lev1 = ev1 * (1 - public_pick_1) ** (1 / max(pool_size - 1, 1))
                    lev2 = ev2 * (1 - public_pick_2) ** (1 / max(pool_size - 1, 1))
                    if lev1 >= lev2:
                        pick, pick_prob, pick_ev = t1, p1, ev1
                    else:
                        pick, pick_prob, pick_ev = t2, p2, ev2
                else:
                    if ev1 >= ev2:
                        pick, pick_prob, pick_ev = t1, p1, ev1
                    else:
                        pick, pick_prob, pick_ev = t2, p2, ev2

                next_pool.append(pick)
                all_picks.append({
                    "Round":    round_name,
                    "Region":   region,
                    "Matchup":  f"{t1['Team']} vs {t2['Team']}",
                    "Pick":     pick["Team"],
                    "Seed":     int(_get_metric(pick, "Seed", 0)),
                    "WinProb":  round(pick_prob, 3),
                    "EV":       round(pick_ev, 2),
                })
            pool = next_pool

        if pool:
            region_winners[region] = pool[0]

    # Final Four
    ff_teams = [region_winners.get(r) for r in REGIONS if r in region_winners]

    if len(ff_teams) >= 4:
        for semi_idx, (i, j) in enumerate([(0, 1), (2, 3)]):
            t1 = ff_teams[i]
            t2 = ff_teams[j]
            p1 = _compute_win_prob(t1, t2, round_num=5)
            p2 = 1 - p1
            ev1 = p1 * points[4]
            ev2 = p2 * points[4]

            if ev1 >= ev2:
                pick, pick_prob, pick_ev = t1, p1, ev1
            else:
                pick, pick_prob, pick_ev = t2, p2, ev2

            ff_teams[semi_idx * 2] = pick
            all_picks.append({
                "Round":    "F4",
                "Region":   "Final Four",
                "Matchup":  f"{t1['Team']} vs {t2['Team']}",
                "Pick":     pick["Team"],
                "Seed":     int(_get_metric(pick, "Seed", 0)),
                "WinProb":  round(pick_prob, 3),
                "EV":       round(pick_ev, 2),
            })

        # Championship
        t1 = ff_teams[0]
        t2 = ff_teams[2]
        p1 = _compute_win_prob(t1, t2, round_num=6)
        p2 = 1 - p1
        ev1 = p1 * points[5]
        ev2 = p2 * points[5]

        if ev1 >= ev2:
            pick, pick_prob, pick_ev = t1, p1, ev1
        else:
            pick, pick_prob, pick_ev = t2, p2, ev2

        all_picks.append({
            "Round":    "Champ",
            "Region":   "Championship",
            "Matchup":  f"{t1['Team']} vs {t2['Team']}",
            "Pick":     pick["Team"],
            "Seed":     int(_get_metric(pick, "Seed", 0)),
            "WinProb":  round(pick_prob, 3),
            "EV":       round(pick_ev, 2),
        })

    total_ev = sum(p["EV"] for p in all_picks)

    return {
        "picks":    all_picks,
        "total_ev": round(total_ev, 2),
        "strategy": strategy,
        "scoring":  scoring,
    }
