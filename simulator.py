"""
models/simulator.py — Monte Carlo tournament simulation engine.

Improvements over v1:
  1. Round-specific probability adjustments (later rounds slightly more volatile).
  2. Confidence intervals (standard error) on championship probabilities.
  3. Per-round advancement tracking (not just championship).
  4. Secondary tiebreaker for identical KenPom teams.
"""

import random
import math
import pandas as pd
import numpy as np

from config import REGIONS, BRACKET_MATCHUP_SEEDS
from upset import upset_probability


def _get_team_metric(team, col, default):
    """Safely extract a metric from a team row (dict or Series)."""
    val = team.get(col, default) if isinstance(team, dict) else team.get(col, default)
    try:
        return float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else default
    except (ValueError, TypeError):
        return default


def _play_game(t1, t2, round_num: int = 1) -> dict:
    """
    Resolve a single game between two teams.
    Favorite determined by AdjEM, with Seed as tiebreaker.
    """
    t1_em = _get_team_metric(t1, "AdjEM", 0)
    t2_em = _get_team_metric(t2, "AdjEM", 0)

    if t1_em > t2_em or (t1_em == t2_em and _get_team_metric(t1, "Seed", 16) < _get_team_metric(t2, "Seed", 16)):
        fav, dog = t1, t2
    else:
        fav, dog = t2, t1

    up = upset_probability(
        fav_seed=int(_get_team_metric(fav, "Seed", 1)),
        dog_seed=int(_get_team_metric(dog, "Seed", 16)),
        fav_adj_oe=_get_team_metric(fav, "AdjOE", 110),
        dog_adj_oe=_get_team_metric(dog, "AdjOE", 100),
        fav_adj_de=_get_team_metric(fav, "AdjDE", 95),
        dog_adj_de=_get_team_metric(dog, "AdjDE", 105),
        fav_sos=_get_team_metric(fav, "SOS", 0.55),
        dog_sos=_get_team_metric(dog, "SOS", 0.45),
        fav_continuity=_get_team_metric(fav, "Continuity", 70),
        dog_continuity=_get_team_metric(dog, "Continuity", 70),
        fav_tempo=_get_team_metric(fav, "Tempo", 68),
        dog_tempo=_get_team_metric(dog, "Tempo", 68),
        fav_tov=_get_team_metric(fav, "TOV%", 16),
        dog_tov=_get_team_metric(dog, "TOV%", 16),
        dog_three_var=_get_team_metric(dog, "3P_Var", 0.04),
        fav_ft_rate=_get_team_metric(fav, "FTRate", 0.30),
        dog_ft_rate=_get_team_metric(dog, "FTRate", 0.30),
        round_num=round_num,
    )

    return dog if random.random() < up["upset_prob"] else fav


def simulate_tournament(
    df: pd.DataFrame,
    n_sims: int = 1000,
    seed: int | None = None,
) -> dict:
    """
    Run n_sims full tournament simulations.

    Returns dict with:
      - "results": DataFrame with per-team probabilities for each round
      - "champion_counts": dict of team -> count
      - "confidence": dict of team -> (lower_95, upper_95) for championship prob
      - "n_sims": number of simulations run
    """
    if seed is not None:
        random.seed(seed)

    round_names = ["R64", "R32", "S16", "E8", "F4", "Champ"]
    team_names = df["Team"].tolist()

    # Initialize counters: team -> [R64_wins, R32_wins, S16_wins, E8_wins, F4_wins, Champ_wins]
    advancement = {name: [0] * 6 for name in team_names}

    for _ in range(n_sims):
        # Region phase: R64 -> E8
        region_winners = []

        for region in REGIONS:
            region_df = df[df["Region"] == region].sort_values("Seed")
            seeded = {int(row["Seed"]): row.to_dict() for _, row in region_df.iterrows()}

            # Build bracket tree in standard order
            bracket_order = [s for s1, s2 in BRACKET_MATCHUP_SEEDS for s in (s1, s2)]
            pool = [seeded[s] for s in bracket_order if s in seeded]

            round_num = 1  # R64
            while len(pool) > 1:
                next_pool = []
                for i in range(0, len(pool), 2):
                    if i + 1 >= len(pool):
                        next_pool.append(pool[i])
                    else:
                        winner = _play_game(pool[i], pool[i + 1], round_num=round_num)
                        round_idx = min(round_num - 1, 5)
                        advancement[winner["Team"]][round_idx] += 1
                        next_pool.append(winner)
                pool = next_pool
                round_num += 1

            if pool:
                region_winners.append(pool[0])

        # Final Four + Championship
        ff_pool = region_winners[:]
        round_num = 5  # Final Four

        while len(ff_pool) > 1:
            next_pool = []
            for i in range(0, len(ff_pool), 2):
                if i + 1 >= len(ff_pool):
                    next_pool.append(ff_pool[i])
                else:
                    winner = _play_game(ff_pool[i], ff_pool[i + 1], round_num=round_num)
                    round_idx = min(round_num - 1, 5)
                    advancement[winner["Team"]][round_idx] += 1
                    next_pool.append(winner)
            ff_pool = next_pool
            round_num += 1

    # Build results DataFrame
    df_result = df.copy()

    for i, rname in enumerate(round_names):
        df_result[f"{rname}_Prob"] = df_result["Team"].map(
            lambda t, idx=i: round(advancement.get(t, [0]*6)[idx] / n_sims, 4)
        )

    df_result["ChampionshipProb"] = df_result["Champ_Prob"]

    # Confidence intervals (Wilson score interval for binomial)
    confidence = {}
    for team_name in team_names:
        champ_count = advancement.get(team_name, [0]*6)[5]
        p_hat = champ_count / n_sims
        z = 1.96  # 95% CI

        if n_sims > 0:
            denom = 1 + z**2 / n_sims
            center = (p_hat + z**2 / (2 * n_sims)) / denom
            spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_sims)) / n_sims) / denom
            lower = max(0, center - spread)
            upper = min(1, center + spread)
        else:
            lower, upper = 0, 0

        confidence[team_name] = (round(lower, 4), round(upper, 4))

    df_result["CI_Lower"] = df_result["Team"].map(lambda t: confidence.get(t, (0,0))[0])
    df_result["CI_Upper"] = df_result["Team"].map(lambda t: confidence.get(t, (0,0))[1])

    df_result = df_result.sort_values("ChampionshipProb", ascending=False)

    return {
        "results": df_result,
        "advancement": advancement,
        "confidence": confidence,
        "n_sims": n_sims,
    }
