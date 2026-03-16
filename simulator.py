"""
simulator.py — March Madness Analyzer v3
==========================================
Monte Carlo tournament simulation using 9-feature logistic model.
"""

import random
import math
from collections import defaultdict
from config import REGIONS, BRACKET_MATCHUP_SEEDS
from upset import _matchup_from_rows


def _game_result(team_a, team_b, round_num=1):
    """Simulate a single game. Returns winner row."""
    seed_a = int(team_a.get("Seed", 8))
    seed_b = int(team_b.get("Seed", 8))

    if seed_a <= seed_b:
        fav, dog = team_a, team_b
    else:
        fav, dog = team_b, team_a

    m = _matchup_from_rows(fav, dog, round_num=round_num)
    upset_prob = m["upset_prob"]

    if random.random() < upset_prob:
        return dog
    return fav


def _simulate_region(teams_by_seed, round_num_start=1):
    """
    Simulate a single region from R64 through regional final.
    Returns list of advancing teams at each round + regional champion.
    """
    # R64 matchups
    matchup_order = BRACKET_MATCHUP_SEEDS
    current_round = []

    for s1, s2 in matchup_order:
        if s1 in teams_by_seed and s2 in teams_by_seed:
            winner = _game_result(teams_by_seed[s1], teams_by_seed[s2], round_num_start)
            current_round.append(winner)
        elif s1 in teams_by_seed:
            current_round.append(teams_by_seed[s1])
        elif s2 in teams_by_seed:
            current_round.append(teams_by_seed[s2])

    round_num = round_num_start + 1

    # Subsequent rounds until 1 remains
    while len(current_round) > 1:
        next_round = []
        for i in range(0, len(current_round), 2):
            if i + 1 < len(current_round):
                winner = _game_result(current_round[i], current_round[i + 1], round_num)
                next_round.append(winner)
            else:
                next_round.append(current_round[i])
        current_round = next_round
        round_num += 1

    return current_round[0] if current_round else None


def simulate_tournament(df, n_sims=2000):
    """
    Run n_sims Monte Carlo simulations of the full tournament.

    Returns dict with:
        - results: DataFrame sorted by championship probability
        - confidence: dict of team → (ci_lower, ci_upper) at 95%
    """
    # Build per-region seed maps
    region_teams = {}
    for region in REGIONS:
        reg_df = df[df["Region"] == region]
        region_teams[region] = {int(row["Seed"]): row for _, row in reg_df.iterrows()}

    # Tracking
    round_counts = defaultdict(lambda: defaultdict(int))  # team → round → count
    champ_counts = defaultdict(int)

    round_names = ["R64", "R32", "S16", "E8", "F4", "Champ"]

    for _ in range(n_sims):
        # All teams start in R64
        for region in REGIONS:
            for seed, team_row in region_teams[region].items():
                team = str(team_row.get("Team", ""))
                round_counts[team]["R64"] += 1

        # Simulate each region
        regional_champs = []
        for region in REGIONS:
            teams = region_teams[region]
            # R64
            matchups = BRACKET_MATCHUP_SEEDS
            r64_winners = []
            for s1, s2 in matchups:
                if s1 in teams and s2 in teams:
                    w = _game_result(teams[s1], teams[s2], 1)
                    r64_winners.append(w)
                    round_counts[str(w.get("Team", ""))]["R32"] += 1

            # R32
            r32_winners = []
            for i in range(0, len(r64_winners), 2):
                if i + 1 < len(r64_winners):
                    w = _game_result(r64_winners[i], r64_winners[i + 1], 2)
                    r32_winners.append(w)
                    round_counts[str(w.get("Team", ""))]["S16"] += 1

            # S16
            s16_winners = []
            for i in range(0, len(r32_winners), 2):
                if i + 1 < len(r32_winners):
                    w = _game_result(r32_winners[i], r32_winners[i + 1], 3)
                    s16_winners.append(w)
                    round_counts[str(w.get("Team", ""))]["E8"] += 1

            # E8 (regional final)
            if len(s16_winners) >= 2:
                champ = _game_result(s16_winners[0], s16_winners[1], 4)
                round_counts[str(champ.get("Team", ""))]["F4"] += 1
                regional_champs.append(champ)
            elif s16_winners:
                round_counts[str(s16_winners[0].get("Team", ""))]["F4"] += 1
                regional_champs.append(s16_winners[0])

        # Final Four
        if len(regional_champs) >= 4:
            # Semis: East vs Midwest, South vs West (standard bracket)
            semi1 = _game_result(regional_champs[0], regional_champs[2], 5)
            semi2 = _game_result(regional_champs[3], regional_champs[1], 5)
            # Championship
            champion = _game_result(semi1, semi2, 6)
            champ_name = str(champion.get("Team", ""))
            champ_counts[champ_name] += 1
            round_counts[champ_name]["Champ"] += 1

    # Build results DataFrame
    results = []
    for _, row in df.iterrows():
        team = str(row["Team"])
        entry = row.to_dict()
        for rnd in round_names:
            entry[f"{rnd}_Prob"] = round_counts[team].get(rnd, 0) / n_sims
        entry["ChampionshipProb"] = champ_counts.get(team, 0) / n_sims
        results.append(entry)

    results_df = pd.DataFrame(results).sort_values("ChampionshipProb", ascending=False)

    # 95% confidence intervals (Wilson score interval approximation)
    confidence = {}
    for team, count in champ_counts.items():
        p = count / n_sims
        z = 1.96
        n = n_sims
        denom = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denom
        spread = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
        confidence[team] = (max(0, center - spread), min(1, center + spread))

    # Add CI columns
    results_df["CI_Lower"] = results_df["Team"].apply(
        lambda t: confidence.get(t, (0, 0))[0])
    results_df["CI_Upper"] = results_df["Team"].apply(
        lambda t: confidence.get(t, (0, 0))[1])

    return {"results": results_df, "confidence": confidence}


# Need pandas for DataFrame
import pandas as pd
