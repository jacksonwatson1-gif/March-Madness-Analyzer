"""
optimizer.py — March Madness Analyzer v3
==========================================
EV-optimal bracket generator using scoring-system-aware expected value.
"""

import math
from config import REGIONS, BRACKET_MATCHUP_SEEDS, ROUND_POINTS, SEED_ROUND_WIN_RATES
from upset import _matchup_from_rows


def optimize_bracket(df, scoring="standard", strategy="max_ev", pool_size=5):
    """
    Generate an EV-optimal bracket.

    Parameters
    ----------
    df : DataFrame — Full tournament field
    scoring : str — Scoring system key
    strategy : str — 'max_ev', 'chalk', or 'contrarian'
    pool_size : int — Number of competitors in pool

    Returns
    -------
    dict with 'picks' (list of dicts) and 'total_ev' (float)
    """
    points = ROUND_POINTS.get(scoring, ROUND_POINTS["standard"])
    all_picks = []

    round_labels = ["R64", "R32", "S16", "E8", "F4", "Champ"]

    for region in REGIONS:
        reg_df = df[df["Region"] == region].sort_values("Seed")
        seeded = {int(row["Seed"]): row for _, row in reg_df.iterrows()}

        # R64 picks
        r64_winners = []
        for s1, s2 in BRACKET_MATCHUP_SEEDS:
            if s1 not in seeded or s2 not in seeded:
                continue
            fav, dog = seeded[s1], seeded[s2]
            m = _matchup_from_rows(fav, dog)

            fav_name = str(fav.get("Team", ""))
            dog_name = str(dog.get("Team", ""))
            fav_seed = int(fav.get("Seed", 1))
            dog_seed = int(dog.get("Seed", 16))

            fav_ev = m["fav_prob"] * points["R64"]
            dog_ev = m["upset_prob"] * points["R64"]

            # Contrarian bonus: scale by inverse popularity
            if strategy == "contrarian":
                dog_ev *= (1 + 0.15 * (dog_seed - fav_seed) / 15)
            elif strategy == "chalk":
                fav_ev *= 1.2  # Chalk bias

            if fav_ev >= dog_ev:
                pick = fav_name
                pick_seed = fav_seed
                pick_prob = m["fav_prob"]
                pick_ev = fav_ev
                winner_row = fav
            else:
                pick = dog_name
                pick_seed = dog_seed
                pick_prob = m["upset_prob"]
                pick_ev = dog_ev
                winner_row = dog

            all_picks.append({
                "Round": "R64", "Region": region, "Matchup": f"({s1})v({s2})",
                "Pick": pick, "Seed": pick_seed, "WinProb": pick_prob, "EV": pick_ev,
            })
            r64_winners.append(winner_row)

        # R32 through Regional Final
        current = r64_winners
        for r_idx, rnd in enumerate(["R32", "S16", "E8"], start=2):
            next_round = []
            for i in range(0, len(current), 2):
                if i + 1 >= len(current):
                    next_round.append(current[i])
                    continue
                a, b = current[i], current[i + 1]
                seed_a = int(a.get("Seed", 8))
                seed_b = int(b.get("Seed", 8))

                if seed_a <= seed_b:
                    fav, dog = a, b
                else:
                    fav, dog = b, a

                m = _matchup_from_rows(fav, dog, round_num=r_idx)
                fav_name = str(fav.get("Team", ""))
                dog_name = str(dog.get("Team", ""))

                # Cumulative win probability
                fav_cum = SEED_ROUND_WIN_RATES.get(int(fav.get("Seed", 1)), {}).get(rnd, 0.5)
                dog_cum = SEED_ROUND_WIN_RATES.get(int(dog.get("Seed", 16)), {}).get(rnd, 0.1)

                fav_ev = fav_cum * points[rnd]
                dog_ev = dog_cum * points[rnd]

                if strategy == "contrarian":
                    dog_ev *= 1.1
                elif strategy == "chalk":
                    fav_ev *= 1.1

                if fav_ev >= dog_ev:
                    winner = fav
                    pick_ev = fav_ev
                    pick_prob = fav_cum
                else:
                    winner = dog
                    pick_ev = dog_ev
                    pick_prob = dog_cum

                all_picks.append({
                    "Round": rnd, "Region": region,
                    "Matchup": f"{fav_name[:12]} v {dog_name[:12]}",
                    "Pick": str(winner.get("Team", "")),
                    "Seed": int(winner.get("Seed", 0)),
                    "WinProb": pick_prob, "EV": pick_ev,
                })
                next_round.append(winner)
            current = next_round

    # Final Four & Championship from regional winners
    regional_champs = []
    for region in REGIONS:
        region_picks = [p for p in all_picks if p["Region"] == region and p["Round"] == "E8"]
        if region_picks:
            champ_pick = region_picks[0]
            # Find the team row
            team_row = df[df["Team"] == champ_pick["Pick"]]
            if not team_row.empty:
                regional_champs.append(team_row.iloc[0])

    if len(regional_champs) >= 4:
        # Semi 1: East vs Midwest
        for i, (a_idx, b_idx) in enumerate([(0, 2), (3, 1)]):
            a, b = regional_champs[a_idx], regional_champs[b_idx]
            seed_a, seed_b = int(a.get("Seed", 1)), int(b.get("Seed", 1))
            a_cum = SEED_ROUND_WIN_RATES.get(seed_a, {}).get("F4", 0.1)
            b_cum = SEED_ROUND_WIN_RATES.get(seed_b, {}).get("F4", 0.1)

            a_ev = a_cum * points["F4"]
            b_ev = b_cum * points["F4"]

            if a_ev >= b_ev:
                winner = a
                pick_ev, pick_prob = a_ev, a_cum
            else:
                winner = b
                pick_ev, pick_prob = b_ev, b_cum

            all_picks.append({
                "Round": "F4", "Region": "Final Four",
                "Matchup": f"{a.get('Team','')} v {b.get('Team','')}",
                "Pick": str(winner.get("Team", "")),
                "Seed": int(winner.get("Seed", 0)),
                "WinProb": pick_prob, "EV": pick_ev,
            })

    # Championship pick (top EV among F4 picks)
    f4_picks = [p for p in all_picks if p["Round"] == "F4"]
    if f4_picks:
        best = max(f4_picks, key=lambda x: x["EV"])
        seed = best["Seed"]
        champ_prob = SEED_ROUND_WIN_RATES.get(seed, {}).get("Champ", 0.05)
        all_picks.append({
            "Round": "Champ", "Region": "Championship",
            "Matchup": "National Championship",
            "Pick": best["Pick"],
            "Seed": best["Seed"],
            "WinProb": champ_prob,
            "EV": champ_prob * points["Champ"],
        })

    total_ev = sum(p["EV"] for p in all_picks)

    return {"picks": all_picks, "total_ev": total_ev}
