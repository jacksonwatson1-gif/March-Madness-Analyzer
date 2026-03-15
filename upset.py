"""
models/upset.py — Logistic upset probability model.

Key improvements over v1:
  1. Coefficients fitted via logistic regression on historical tournament data.
  2. Expanded feature set: AdjOE, AdjDE (no AdjEM — eliminates collinearity),
     tempo mismatch, SOS, continuity, TOV%, 3P variance, FT rate.
  3. Round-specific intercept adjustment.
  4. Probability clamp widened to [0.005, 0.95].
  5. Brier score and calibration reporting.
"""

import math
import numpy as np
import pandas as pd

from config import HISTORICAL_UPSET_RATES


_DEFAULT_COEFFICIENTS = {
    "intercept":        -0.15,
    "adj_oe_diff":      -0.055,
    "adj_de_diff":      -0.045,
    "tempo_mismatch":    0.035,
    "sos_diff":         -1.80,
    "continuity_diff":   0.015,
    "tov_rate_diff":    -0.020,
    "three_pt_var":      2.50,
    "ft_rate_diff":     -1.20,
    "round_adj":         0.025,
}

_fitted_coefficients = dict(_DEFAULT_COEFFICIENTS)
_model_fitted = False


def fit_model() -> dict:
    global _fitted_coefficients, _model_fitted
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from historical_data import get_historical_matchups

        df = get_historical_matchups()
        feature_cols = [
            "adj_oe_diff", "adj_de_diff", "tempo_mismatch",
            "sos_diff", "continuity_diff", "tov_rate_diff",
            "three_pt_var", "ft_rate_diff", "round_num",
        ]
        X = df[feature_cols].values
        y = df["upset"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
        model.fit(X_scaled, y)
        coefs = model.coef_[0]
        scales = scaler.scale_
        _fitted_coefficients = {
            "intercept":        float(model.intercept_[0]),
            "adj_oe_diff":      float(coefs[0] / scales[0]),
            "adj_de_diff":      float(coefs[1] / scales[1]),
            "tempo_mismatch":   float(coefs[2] / scales[2]),
            "sos_diff":         float(coefs[3] / scales[3]),
            "continuity_diff":  float(coefs[4] / scales[4]),
            "tov_rate_diff":    float(coefs[5] / scales[5]),
            "three_pt_var":     float(coefs[6] / scales[6]),
            "ft_rate_diff":     float(coefs[7] / scales[7]),
            "round_adj":        float(coefs[8] / scales[8]),
        }
        _model_fitted = True
        y_pred = model.predict_proba(X_scaled)[:, 1]
        brier = float(np.mean((y_pred - y) ** 2))
        return {
            "coefficients": dict(_fitted_coefficients),
            "brier_score": round(brier, 4),
            "n_samples": len(df),
            "accuracy": float(model.score(X_scaled, y)),
            "fitted": True,
        }
    except ImportError:
        _model_fitted = False
        return {
            "coefficients": dict(_DEFAULT_COEFFICIENTS),
            "brier_score": None,
            "n_samples": 0,
            "accuracy": None,
            "fitted": False,
        }


def get_model_info() -> dict:
    return {
        "coefficients": dict(_fitted_coefficients),
        "fitted": _model_fitted,
    }


def upset_probability(
    fav_seed: int, dog_seed: int,
    fav_adj_oe: float = 110.0, dog_adj_oe: float = 100.0,
    fav_adj_de: float = 95.0, dog_adj_de: float = 105.0,
    fav_sos: float = 0.55, dog_sos: float = 0.45,
    fav_continuity: float = 70.0, dog_continuity: float = 70.0,
    fav_tempo: float = 68.0, dog_tempo: float = 68.0,
    fav_tov: float = 16.0, dog_tov: float = 16.0,
    dog_three_var: float = 0.04,
    fav_ft_rate: float = 0.30, dog_ft_rate: float = 0.30,
    round_num: int = 1,
) -> dict:
    c = _fitted_coefficients
    matchup = (min(fav_seed, dog_seed), max(fav_seed, dog_seed))
    base_rate = HISTORICAL_UPSET_RATES.get(matchup, 0.35)
    adj_oe_diff = fav_adj_oe - dog_adj_oe
    adj_de_diff = dog_adj_de - fav_adj_de
    tempo_mismatch = abs(fav_tempo - dog_tempo)
    sos_diff = fav_sos - dog_sos
    continuity_diff = dog_continuity - fav_continuity
    tov_rate_diff = dog_tov - fav_tov
    three_pt_var = dog_three_var
    ft_rate_diff = fav_ft_rate - dog_ft_rate
    logit_base = math.log(max(base_rate, 1e-6) / max(1 - base_rate, 1e-6))
    model_adj = (
        c["adj_oe_diff"]     * adj_oe_diff
        + c["adj_de_diff"]     * adj_de_diff
        + c["tempo_mismatch"]  * tempo_mismatch
        + c["sos_diff"]        * sos_diff
        + c["continuity_diff"] * continuity_diff
        + c["tov_rate_diff"]   * tov_rate_diff
        + c["three_pt_var"]    * three_pt_var
        + c["ft_rate_diff"]    * ft_rate_diff
        + c["round_adj"]       * round_num
    )
    logit_adj = logit_base * 0.70 + model_adj * 0.30
    prob_upset = 1 / (1 + math.exp(-logit_adj))
    prob_upset = max(0.005, min(0.95, prob_upset))
    if prob_upset > 0.45:
        severity = "HIGH"
    elif prob_upset > 0.25:
        severity = "MEDIUM"
    else:
        severity = "LOW"
    return {
        "upset_prob":        round(prob_upset, 4),
        "fav_prob":          round(1 - prob_upset, 4),
        "severity":          severity,
        "seed_diff":         dog_seed - fav_seed,
        "base_rate":         base_rate,
        "adj_oe_edge":       round(adj_oe_diff, 2),
        "adj_de_edge":       round(adj_de_diff, 2),
        "tempo_mismatch":    round(tempo_mismatch, 1),
        "sos_edge":          round(sos_diff, 3),
        "continuity_edge":   round(continuity_diff, 1),
        "tov_edge":          round(tov_rate_diff, 1),
        "three_pt_var":      round(three_pt_var, 4),
        "ft_rate_edge":      round(ft_rate_diff, 3),
        "round_num":         round_num,
    }


def compute_all_first_round(df: pd.DataFrame) -> list[dict]:
    results = []
    for region in ["East", "West", "South", "Midwest"]:
        reg_df = df[df["Region"] == region].sort_values("Seed")
        seeded = {int(row["Seed"]): row for _, row in reg_df.iterrows()}
        pairs = [(1,16),(2,15),(3,14),(4,13),(5,12),(6,11),(7,10),(8,9)]
        for s1, s2 in pairs:
            if s1 in seeded and s2 in seeded:
                fav = seeded[s1]
                dog = seeded[s2]
                up = _matchup_from_rows(fav, dog)
                results.append({
                    "Region": region, "Matchup": f"({s1}) vs ({s2})",
                    "Favorite": fav["Team"], "Underdog": dog["Team"], **up,
                })
    return results


def _matchup_from_rows(fav, dog, round_num: int = 1) -> dict:
    return upset_probability(
        fav_seed=int(fav["Seed"]), dog_seed=int(dog["Seed"]),
        fav_adj_oe=float(fav.get("AdjOE", 110)), dog_adj_oe=float(dog.get("AdjOE", 100)),
        fav_adj_de=float(fav.get("AdjDE", 95)), dog_adj_de=float(dog.get("AdjDE", 105)),
        fav_sos=float(fav.get("SOS", 0.55)), dog_sos=float(dog.get("SOS", 0.45)),
        fav_continuity=float(fav.get("Continuity", 70)), dog_continuity=float(dog.get("Continuity", 70)),
        fav_tempo=float(fav.get("Tempo", 68)), dog_tempo=float(dog.get("Tempo", 68)),
        fav_tov=float(fav.get("TOV%", 16)), dog_tov=float(dog.get("TOV%", 16)),
        dog_three_var=float(dog.get("3P_Var", 0.04)),
        fav_ft_rate=float(fav.get("FTRate", 0.30)), dog_ft_rate=float(dog.get("FTRate", 0.30)),
        round_num=round_num,
    )
