# goals_pipelines.py
# -*- coding: utf-8 -*-
"""
Encapsulated Over/Under 2.5 pipelines + Second-Half-Goal-by-HT-scoreline scorer.
Public API:
  - run_u25(end_period)
  - run_o25(end_period)
  - run_2h_htscore(end_period, model_dir=..., market_map=...)
"""

# ── Shared imports ────────────────────────────────────────────────────────
import os
import glob
from datetime import datetime, date
from typing import Dict, Any, Iterable, Tuple, Optional
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
from joblib import load

# Optional dependency (team name standardiser)
try:
    import function_library as _fl  # must provide _fl.team_name_map(df)
except Exception:
    _fl = None

# ── CONFIG (edit your local paths if needed) ─────────────────────────────
FILE_PATH      = r"C:\Users\leere\OneDrive\Desktop\RAW DATA\ml_goals.xls"
MODEL_DIR_O25  = r"C:\Users\leere\PycharmProjects\Football_ML3\Goals\Over_2_5\model_file"
MODEL_DIR_U25  = r"C:\Users\leere\PycharmProjects\Football_ML3\Goals\Under_2_5\model_file"
IMPORT_DIR     = r"C:\Users\leere\OneDrive\Desktop\IMPORTS"

# Default dir for the “2H goal by HT scoreline” models (can be overridden)
MODEL_DIR_2H_HTSCORE = r"C:\Users\leere\PycharmProjects\Football_ML3\Goals\2H_goal\ht_scoreline\path_ht_score"

# Optional map from HT scoreline → total-goals line (used for labelling)
DEFAULT_MARKET_MAP = {
    "0-0": 0.5, "0-1": 1.5, "1-0": 1.5, "1-1": 2.5,
    "0-2": 2.5, "2-0": 2.5, "2-1": 3.5, "1-2": 3.5, "3-0": 3.5,
}

# ── COLUMN MAP (shared) ─────────────────────────────────────────────────
_COLUMN_DICT = {
    "country": "country",
    "league": "league",
    "sezonul": "season",
    "datameci": "date",
    "orameci": "ko_time",
    "etapa": "round",
    "txtechipa1": "home_team",
    "txtechipa2": "away_team",
    "scor1": "home_goals_ft",
    "scor2": "away_goals_ft",
    "scorp1": "home_goals_ht",
    "scorp2": "away_goals_ht",
    "place1": "home_team_place_total",
    "place1a": "home_team_place_home",
    "place2": "away_team_place_total",
    "place2d": "away_team_place_away",
    "cotaa": "home_odds",
    "cotae": "draw_odds",
    "cotad": "away_odds",
    "cotao": "over_25_odds",
    "cotau": "under_25_odds",
    "elohomeo": "elo_home",
    "eloawayo": "elo_away",
    "formah": "form_home",
    "formaa": "form_away",
    "suth": "shots_home",
    "suth1": "shots_home_1h",
    "suth2": "shots_home_2h",
    "suta": "shots_away",
    "suta1": "shots_away_1h",
    "suta2": "shots_away_2h",
    "sutht": "shots_on_target_home",
    "sutht1": "shots_on_target_home_1h",
    "sutht2": "shots_on_target_home_2h",
    "sutat": "shots_on_target_away",
    "sutat1": "shots_on_target_away_1h",
    "sutat2": "shots_on_target_away_2h",
    "corh": "corners_home",
    "corh1": "corners_home_1h",
    "corh2": "corners_home_2h",
    "cora": "corners_away",
    "cora1": "corners_away_1h",
    "cora2": "corners_away_2h",
    "foulsh": "fouls_home",
    "foulsh1": "fouls_home_1h",
    "foulsh2": "fouls_home_2h",
    "foulsa": "fouls_away",
    "foulsa1": "fouls_away_1h",
    "foulsa2": "fouls_away_2h",
    "yellowh": "yellow_cards_home",
    "yellowh1": "yellow_cards_home_1h",
    "yellowh2": "yellow_cards_home_2h",
    "yellowa": "yellow_cards_away",
    "yellowa1": "yellow_cards_away_1h",
    "yellowa2": "yellow_cards_away_2h",
    "ballph": "possession_home",
    "ballph1": "possession_home_1h",
    "ballph2": "possession_home_2h",
    "ballpa": "possession_away",
    "ballpa1": "possession_away_1h",
    "ballpa2": "possession_away_2h",
    "gsh": "goals_scored_total_home",
    "gch": "goals_conceded_total_home",
    "gsa": "goals_scored_total_away",
    "gca": "goals_conceded_total_away",
}

# ── Shared helpers ───────────────────────────────────────────────────────
def _round_half_up_2(x) -> str:
    """Exact two-decimal string with ROUND_HALF_UP (Excel-style)."""
    return str(Decimal(str(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _align_features_numeric_fill(df_in: pd.DataFrame, feature_contract: Iterable[str]) -> pd.DataFrame:
    """
    Reindex to training features (missing→0), coerce object→numeric (errors→NaN), fill NaN with 0.
    Keeps row order and index intact; does not drop rows.
    """
    X = df_in.reindex(columns=list(feature_contract), fill_value=0)
    for c in X.columns:
        if pd.api.types.is_object_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.fillna(0)

def _parse_end_period(ep) -> date:
    if isinstance(ep, date):
        return ep
    return pd.to_datetime(ep).date()

def _newest_pkl(model_dir: str) -> str:
    pkls = glob.glob(os.path.join(model_dir, "best_model_*_calibrated_*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No model PKLs found in: {model_dir}")
    return max(pkls, key=os.path.getmtime)

def _newest_pkls_by_htscore(model_dir: str) -> dict[str, str]:
    """Return {ht_score: path_to_newest_pkl} using md['ht_score'] from saved PKLs."""
    picks: dict[str, str] = {}
    for pkl in glob.glob(os.path.join(model_dir, "best_model_*_calibrated_*.pkl")):
        try:
            md = load(pkl)
        except Exception:
            continue
        ht = str(md.get('ht_score', '')).strip()
        if not ht:
            continue
        if (ht not in picks) or (os.path.getmtime(pkl) > os.path.getmtime(picks[ht])):
            picks[ht] = pkl
    return picks

def _align_features(df_in: pd.DataFrame, feature_contract: Iterable[str]) -> pd.DataFrame:
    X = df_in.reindex(columns=list(feature_contract), fill_value=0)
    for c in X.columns:
        if pd.api.types.is_object_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="ignore")
    return X

def _compute_slope(x: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    xs = np.arange(len(x))
    return np.polyfit(xs, x, 1)[0]

def _compute_rolling_features_metric(
    df_sub: pd.DataFrame, full_col: str, first_half_col: str, prefix: str,
    window_long: int = 5, window_short: int = 3
) -> pd.DataFrame:
    out = {}
    # Full match
    roll_long  = df_sub[full_col].rolling(window_long, min_periods=1).mean().shift(1)
    roll_std   = df_sub[full_col].rolling(window_long, min_periods=1).std().shift(1)
    roll_short = df_sub[full_col].rolling(window_short, min_periods=1).mean().shift(1)
    trend      = df_sub[full_col].rolling(window_long, min_periods=2).apply(_compute_slope, raw=True).shift(1)
    out[f'{prefix}_Rolling_{full_col}_Mean'] = roll_long
    out[f'{prefix}_Rolling_{full_col}_Std'] = roll_std
    out[f'{prefix}_Rolling_{full_col}_Mean_Short'] = roll_short
    out[f'{prefix}_Momentum_{full_col}'] = roll_short - roll_long
    out[f'{prefix}_Trend_Slope_{full_col}'] = trend
    # First half
    fh_roll_long  = df_sub[first_half_col].rolling(window_long, min_periods=1).mean().shift(1)
    fh_roll_std   = df_sub[first_half_col].rolling(window_long, min_periods=1).std().shift(1)
    fh_roll_short = df_sub[first_half_col].rolling(window_short, min_periods=1).mean().shift(1)
    fh_trend      = df_sub[first_half_col].rolling(window_long, min_periods=2).apply(_compute_slope, raw=True).shift(1)
    out[f'{prefix}_Rolling_{first_half_col}_Mean'] = fh_roll_long
    out[f'{prefix}_Rolling_{first_half_col}_Std'] = fh_roll_std
    out[f'{prefix}_Rolling_{first_half_col}_Mean_Short'] = fh_roll_short
    out[f'{prefix}_Momentum_{first_half_col}'] = fh_roll_short - fh_roll_long
    out[f'{prefix}_Trend_Slope_{first_half_col}'] = fh_trend
    return pd.DataFrame(out, index=df_sub.index)

def _add_rolling_features_split(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values('date').reset_index(drop=True)

    overall = pd.concat([
        _compute_rolling_features_metric(group, 'GoalsScored', 'FirstHalfGoalsScored', 'Overall'),
        _compute_rolling_features_metric(group, 'Shots', 'Shots_1h', 'Overall'),
        _compute_rolling_features_metric(group, 'Corners', 'Corners_1h', 'Overall'),
        _compute_rolling_features_metric(group, 'ShotsOnTarget', 'ShotsOnTarget_1h', 'Overall'),
    ], axis=1)
    group = pd.concat([group, overall], axis=1)

    home_mask = group['is_home'] == 1
    away_mask = ~home_mask

    if home_mask.any():
        home_feats = pd.concat([
            _compute_rolling_features_metric(group.loc[home_mask], 'GoalsScored', 'FirstHalfGoalsScored', 'Home'),
            _compute_rolling_features_metric(group.loc[home_mask], 'Shots', 'Shots_1h', 'Home'),
            _compute_rolling_features_metric(group.loc[home_mask], 'Corners', 'Corners_1h', 'Home'),
            _compute_rolling_features_metric(group.loc[home_mask], 'ShotsOnTarget', 'ShotsOnTarget_1h', 'Home'),
        ], axis=1)
        group.loc[home_mask, home_feats.columns] = home_feats

    if away_mask.any():
        away_feats = pd.concat([
            _compute_rolling_features_metric(group.loc[away_mask], 'GoalsScored', 'FirstHalfGoalsScored', 'Away'),
            _compute_rolling_features_metric(group.loc[away_mask], 'Shots', 'Shots_1h', 'Away'),
            _compute_rolling_features_metric(group.loc[away_mask], 'Corners', 'Corners_1h', 'Away'),
            _compute_rolling_features_metric(group.loc[away_mask], 'ShotsOnTarget', 'ShotsOnTarget_1h', 'Away'),
        ], axis=1)
        group.loc[away_mask, away_feats.columns] = away_feats

    # Goals thresholds
    thresh = {}
    for t in [1.5, 2.5, 3.5]:
        s = group['GoalsScored'].gt(t).shift(1)
        thresh[f'Overall_Percent_Over_{t}'] = s.expanding(1).mean()
        thresh[f'Overall_Rolling5_Percent_Over_{t}'] = s.rolling(5, 1).mean()
        if home_mask.any():
            h = group.loc[home_mask, 'GoalsScored'].gt(t).shift(1)
            thresh[f'Home_Percent_Over_{t}'] = h.expanding(1).mean()
            thresh[f'Home_Rolling5_Percent_Over_{t}'] = h.rolling(5, 1).mean()
        if away_mask.any():
            a = group.loc[away_mask, 'GoalsScored'].gt(t).shift(1)
            thresh[f'Away_Percent_Over_{t}'] = a.expanding(1).mean()
            thresh[f'Away_Rolling5_Percent_Over_{t}'] = a.rolling(5, 1).mean()
    group = pd.concat([group, pd.DataFrame(thresh, index=group.index)], axis=1)

    # Team outcome %
    outcome = {}
    for t in [0.5, 1.5, 2.5, 3.5]:
        outcome[f'TeamPct_Over_{t}'] = group['GoalsScored'].gt(t).shift(1).expanding(1).mean()
        if home_mask.any():
            outcome[f'Home_TeamPct_Over_{t}'] = group.loc[home_mask, 'GoalsScored'].gt(t).shift(1).expanding(1).mean()
        if away_mask.any():
            outcome[f'Away_TeamPct_Over_{t}'] = group.loc[away_mask, 'GoalsScored'].gt(t).shift(1).expanding(1).mean()
    group = pd.concat([group, pd.DataFrame(outcome, index=group.index)], axis=1)

    # Corners outcome %
    corners = {}
    for t in [3.5, 4.5, 5.5, 6.5]:
        s = group['Corners'].gt(t).shift(1)
        corners[f'CornersPct_Over_{t}'] = s.expanding(1).mean()
        corners[f'CornersRolling5Pct_Over_{t}'] = s.rolling(5, 1).mean()
        if home_mask.any():
            h = group.loc[home_mask, 'Corners'].gt(t).shift(1)
            corners[f'Home_CornersPct_Over_{t}'] = h.expanding(1).mean()
            corners[f'Home_CornersRolling5Pct_Over_{t}'] = h.rolling(5, 1).mean()
        if away_mask.any():
            a = group.loc[away_mask, 'Corners'].gt(t).shift(1)
            corners[f'Away_CornersPct_Over_{t}'] = a.expanding(1).mean()
            corners[f'Away_CornersRolling5Pct_Over_{t}'] = a.rolling(5, 1).mean()
    group = pd.concat([group, pd.DataFrame(corners, index=group.index)], axis=1)

    return group

def _load_and_prepare_raw(file_path: str) -> pd.DataFrame:
    print(f"[1/9] Loading raw data: {file_path}")
    df = pd.read_excel(file_path)
    df = df.rename(columns=_COLUMN_DICT).filter(items=_COLUMN_DICT.values())
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y", errors='coerce')
    df = df.sort_values('date')
    return df

def _clip_to_period(df: pd.DataFrame, end_period: date) -> pd.DataFrame:
    today = datetime.today().date()
    before_end = df[df['date'].dt.date <= end_period].copy()
    played_mask = before_end['date'].dt.date < today
    before_end.loc[played_mask, 'points_home'] = np.where(
        before_end.loc[played_mask, 'home_goals_ft'] > before_end.loc[played_mask, 'away_goals_ft'], 3,
        np.where(before_end.loc[played_mask, 'home_goals_ft'] == before_end.loc[played_mask, 'away_goals_ft'], 1, 0)
    )
    before_end.loc[played_mask, 'points_away'] = np.where(
        before_end.loc[played_mask, 'away_goals_ft'] > before_end.loc[played_mask, 'home_goals_ft'], 3,
        np.where(before_end.loc[played_mask, 'away_goals_ft'] == before_end.loc[played_mask, 'home_goals_ft'], 1, 0)
    )
    print(f"[2/9] Clipped to end_period={end_period} | rows={len(before_end)}")
    return before_end

def _build_team_frame(data: pd.DataFrame) -> pd.DataFrame:
    home_df = data[['country', 'season', 'date', 'home_team', 'away_team',
                    'home_goals_ft', 'away_goals_ft', 'home_goals_ht', 'away_goals_ht',
                    'shots_home', 'shots_home_1h', 'shots_home_2h',
                    'shots_on_target_home', 'shots_on_target_home_1h', 'shots_on_target_home_2h',
                    'corners_home', 'corners_home_1h', 'corners_home_2h']].copy()
    home_df.rename(columns={
        'home_team': 'Team', 'away_team': 'Opponent',
        'home_goals_ft': 'GoalsScored', 'away_goals_ft': 'GoalsConceded',
        'home_goals_ht': 'FirstHalfGoalsScored', 'away_goals_ht': 'FirstHalfGoalsConceded',
        'shots_home': 'Shots', 'shots_home_1h': 'Shots_1h', 'shots_home_2h': 'Shots_2h',
        'shots_on_target_home': 'ShotsOnTarget', 'shots_on_target_home_1h': 'ShotsOnTarget_1h',
        'shots_on_target_home_2h': 'ShotsOnTarget_2h',
        'corners_home': 'Corners', 'corners_home_1h': 'Corners_1h', 'corners_home_2h': 'Corners_2h',
    }, inplace=True)
    home_df['is_home'] = 1

    away_df = data[['country', 'season', 'date', 'away_team', 'home_team',
                    'away_goals_ft', 'home_goals_ft', 'away_goals_ht', 'home_goals_ht',
                    'shots_away', 'shots_away_1h', 'shots_away_2h',
                    'shots_on_target_away', 'shots_on_target_away_1h', 'shots_on_target_away_2h',
                    'corners_away', 'corners_away_1h', 'corners_away_2h']].copy()
    away_df.rename(columns={
        'away_team': 'Team', 'home_team': 'Opponent',
        'away_goals_ft': 'GoalsScored', 'home_goals_ft': 'GoalsConceded',
        'away_goals_ht': 'FirstHalfGoalsScored', 'home_goals_ht': 'FirstHalfGoalsConceded',
        'shots_away': 'Shots', 'shots_away_1h': 'Shots_1h', 'shots_away_2h': 'Shots_2h',
        'shots_on_target_away': 'ShotsOnTarget', 'shots_on_target_away_1h': 'ShotsOnTarget_1h',
        'shots_on_target_away_2h': 'ShotsOnTarget_2h',
        'corners_away': 'Corners', 'corners_away_1h': 'Corners_1h', 'corners_away_2h': 'Corners_2h',
    }, inplace=True)
    away_df['is_home'] = 0

    team_df = pd.concat([home_df, away_df], ignore_index=True)
    team_df.sort_values(['country', 'season', 'Team', 'date'], inplace=True)
    print(f"[3/9] Built team-frame | rows={len(team_df)}")
    return team_df

def _enrich_team_features(team_df: pd.DataFrame) -> pd.DataFrame:
    team_df = team_df.groupby(['country', 'season', 'Team'], group_keys=False).apply(_add_rolling_features_split)
    team_df = team_df.copy()
    print(f"[4/9] Added rolling features | cols={team_df.shape[1]}")
    return team_df

def _add_corners_outcomes_to_team(team_df: pd.DataFrame, match_data: pd.DataFrame) -> pd.DataFrame:
    md = match_data.copy()
    md['Total_Corners'] = md['corners_home'] + md['corners_away']
    home_m = md[['country', 'season', 'date', 'home_team', 'Total_Corners']].rename(columns={'home_team': 'Team'})
    away_m = md[['country', 'season', 'date', 'away_team', 'Total_Corners']].rename(columns={'away_team': 'Team'})
    tcm = pd.concat([home_m, away_m], ignore_index=True).sort_values(['country', 'season', 'Team', 'date'])

    for thr in [9.5, 10.5, 11.5]:
        ind = f'Over_{thr}'
        tcm[ind] = (tcm['Total_Corners'] > thr).astype(int)
        tcm[f'SeasonPct_{ind}']   = tcm.groupby(['country', 'season', 'Team'])[ind].transform(lambda x: x.shift(1).expanding(1).mean())
        tcm[f'Rolling5Pct_{ind}'] = tcm.groupby(['country', 'season', 'Team'])[ind].transform(lambda x: x.shift(1).rolling(5, 1).mean())

    cols = ['country', 'season', 'date', 'Team',
            'SeasonPct_Over_9.5', 'Rolling5Pct_Over_9.5',
            'SeasonPct_Over_10.5', 'Rolling5Pct_Over_10.5',
            'SeasonPct_Over_11.5', 'Rolling5Pct_Over_11.5']
    out = team_df.merge(tcm[cols], on=['country', 'season', 'date', 'Team'], how='left')
    print(f"[5/9] Merged corners outcomes")
    return out

def _prepare_match_level(team_df: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    # home subset
    home_subset = team_df[team_df['is_home'] == 1].drop(columns=['Opponent']).rename(columns={'Team': 'home_team'})
    home_key = ['country', 'season', 'date', 'home_team', 'is_home']
    home_feats = [c for c in home_subset.columns if c not in home_key and (
        c.startswith("Overall_") or c.startswith("Home_") or c.startswith("SeasonPct_Over_") or c.startswith("Rolling5Pct_Over_"))]
    home_features = home_subset[home_key + home_feats].copy()
    home_features.rename(columns={c: ("home_" + (c[len("Home_"):] if c.startswith("Home_") else c)) for c in home_feats}, inplace=True)

    # away subset
    away_subset = team_df[team_df['is_home'] == 0].drop(columns=['Opponent']).rename(columns={'Team': 'away_team'})
    away_key = ['country', 'season', 'date', 'away_team', 'is_home']
    away_feats = [c for c in away_subset.columns if c not in away_key and (
        c.startswith("Overall_") or c.startswith("Away_") or c.startswith("SeasonPct_Over_") or c.startswith("Rolling5Pct_Over_"))]
    away_features = away_subset[away_key + away_feats].copy()
    away_features.rename(columns={c: ("away_" + (c[len("Away_"):] if c.startswith("Away_") else c)) for c in away_feats}, inplace=True)

    match_df = original.copy()
    match_df = match_df.merge(home_features, on=['country', 'season', 'date', 'home_team'], how='left')
    match_df = match_df.merge(away_features, on=['country', 'season', 'date', 'away_team'], how='left')
    print(f"[6/9] Built match-level feature table | cols={match_df.shape[1]}")
    return match_df

def _standardise_fixture_names(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    out = fixtures_df.copy()
    if _fl and hasattr(_fl, "team_name_map"):
        out = _fl.team_name_map(out)
    # Specific Chile mapping
    mask = (out['country'] == 'Chl1')
    out.loc[mask, ['home_team', 'away_team']] = (
        out.loc[mask, ['home_team', 'away_team']].replace('Everton', 'Everton De Vina')
    )
    return out

def _load_model(model_dir: str) -> Dict[str, Any]:
    pkl_path = _newest_pkl(model_dir)
    md = load(pkl_path)
    model = md['model']
    threshold = float(md.get('threshold', 0.5))
    feat_list = list(md['features'])
    print(f"[7/9] Loaded model:\n       {os.path.basename(pkl_path)}\n       threshold={threshold}")
    return {"path": pkl_path, "model": model, "threshold": threshold, "features": feat_list}

def _score_fixtures(fixtures: pd.DataFrame, model_bundle: Dict[str, Any], proba_col: str) -> pd.DataFrame:
    model   = model_bundle["model"]
    feats   = model_bundle["features"]

    if "country" in fixtures.columns and any(f.startswith("country_") for f in feats):
        fixtures = pd.get_dummies(fixtures, columns=["country"], prefix="country")

    X = _align_features(fixtures, feats)
    X = X.dropna()
    scored = fixtures.loc[X.index].copy()

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = proba[:, 1] if proba.ndim == 2 else proba
    else:
        proba = model.predict(X).astype(float)

    scored[proba_col] = proba
    return scored

def _write_outputs(
    scored: pd.DataFrame,
    threshold: float,
    import_dir: str,
    model_path: str,
    proba_col: str,
    file_tag: str
) -> Tuple[Dict[str, str], int]:
    """
    Writes ONLY the positives CSV with the four required columns:
    EventName, Provider, MarketName, SelectionName.
    """
    os.makedirs(import_dir, exist_ok=True)
    out_pos = os.path.join(import_dir, f"{file_tag}_predictions.csv")

    # Build EventName and bet flag
    if {"home_team", "away_team"}.issubset(scored.columns):
        scored["EventName"] = scored["home_team"] + " v " + scored["away_team"]
    scored["bet"] = scored[proba_col] >= threshold

    # Keep only positives and export the four required columns
    positives = scored.loc[scored["bet"]].copy()

    # Ensure the three label columns exist (they’re set in _run_pipeline via `provider=...`)
    for c in ["Provider", "MarketName", "SelectionName"]:
        if c not in positives.columns:
            positives[c] = ""

    out_df = positives[["EventName", "Provider", "MarketName", "SelectionName"]].copy()
    out_df.to_csv(out_pos, index=False)

    print(f"[9/9] Wrote positives only (4 columns): {out_pos}")
    print(f"      Model file   → {model_path}")
    print(f"      Positive selections: {len(out_df)}")

    return {"positives": out_pos}, len(out_df)


# ── Single private runner used by U25/O25 ────────────────────────────────
def _run_pipeline(
    end_period: Any,
    *,
    file_path: str,
    model_dir: str,
    import_dir: str,
    proba_col: str,
    provider: Tuple[str, str, str] | None,
    file_tag: str
) -> Dict[str, Any]:
    end_date = _parse_end_period(end_period)
    today = datetime.today().date()

    raw = _load_and_prepare_raw(file_path)
    data = _clip_to_period(raw, end_date)

    team_df = _build_team_frame(data)
    team_df = _enrich_team_features(team_df)
    team_df = _add_corners_outcomes_to_team(team_df, data)

    match_df = _prepare_match_level(team_df, data)

    fixtures = match_df[(match_df['date'].dt.date >= today) & (match_df['date'].dt.date <= end_date)].copy()
    print(f"[7a/9] Fixtures in window [{today} → {end_date}]: {len(fixtures)}")

    fixtures = _standardise_fixture_names(fixtures)

    mb = _load_model(model_dir)
    scored = _score_fixtures(fixtures, mb, proba_col=proba_col)

    if provider:
        scored["Provider"], scored["MarketName"], scored["SelectionName"] = provider

    paths, n_pos = _write_outputs(scored, mb["threshold"], import_dir, mb["path"], proba_col, file_tag)

    return {
        "end_period": end_date,
        "rows_fixtures": len(fixtures),
        "rows_scored": len(scored),
        "n_positive": n_pos,
        "threshold": mb["threshold"],
        "model_path": mb["path"],
        "output_paths": paths,
        "scored_df": scored,
        "fixtures_df": fixtures,  # exposed for reuse by 2H scorer
    }

# ── Public entry points (U25/O25) ────────────────────────────────────────
def run_u25(end_period: Any) -> Dict[str, Any]:
    """Under 2.5 Goals pipeline (writes u25_predictions.csv only)."""
    return _run_pipeline(
        end_period,
        file_path=FILE_PATH,
        model_dir=MODEL_DIR_U25,
        import_dir=IMPORT_DIR,
        proba_col="pred_proba_u25",
        provider=("under_2_5_goals", "Over/Under 2.5 Goals", "Under 2.5 Goals"),
        file_tag="u25",
    )

def run_o25(end_period: Any) -> Dict[str, Any]:
    """Over 2.5 Goals pipeline (writes o25_predictions.csv only)."""
    return _run_pipeline(
        end_period,
        file_path=FILE_PATH,
        model_dir=MODEL_DIR_O25,
        import_dir=IMPORT_DIR,
        proba_col="pred_proba_o25",
        provider=("over_2_5_goals", "Over/Under 2.5 Goals", "Over 2.5 Goals"),
        file_tag="o25",
    )

# ── Public entry point: Second-Half-Goal by HT scoreline ─────────────────
def run_2h_htscore(
    end_period: Any,
    *,
    file_path: str = FILE_PATH,
    import_dir: str = IMPORT_DIR,
    model_dir: str = MODEL_DIR_2H_HTSCORE,
    market_map: Optional[dict[str, float]] = DEFAULT_MARKET_MAP,
) -> Dict[str, Any]:
    """
    Scores second-half goal markets conditioned on half-time scorelines.
    Writes:
      - IMPORTS/2H_GOAL_SCORELINE_ALL.csv       (positives only, compact)
    (No long-form scored CSV is written.)
    Returns a dict with output paths and the long-form DataFrame in-memory.
    """
    os.makedirs(import_dir, exist_ok=True)

    end_date = _parse_end_period(end_period)
    today = datetime.today().date()

    raw = _load_and_prepare_raw(file_path)
    data = _clip_to_period(raw, end_date)
    team_df = _build_team_frame(data)
    team_df = _enrich_team_features(team_df)
    team_df = _add_corners_outcomes_to_team(team_df, data)
    match_df = _prepare_match_level(team_df, data)

    fixtures_df = match_df[(match_df['date'].dt.date >= today) & (match_df['date'].dt.date <= end_date)].copy()
    print(f"[A/6] Fixtures in window [{today} → {end_date}]: {len(fixtures_df)}")

    fixtures_df = _standardise_fixture_names(fixtures_df)

    # Load newest model per HT score
    ht_to_pkl = _newest_pkls_by_htscore(model_dir)
    if not ht_to_pkl:
        raise RuntimeError(f"No PKLs found in {model_dir}")

    # If a market_map is provided, keep only those HT keys (optional)
    if market_map:
        ht_to_pkl = {ht: p for ht, p in ht_to_pkl.items() if ht in market_map}

    all_rows = []
    all_positives = []

    for scoreline, pkl_path in ht_to_pkl.items():
        md = load(pkl_path)
        model      = md['model']
        threshold  = float(md.get('threshold', 0.5))
        feat_list  = list(md['features'])

        df_sub = fixtures_df.copy()
        df_sub['ht_score'] = scoreline

        if 'country' in df_sub.columns and any(f.startswith('country_') for f in feat_list):
            df_sub = pd.get_dummies(df_sub, columns=['country'], prefix='country')

        X = _align_features(df_sub, feat_list)
        before = len(X)
        X = X.dropna()
        df_sub = df_sub.loc[X.index]
        dropped = before - len(X)
        if dropped:
            print(f"[{scoreline}] Dropped {dropped} rows with NaNs after feature alignment.")

        if len(X) == 0:
            print(f"[{scoreline}] No rows left to score.")
            continue

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            from sklearn.preprocessing import MinMaxScaler
            scores = model.decision_function(X).reshape(-1, 1)
            proba = MinMaxScaler().fit_transform(scores).ravel()
        else:
            proba = model.predict(X).astype(float)

        bet_flag = proba >= threshold

        # Labels
        if market_map and scoreline in market_map:
            line = market_map[scoreline]
            provider       = f"second_half_goal_{scoreline.replace('-', '_')}"
            market_name    = f"Over/Under {line} Goals"
            selection_name = f"Under {line} Goals"
        else:
            provider       = f"second_half_goal_{scoreline.replace('-', '_')}"
            market_name    = "Second Half Goal"
            selection_name = "Yes"

        block = df_sub.copy()
        block['ht_score']       = scoreline
        block['proba']          = proba
        block['threshold']      = threshold
        block['bet']            = bet_flag
        block['provider']       = provider
        block['market_name']    = market_name
        block['selection_name'] = selection_name
        all_rows.append(block)

        positives = block.loc[bet_flag].copy()
        if not positives.empty:
            all_positives.append(positives)
            print(f"✓ {len(positives)} selections for HT {scoreline} (≥ {threshold:.2f})")
        else:
            print(f"— 0 selections for HT {scoreline} (≥ {threshold:.2f}) from {len(block)} rows")

    # Long-form scored results kept in memory only
    results_long = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    # Compact positives export (EventName + labels)
    positives_out_path = os.path.join(import_dir, "2H_GOAL_SCORELINE_ALL.csv")

    if all_positives:
        master_df = pd.concat(all_positives, ignore_index=True)
        out_df = pd.DataFrame({
            'EventName':     master_df['home_team'] + ' v ' + master_df['away_team'],
            'Provider':      master_df['provider'],
            'MarketName':    master_df['market_name'],
            'SelectionName': master_df['selection_name'],
        })
        out_df.to_csv(positives_out_path, index=False)
        print(f"✓ Wrote {len(out_df)} positives to:\n   {positives_out_path}")
    else:
        print("No positives found; nothing to write to 2H_GOAL_SCORELINE_ALL.csv")

    # No long-form CSV written
    return {
        "end_period": end_date,
        "rows_fixtures": len(fixtures_df),
        "rows_scored_long": 0 if results_long is None else len(results_long),
        "positives_csv": positives_out_path if all_positives else None,
        "scored_csv": None,
        "results_long_df": results_long,
        "fixtures_df": fixtures_df,  # for inspection if needed
    }

def _prepare_fixtures_window(end_period: Any, file_path: str) -> pd.DataFrame:
    """
    Shared: build match-level features and return only fixtures in [today, end_period],
    with team-name standardisation applied.
    """
    end_date = _parse_end_period(end_period)
    today = datetime.today().date()

    raw = _load_and_prepare_raw(file_path)
    data = _clip_to_period(raw, end_date)

    team_df = _build_team_frame(data)
    team_df = _enrich_team_features(team_df)
    team_df = _add_corners_outcomes_to_team(team_df, data)
    match_df = _prepare_match_level(team_df, data)

    fixtures = match_df[(match_df['date'].dt.date >= today) & (match_df['date'].dt.date <= end_date)].copy()
    print(f"[fixtures] Window [{today} → {end_date}]: {len(fixtures)}")
    fixtures = _standardise_fixture_names(fixtures)
    return fixtures


def _ensure_training_dummies(df_in: pd.DataFrame, feature_contract: list[str]) -> pd.DataFrame:
    """
    Ensure that any one-hot columns expected by the trained model exist.
    Heuristic: if feature name looks like '<base>_<level>' and 'base' exists in df_in,
    create the full dummy set with the expected levels.
    """
    df = df_in.copy()
    bases: dict[str, set[str]] = {}
    for f in feature_contract:
        if "_" in f:
            base, lvl = f.split("_", 1)
            if base in df.columns:
                bases.setdefault(base, set()).add(lvl)

    for base, expected_lvls in bases.items():
        ser = df[base].astype("string")
        cat = pd.Categorical(ser, categories=list(expected_lvls))
        dms = pd.get_dummies(cat)
        dms.columns = [f"{base}_{c}" for c in dms.columns]
        # Make sure every expected level exists
        for lvl in expected_lvls:
            col = f"{base}_{lvl}"
            if col not in dms.columns:
                dms[col] = 0
        # Add any missing dummy columns to df
        for col in dms.columns:
            if col not in df.columns:
                df[col] = dms[col].astype("int8")
    return df


def run_lay_home(
    end_period: Any,
    pkl_path: str,
    *,
    file_path: str = FILE_PATH,
    import_dir: str = IMPORT_DIR,
    file_tag: str = "lay_home",
) -> Dict[str, Any]:
    """
    LAY HOME runner.
    - Loads a calibrated classifier bundle at `pkl_path`.
    - Builds fixtures in [today, end_period] using the shared pipeline.
    - Computes model-implied lay max price for the HOME selection.
    - Writes IMPORT CSV with columns: Provider, MarketName, SelectionName, MaxPrice, EventName.
    Returns metadata + the fixtures scored in-memory.
    """
    os.makedirs(import_dir, exist_ok=True)
    end_date = _parse_end_period(end_period)

    fixtures_df = _prepare_fixtures_window(end_date, file_path)
    if fixtures_df.empty:
        out_csv = os.path.join(import_dir, f"{file_tag}_import.csv")
        print(f"[lay_home] No fixtures in window. Writing empty file: {out_csv}")
        pd.DataFrame(columns=["Provider","MarketName","SelectionName","MaxPrice","EventName"]).to_csv(out_csv, index=False)
        return {
            "end_period": end_date,
            "rows_fixtures": 0,
            "output_csv": out_csv,
            "scored_df": fixtures_df,
        }

    # Load trained bundle
    bundle = load(pkl_path)
    model     = bundle["model"]
    feat_list = list(bundle["features"])
    mode      = str(bundle.get("mode", ""))            # e.g. 'VALUE_LAY'
    edge      = float(bundle.get("edge_param", 0.0))   # e.g. 0.05 for 5%

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model has no predict_proba; expected a calibrated classifier.")

    # Make sure all expected dummy cols exist (e.g. country_*)
    df_aug = _ensure_training_dummies(fixtures_df, feat_list)

    # Final alignment/coercion
    X = _align_features_numeric_fill(df_aug, feat_list)

    # Predict class-1 prob as trained; for LAY_HOME we want P(home win)
    proba = model.predict_proba(X)
    p_pos = proba[:, 1] if getattr(proba, "ndim", 1) == 2 else proba
    # Training convention you provided: lay-home uses (1 - p_pos) to get P(home win)
    p_home = 1.0 - np.asarray(p_pos, dtype=float)

    fair_odds_home = 1.0 / np.clip(p_home, 1e-9, 1.0)

    # Edge handling (only if mode/value provided)
    use_edge = (mode.upper() == "VALUE_LAY") and np.isfinite(edge) and (edge >= 0.0)
    max_price = np.divide(fair_odds_home, (1.0 + edge)) if use_edge else fair_odds_home
    # Excel-style to 2dp (consistent with your helper)
    max_price = np.vectorize(_round_half_up_2)(max_price)

    # EventName preference: EvenName → EventName → teams
    if "EvenName" in df_aug.columns:
        event_name = df_aug["EvenName"].astype(str)
    elif "EventName" in df_aug.columns:
        event_name = df_aug["EventName"].astype(str)
    elif {"home_team", "away_team"}.issubset(df_aug.columns):
        event_name = df_aug["home_team"].astype(str) + " v " + df_aug["away_team"].astype(str)
    else:
        raise KeyError("No 'EvenName'/'EventName' and cannot build from teams.")

    if "home_team" not in df_aug.columns:
        raise KeyError("Fixtures must contain 'home_team' to build SelectionName.")

    import_df = pd.DataFrame({
        "Provider":      "lay_home",
        "MarketName":    "Match Odds",
        "SelectionName": df_aug["home_team"].astype(str),
        "MaxPrice":      max_price,
        "EventName":     event_name,
    })

    out_csv = os.path.join(import_dir, f"{file_tag}_import.csv")
    import_df.to_csv(out_csv, index=False)

    print(f"[lay_home] Wrote {len(import_df)} rows to: {out_csv}")
    print(f"           Model file → {pkl_path} | mode={mode} edge={edge}")

    # Attach the prices back to fixtures_df for inspection
    out_df = fixtures_df.loc[X.index].copy()
    out_df["lay_home_max_price"] = pd.to_numeric(import_df["MaxPrice"], errors="coerce")

    return {
        "end_period": end_date,
        "rows_fixtures": len(fixtures_df),
        "rows_scored": len(out_df),
        "output_csv": out_csv,
        "model_path": pkl_path,
        "mode": mode,
        "edge": edge,
        "scored_df": out_df,
    }

def run_lay_away(
    end_period: Any,
    pkl_path: str,
    *,
    file_path: str = FILE_PATH,
    import_dir: str = IMPORT_DIR,
) -> Dict[str, Any]:
    """
    LAY AWAY runner.

    Loads a calibrated classifier bundle at `pkl_path`, builds fixtures in
    [today, end_period], computes model-implied lay max price for the AWAY
    selection (edge-adjusted if the bundle has mode='VALUE_LAY' and edge_param),
    and writes IMPORT CSV with columns:
      Provider, MarketName, SelectionName, MaxPrice, EventName

    Returns metadata and a scored DataFrame in-memory.
    """
    os.makedirs(import_dir, exist_ok=True)
    end_date = _parse_end_period(end_period)

    fixtures_df = _prepare_fixtures_window(end_date, file_path)
    out_csv = os.path.join(import_dir, "lay_away_import.csv")

    if fixtures_df.empty:
        pd.DataFrame(columns=["Provider","MarketName","SelectionName","MaxPrice","EventName"]).to_csv(out_csv, index=False)
        print(f"[lay_away] No fixtures in window. Wrote empty CSV: {out_csv}")
        return {
            "end_period": end_date, "rows_fixtures": 0, "rows_scored": 0,
            "output_csv": out_csv, "model_path": pkl_path, "scored_df": fixtures_df,
        }

    bundle    = load(pkl_path)
    model     = bundle["model"]
    feat_list = list(bundle["features"])
    mode      = str(bundle.get("mode", ""))
    edge      = float(bundle.get("edge_param", 0.0))

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model has no predict_proba; expected a calibrated classifier.")

    # Make sure all expected dummy columns exist
    df_aug = _ensure_training_dummies(fixtures_df, feat_list)
    # Align/coerce numerics safely (uses your existing helper)
    X = _align_features_numeric_fill(df_aug, feat_list)

    proba = model.predict_proba(X)
    p_pos = proba[:, 1] if getattr(proba, "ndim", 1) == 2 else proba

    # Training convention: for LAY we invert class-1 to get P(away win)
    p_away = 1.0 - np.asarray(p_pos, dtype=float)
    fair_odds_away = 1.0 / np.clip(p_away, 1e-9, 1.0)

    # Edge handling
    use_edge = (mode.upper() == "VALUE_LAY") and np.isfinite(edge) and (edge >= 0.0)
    max_price = np.divide(fair_odds_away, (1.0 + edge)) if use_edge else fair_odds_away
    # Excel-style 2dp rounding
    max_price = np.array([_round_half_up_2(v) for v in max_price], dtype=object)

    # EventName: EvenName → EventName → "Home v Away"
    if "EvenName" in df_aug.columns:
        event_name = df_aug["EvenName"].astype(str)
    elif "EventName" in df_aug.columns:
        event_name = df_aug["EventName"].astype(str)
    elif {"home_team","away_team"}.issubset(df_aug.columns):
        event_name = df_aug["home_team"].astype(str) + " v " + df_aug["away_team"].astype(str)
    else:
        raise KeyError("No 'EvenName'/'EventName' present and cannot build from teams.")

    if "away_team" not in df_aug.columns:
        raise KeyError("Fixtures must contain 'away_team' for LAY AWAY SelectionName.")

    import_df = pd.DataFrame({
        "Provider":      "lay_away",
        "MarketName":    "Match Odds",
        "SelectionName": df_aug["away_team"].astype(str),
        "MaxPrice":      max_price,
        "EventName":     event_name,
    })

    import_df.to_csv(out_csv, index=False)
    print(f"[lay_away] Wrote {len(import_df)} rows → {out_csv}")
    print(f"           Model → {pkl_path} | mode={mode} edge={edge}")

    # Attach price back onto fixtures for inspection
    out_df = fixtures_df.loc[X.index].copy()
    out_df["lay_away_max_price"] = pd.to_numeric(import_df["MaxPrice"], errors="coerce")

    return {
        "end_period": end_date,
        "rows_fixtures": len(fixtures_df),
        "rows_scored": len(out_df),
        "output_csv": out_csv,
        "model_path": pkl_path,
        "mode": mode,
        "edge": edge,
        "scored_df": out_df,
    }

def run_lay_draw(
    end_period: Any,
    pkl_path: str,
    *,
    file_path: str = FILE_PATH,
    import_dir: str = IMPORT_DIR,
    file_tag: str = "lay_draw",
) -> Dict[str, Any]:
    """
    LAY DRAW runner.

    - Loads a calibrated classifier bundle at `pkl_path`.
    - Builds fixtures in [today, end_period] using the shared pipeline.
    - Estimates model-implied fair odds for the DRAW, then applies optional
      edge adjustment when bundle has mode='VALUE_LAY' and 'edge_param'.
    - Writes IMPORT CSV with columns:
        Provider, MarketName, SelectionName, MaxPrice, EventName

    Returns metadata + a scored DataFrame in-memory with audit columns.
    """
    import os
    os.makedirs(import_dir, exist_ok=True)
    end_date = _parse_end_period(end_period)

    # Build fixtures window via your shared pipeline
    fixtures_df = _prepare_fixtures_window(end_date, file_path)
    out_csv = os.path.join(import_dir, f"{file_tag}_import.csv")

    if fixtures_df.empty:
        pd.DataFrame(
            columns=["Provider", "MarketName", "SelectionName", "MaxPrice", "EventName"]
        ).to_csv(out_csv, index=False)
        print(f"[lay_draw] No fixtures in window. Wrote empty CSV: {out_csv}")
        return {
            "end_period": end_date,
            "rows_fixtures": 0,
            "rows_scored": 0,
            "output_csv": out_csv,
            "model_path": pkl_path,
            "scored_df": fixtures_df,
        }

    # Load model bundle
    bundle    = load(pkl_path)
    model     = bundle["model"]
    feat_list = list(bundle["features"])
    mode      = str(bundle.get("mode", ""))
    edge      = float(bundle.get("edge_param", 0.0))
    target_positive_label = str(bundle.get("target_positive_label", "")).lower()

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model has no predict_proba; expected a calibrated classifier.")

    # Minor robustness (typoed column sometimes appears)
    if "county" in fixtures_df.columns and "country" not in fixtures_df.columns:
        fixtures_df = fixtures_df.rename(columns={"county": "country"})

    # Ensure expected one-hot columns and align numerics
    df_aug = _ensure_training_dummies(fixtures_df, feat_list)
    X      = _align_features_numeric_fill(df_aug, feat_list)

    # Predict class-1 probability from calibrated classifier
    proba = model.predict_proba(X)
    p_pos = proba[:, 1] if getattr(proba, "ndim", 1) == 2 else proba

    # Map to DRAW probability:
    # If bundle declares the positive label explicitly, use it; otherwise default
    # to the same convention you used for LAY_HOME/AWAY (invert p_pos).
    if target_positive_label in {"draw", "the_draw", "x", "d"}:
        p_draw = np.asarray(p_pos, dtype=float)
        assumed = False
    elif target_positive_label in {"not_draw", "no_draw", "!draw"}:
        p_draw = 1.0 - np.asarray(p_pos, dtype=float)
        assumed = False
    else:
        # Fallback (same pattern as your other LAY runners)
        p_draw = 1.0 - np.asarray(p_pos, dtype=float)
        assumed = True

    # Fair odds & edge-adjusted lay max price
    fair_odds_draw = 1.0 / np.clip(p_draw, 1e-9, 1.0)
    use_edge = (mode.upper() == "VALUE_LAY") and np.isfinite(edge) and (edge >= 0.0)
    max_price = np.divide(fair_odds_draw, (1.0 + edge)) if use_edge else fair_odds_draw
    max_price_str = np.vectorize(_round_half_up_2)(max_price)

    # EventName preference: EvenName → EventName → "Home v Away"
    if "EvenName" in df_aug.columns:
        event_name = df_aug["EvenName"].astype(str)
    elif "EventName" in df_aug.columns:
        event_name = df_aug["EventName"].astype(str)
    elif {"home_team", "away_team"}.issubset(df_aug.columns):
        event_name = df_aug["home_team"].astype(str) + " v " + df_aug["away_team"].astype(str)
    else:
        raise KeyError("No 'EvenName'/'EventName' present and cannot build from teams.")

    # Build IMPORT CSV (exact columns)
    import_df = pd.DataFrame({
        "Provider":      "lay_draw",
        "MarketName":    "Match Odds",
        "SelectionName": "The Draw",
        "MaxPrice":      max_price_str,   # 2dp Excel-style strings
        "EventName":     event_name,
    })
    import_df.to_csv(out_csv, index=False)
    print(f"[lay_draw] Wrote {len(import_df)} rows → {out_csv}")
    print(f"           Model → {pkl_path} | mode={mode} edge={edge}"
          + (" | (assumed invert p_pos for draw)" if assumed else ""))

    # Attach audit columns back onto fixtures for inspection
    out_df = fixtures_df.loc[X.index].copy()
    out_df["lay_draw_fair"] = fair_odds_draw
    out_df["lay_draw_max_price"] = pd.to_numeric(import_df["MaxPrice"], errors="coerce")
    out_df["lay_draw_edge_param"] = edge
    out_df["lay_draw_label_assumed"] = assumed

    return {
        "end_period": end_date,
        "rows_fixtures": len(fixtures_df),
        "rows_scored": len(out_df),
        "output_csv": out_csv,
        "model_path": pkl_path,
        "mode": mode,
        "edge": edge,
        "scored_df": out_df,
    }


