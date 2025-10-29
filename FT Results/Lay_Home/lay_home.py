import time
from datetime import datetime
import numpy as np
import pandas as pd
import re
import glob
import function_library as fl
import os
import re
import glob
from typing import Tuple

features_old = [
    # General
    'round',
    'home_team_place_total',
    'home_team_place_home',
    'away_team_place_total',
    'away_team_place_away',
    'home_odds',
    'draw_odds',
    'away_odds',
    'over_25_odds',
    'under_25_odds',
    'elo_home',
    'elo_away',
    'form_home',
    'form_away',

    # Home team overall features
    'home_Overall_RollingGoalsScored_Mean',
    'home_Overall_RollingGoalsConceded_Mean',
    'home_Overall_RollingGoalsScored_Std',
    'home_Overall_RollingGoalsConceded_Std',
    'home_Overall_RollingGoalsScored_Mean_Short',
    'home_Overall_Momentum_GoalsScored',
    'home_Overall_Trend_Slope_GoalsScored',
    'home_Overall_RollingFirstHalfGoalsScored_Mean',
    'home_Overall_RollingFirstHalfGoalsConceded_Mean',
    'home_Overall_RollingFirstHalfGoalsScored_Std',
    'home_Overall_RollingFirstHalfGoalsConceded_Std',
    'home_Overall_RollingFirstHalfGoalsScored_Mean_Short',
    'home_Overall_Momentum_FirstHalfGoalsScored',
    'home_Overall_Trend_Slope_FirstHalfGoalsScored',

    # Home team specific features (home-only)
    'home_Home_RollingGoalsScored_Mean',
    'home_Home_RollingGoalsConceded_Mean',
    'home_Home_RollingGoalsScored_Std',
    'home_Home_RollingGoalsConceded_Std',
    'home_Home_RollingGoalsScored_Mean_Short',
    'home_Home_Momentum_GoalsScored',
    'home_Home_Trend_Slope_GoalsScored',
    'home_Home_RollingFirstHalfGoalsScored_Mean',
    'home_Home_RollingFirstHalfGoalsConceded_Mean',
    'home_Home_RollingFirstHalfGoalsScored_Std',
    'home_Home_RollingFirstHalfGoalsConceded_Std',
    'home_Home_RollingFirstHalfGoalsScored_Mean_Short',
    'home_Home_Momentum_FirstHalfGoalsScored',
    'home_Home_Trend_Slope_FirstHalfGoalsScored',

    # Away team overall features
    'away_Overall_RollingGoalsScored_Mean',
    'away_Overall_RollingGoalsConceded_Mean',
    'away_Overall_RollingGoalsScored_Std',
    'away_Overall_RollingGoalsConceded_Std',
    'away_Overall_RollingGoalsScored_Mean_Short',
    'away_Overall_Momentum_GoalsScored',
    'away_Overall_Trend_Slope_GoalsScored',
    'away_Overall_RollingFirstHalfGoalsScored_Mean',
    'away_Overall_RollingFirstHalfGoalsConceded_Mean',
    'away_Overall_RollingFirstHalfGoalsScored_Std',
    'away_Overall_RollingFirstHalfGoalsConceded_Std',
    'away_Overall_RollingFirstHalfGoalsScored_Mean_Short',
    'away_Overall_Momentum_FirstHalfGoalsScored',
    'away_Overall_Trend_Slope_FirstHalfGoalsScored',

    # Away team specific features (away-only)
    'away_Away_RollingGoalsScored_Mean',
    'away_Away_RollingGoalsConceded_Mean',
    'away_Away_RollingGoalsScored_Std',
    'away_Away_RollingGoalsConceded_Std',
    'away_Away_RollingGoalsScored_Mean_Short',
    'away_Away_Momentum_GoalsScored',
    'away_Away_Trend_Slope_GoalsScored',
    'away_Away_RollingFirstHalfGoalsScored_Mean',
    'away_Away_RollingFirstHalfGoalsConceded_Mean',
    'away_Away_RollingFirstHalfGoalsScored_Std',
    'away_Away_RollingFirstHalfGoalsConceded_Std',
    'away_Away_RollingFirstHalfGoalsScored_Mean_Short',
    'away_Away_Momentum_FirstHalfGoalsScored',
    'away_Away_Trend_Slope_FirstHalfGoalsScored',

    # ----- Additional Goal Threshold Percentages (Per-Match Metrics) -----
    # For thresholds: 1.5, 2.5, 3.5
    # Overall team (season cumulative and rolling last 5 matches)
    'home_Overall_Percent_Over_1.5',
    'home_Overall_Rolling5_Percent_Over_1.5',
    'home_Overall_Percent_Over_2.5',
    'home_Overall_Rolling5_Percent_Over_2.5',
    'home_Overall_Percent_Over_3.5',
    'home_Overall_Rolling5_Percent_Over_3.5',

    'away_Overall_Percent_Over_1.5',
    'away_Overall_Rolling5_Percent_Over_1.5',
    'away_Overall_Percent_Over_2.5',
    'away_Overall_Rolling5_Percent_Over_2.5',
    'away_Overall_Percent_Over_3.5',
    'away_Overall_Rolling5_Percent_Over_3.5',

    # Home matches only
    'home_Home_Percent_Over_1.5',
    'home_Home_Rolling5_Percent_Over_1.5',
    'home_Home_Percent_Over_2.5',
    'home_Home_Rolling5_Percent_Over_2.5',
    'home_Home_Percent_Over_3.5',
    'home_Home_Rolling5_Percent_Over_3.5',

    # Away matches only
    'away_Away_Percent_Over_1.5',
    'away_Away_Rolling5_Percent_Over_1.5',
    'away_Away_Percent_Over_2.5',
    'away_Away_Rolling5_Percent_Over_2.5',
    'away_Away_Percent_Over_3.5',
    'away_Away_Rolling5_Percent_Over_3.5',

    # Home matches only
    'home_Home_TeamPct_Over_0.5',
    'home_Home_TeamPct_Over_1.5',
    'home_Home_TeamPct_Over_2.5',
    'home_Home_TeamPct_Over_3.5',

    # Away matches only
    'away_Away_TeamPct_Over_0.5',
    'away_Away_TeamPct_Over_1.5',
    'away_Away_TeamPct_Over_2.5',
    'away_Away_TeamPct_Over_3.5'
]

features = [
    # 'Unnamed: 0',
    # 'country',
    # 'season',
    # 'date',
    # 'ko_time',
    'round',
    # 'home_team',
    # 'away_team',
    # 'home_goals_ft',
    # 'away_goals_ft',
    # 'home_goals_ht',
    # 'away_goals_ht',
    'home_team_place_total',
    'home_team_place_home',
    'away_team_place_total',
    'away_team_place_away',
    'home_odds',
    'draw_odds',
    'away_odds',
    'over_25_odds',
    'under_25_odds',
    'elo_home',
    'elo_away',
    'form_home',
    'form_away',
    # 'shots_home',
    # 'shots_home_1h',
    # 'shots_home_2h',
    # 'shots_away',
    # 'shots_away_1h',
    # 'shots_away_2h',
    # 'shots_on_target_home',
    # 'shots_on_target_home_1h',
    # 'shots_on_target_home_2h',
    # 'shots_on_target_away',
    # 'shots_on_target_away_1h',
    # 'shots_on_target_away_2h',
    # 'corners_home',
    # 'corners_home_1h',
    # 'corners_home_2h',
    # 'corners_away',
    # 'corners_away_1h',
    # 'corners_away_2h',
    # 'fouls_home',
    # 'fouls_home_1h',
    # 'fouls_home_2h',
    # 'fouls_away',
    # 'fouls_away_1h',
    # 'fouls_away_2h',
    # 'yellow_cards_home',
    # 'yellow_cards_home_1h',
    # 'yellow_cards_home_2h',
    # 'yellow_cards_away',
    # 'yellow_cards_away_1h',
    # 'yellow_cards_away_2h',
    # 'possession_home',
    # 'possession_home_1h',
    # 'possession_home_2h',
    # 'possession_away',
    # 'possession_away_1h',
    # 'possession_away_2h',
    # 'goals_scored_total_home',
    # 'goals_conceded_total_home',
    # 'goals_scored_total_away',
    # 'goals_conceded_total_away',
    # 'points_home',
    # 'points_away',
    # 'is_home_x',
    'home_Overall_Rolling_GoalsScored_Mean',
    'home_Overall_Rolling_GoalsScored_Std',
    'home_Overall_Rolling_GoalsScored_Mean_Short',
    'home_Overall_Momentum_GoalsScored',
    'home_Overall_Trend_Slope_GoalsScored',
    'home_Overall_Rolling_FirstHalfGoalsScored_Mean',
    'home_Overall_Rolling_FirstHalfGoalsScored_Std',
    'home_Overall_Rolling_FirstHalfGoalsScored_Mean_Short',
    'home_Overall_Momentum_FirstHalfGoalsScored',
    'home_Overall_Trend_Slope_FirstHalfGoalsScored',
    'home_Overall_Rolling_Shots_Mean',
    'home_Overall_Rolling_Shots_Std',
    'home_Overall_Rolling_Shots_Mean_Short',
    'home_Overall_Momentum_Shots',
    'home_Overall_Trend_Slope_Shots',
    'home_Overall_Rolling_Shots_1h_Mean',
    'home_Overall_Rolling_Shots_1h_Std',
    'home_Overall_Rolling_Shots_1h_Mean_Short',
    'home_Overall_Momentum_Shots_1h',
    'home_Overall_Trend_Slope_Shots_1h',
    'home_Overall_Rolling_Corners_Mean',
    'home_Overall_Rolling_Corners_Std',
    'home_Overall_Rolling_Corners_Mean_Short',
    'home_Overall_Momentum_Corners',
    'home_Overall_Trend_Slope_Corners',
    'home_Overall_Rolling_Corners_1h_Mean',
    'home_Overall_Rolling_Corners_1h_Std',
    'home_Overall_Rolling_Corners_1h_Mean_Short',
    'home_Overall_Momentum_Corners_1h',
    'home_Overall_Trend_Slope_Corners_1h',
    'home_Overall_Rolling_ShotsOnTarget_Mean',
    'home_Overall_Rolling_ShotsOnTarget_Std',
    'home_Overall_Rolling_ShotsOnTarget_Mean_Short',
    'home_Overall_Momentum_ShotsOnTarget',
    'home_Overall_Trend_Slope_ShotsOnTarget',
    'home_Overall_Rolling_ShotsOnTarget_1h_Mean',
    'home_Overall_Rolling_ShotsOnTarget_1h_Std',
    'home_Overall_Rolling_ShotsOnTarget_1h_Mean_Short',
    'home_Overall_Momentum_ShotsOnTarget_1h',
    'home_Overall_Trend_Slope_ShotsOnTarget_1h',
    'home_Rolling_GoalsScored_Mean',
    'home_Rolling_GoalsScored_Std',
    'home_Rolling_GoalsScored_Mean_Short',
    'home_Momentum_GoalsScored',
    'home_Trend_Slope_GoalsScored',
    'home_Rolling_FirstHalfGoalsScored_Mean',
    'home_Rolling_FirstHalfGoalsScored_Std',
    'home_Rolling_FirstHalfGoalsScored_Mean_Short',
    'home_Momentum_FirstHalfGoalsScored',
    'home_Trend_Slope_FirstHalfGoalsScored',
    'home_Rolling_Shots_Mean',
    'home_Rolling_Shots_Std',
    'home_Rolling_Shots_Mean_Short',
    'home_Momentum_Shots',
    'home_Trend_Slope_Shots',
    'home_Rolling_Shots_1h_Mean',
    'home_Rolling_Shots_1h_Std',
    'home_Rolling_Shots_1h_Mean_Short',
    'home_Momentum_Shots_1h',
    'home_Trend_Slope_Shots_1h',
    'home_Rolling_Corners_Mean',
    'home_Rolling_Corners_Std',
    'home_Rolling_Corners_Mean_Short',
    'home_Momentum_Corners',
    'home_Trend_Slope_Corners',
    'home_Rolling_Corners_1h_Mean',
    'home_Rolling_Corners_1h_Std',
    'home_Rolling_Corners_1h_Mean_Short',
    'home_Momentum_Corners_1h',
    'home_Trend_Slope_Corners_1h',
    'home_Rolling_ShotsOnTarget_Mean',
    'home_Rolling_ShotsOnTarget_Std',
    'home_Rolling_ShotsOnTarget_Mean_Short',
    'home_Momentum_ShotsOnTarget',
    'home_Trend_Slope_ShotsOnTarget',
    'home_Rolling_ShotsOnTarget_1h_Mean',
    'home_Rolling_ShotsOnTarget_1h_Std',
    'home_Rolling_ShotsOnTarget_1h_Mean_Short',
    'home_Momentum_ShotsOnTarget_1h',
    'home_Trend_Slope_ShotsOnTarget_1h',
    'home_Overall_Percent_Over_1.5',
    'home_Overall_Rolling5_Percent_Over_1.5',
    'home_Percent_Over_1.5',
    'home_Rolling5_Percent_Over_1.5',
    'home_Overall_Percent_Over_2.5',
    'home_Overall_Rolling5_Percent_Over_2.5',
    'home_Percent_Over_2.5',
    'home_Rolling5_Percent_Over_2.5',
    'home_Overall_Percent_Over_3.5',
    'home_Overall_Rolling5_Percent_Over_3.5',
    'home_Percent_Over_3.5',
    'home_Rolling5_Percent_Over_3.5',
    'home_TeamPct_Over_0.5',
    'home_TeamPct_Over_1.5',
    'home_TeamPct_Over_2.5',
    'home_TeamPct_Over_3.5',
    'home_CornersPct_Over_3.5',
    'home_CornersRolling5Pct_Over_3.5',
    'home_CornersPct_Over_4.5',
    'home_CornersRolling5Pct_Over_4.5',
    'home_CornersPct_Over_5.5',
    'home_CornersRolling5Pct_Over_5.5',
    'home_CornersPct_Over_6.5',
    'home_CornersRolling5Pct_Over_6.5',
    'home_SeasonPct_Over_9.5',
    'home_Rolling5Pct_Over_9.5',
    'home_SeasonPct_Over_10.5',
    'home_Rolling5Pct_Over_10.5',
    'home_SeasonPct_Over_11.5',
    'home_Rolling5Pct_Over_11.5',
    # 'is_home_y',
    'away_Overall_Rolling_GoalsScored_Mean',
    'away_Overall_Rolling_GoalsScored_Std',
    'away_Overall_Rolling_GoalsScored_Mean_Short',
    'away_Overall_Momentum_GoalsScored',
    'away_Overall_Trend_Slope_GoalsScored',
    'away_Overall_Rolling_FirstHalfGoalsScored_Mean',
    'away_Overall_Rolling_FirstHalfGoalsScored_Std',
    'away_Overall_Rolling_FirstHalfGoalsScored_Mean_Short',
    'away_Overall_Momentum_FirstHalfGoalsScored',
    'away_Overall_Trend_Slope_FirstHalfGoalsScored',
    'away_Overall_Rolling_Shots_Mean',
    'away_Overall_Rolling_Shots_Std',
    'away_Overall_Rolling_Shots_Mean_Short',
    'away_Overall_Momentum_Shots',
    'away_Overall_Trend_Slope_Shots',
    'away_Overall_Rolling_Shots_1h_Mean',
    'away_Overall_Rolling_Shots_1h_Std',
    'away_Overall_Rolling_Shots_1h_Mean_Short',
    'away_Overall_Momentum_Shots_1h',
    'away_Overall_Trend_Slope_Shots_1h',
    'away_Overall_Rolling_Corners_Mean',
    'away_Overall_Rolling_Corners_Std',
    'away_Overall_Rolling_Corners_Mean_Short',
    'away_Overall_Momentum_Corners',
    'away_Overall_Trend_Slope_Corners',
    'away_Overall_Rolling_Corners_1h_Mean',
    'away_Overall_Rolling_Corners_1h_Std',
    'away_Overall_Rolling_Corners_1h_Mean_Short',
    'away_Overall_Momentum_Corners_1h',
    'away_Overall_Trend_Slope_Corners_1h',
    'away_Overall_Rolling_ShotsOnTarget_Mean',
    'away_Overall_Rolling_ShotsOnTarget_Std',
    'away_Overall_Rolling_ShotsOnTarget_Mean_Short',
    'away_Overall_Momentum_ShotsOnTarget',
    'away_Overall_Trend_Slope_ShotsOnTarget',
    'away_Overall_Rolling_ShotsOnTarget_1h_Mean',
    'away_Overall_Rolling_ShotsOnTarget_1h_Std',
    'away_Overall_Rolling_ShotsOnTarget_1h_Mean_Short',
    'away_Overall_Momentum_ShotsOnTarget_1h',
    'away_Overall_Trend_Slope_ShotsOnTarget_1h',
    'away_Rolling_GoalsScored_Mean',
    'away_Rolling_GoalsScored_Std',
    'away_Rolling_GoalsScored_Mean_Short',
    'away_Momentum_GoalsScored',
    'away_Trend_Slope_GoalsScored',
    'away_Rolling_FirstHalfGoalsScored_Mean',
    'away_Rolling_FirstHalfGoalsScored_Std',
    'away_Rolling_FirstHalfGoalsScored_Mean_Short',
    'away_Momentum_FirstHalfGoalsScored',
    'away_Trend_Slope_FirstHalfGoalsScored',
    'away_Rolling_Shots_Mean',
    'away_Rolling_Shots_Std',
    'away_Rolling_Shots_Mean_Short',
    'away_Momentum_Shots',
    'away_Trend_Slope_Shots',
    'away_Rolling_Shots_1h_Mean',
    'away_Rolling_Shots_1h_Std',
    'away_Rolling_Shots_1h_Mean_Short',
    'away_Momentum_Shots_1h',
    'away_Trend_Slope_Shots_1h',
    'away_Rolling_Corners_Mean',
    'away_Rolling_Corners_Std',
    'away_Rolling_Corners_Mean_Short',
    'away_Momentum_Corners',
    'away_Trend_Slope_Corners',
    'away_Rolling_Corners_1h_Mean',
    'away_Rolling_Corners_1h_Std',
    'away_Rolling_Corners_1h_Mean_Short',
    'away_Momentum_Corners_1h',
    'away_Trend_Slope_Corners_1h',
    'away_Rolling_ShotsOnTarget_Mean',
    'away_Rolling_ShotsOnTarget_Std',
    'away_Rolling_ShotsOnTarget_Mean_Short',
    'away_Momentum_ShotsOnTarget',
    'away_Trend_Slope_ShotsOnTarget',
    'away_Rolling_ShotsOnTarget_1h_Mean',
    'away_Rolling_ShotsOnTarget_1h_Std',
    'away_Rolling_ShotsOnTarget_1h_Mean_Short',
    'away_Momentum_ShotsOnTarget_1h',
    'away_Trend_Slope_ShotsOnTarget_1h',
    'away_Overall_Percent_Over_1.5',
    'away_Overall_Rolling5_Percent_Over_1.5',
    'away_Percent_Over_1.5',
    'away_Rolling5_Percent_Over_1.5',
    'away_Overall_Percent_Over_2.5',
    'away_Overall_Rolling5_Percent_Over_2.5',
    'away_Percent_Over_2.5',
    'away_Rolling5_Percent_Over_2.5',
    'away_Overall_Percent_Over_3.5',
    'away_Overall_Rolling5_Percent_Over_3.5',
    'away_Percent_Over_3.5',
    'away_Rolling5_Percent_Over_3.5',
    'away_TeamPct_Over_0.5',
    'away_TeamPct_Over_1.5',
    'away_TeamPct_Over_2.5',
    'away_TeamPct_Over_3.5',
    'away_CornersPct_Over_3.5',
    'away_CornersRolling5Pct_Over_3.5',
    'away_CornersPct_Over_4.5',
    'away_CornersRolling5Pct_Over_4.5',
    'away_CornersPct_Over_5.5',
    'away_CornersRolling5Pct_Over_5.5',
    'away_CornersPct_Over_6.5',
    'away_CornersRolling5Pct_Over_6.5',
    'away_SeasonPct_Over_9.5',
    'away_Rolling5Pct_Over_9.5',
    'away_SeasonPct_Over_10.5',
    'away_Rolling5Pct_Over_10.5',
    'away_SeasonPct_Over_11.5',
    'away_Rolling5Pct_Over_11.5'
]


def prepare_data(file_path):
    # Attempt to read the CSV file using different encodings
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            print(f"Successfully read the file with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            print(f"Failed to decode with encoding: {encoding}")

    # Define the column renaming dictionary
    column_dict = {
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

    # Rename and filter columns
    data = data.rename(columns=column_dict).filter(items=column_dict.values())

    # Convert 'date' column to datetime object
    data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d", errors='coerce')
    data = data.sort_values(by='date')

    # Convert today's date to a pandas Timestamp for compatibility.
    today = pd.Timestamp(datetime.today().date())
    data = data[data['date'] <= today]

    # Assign points based on match results
    data["points_home"] = data.apply(
        lambda row: 3 if row["home_goals_ft"] > row["away_goals_ft"]
        else (1 if row["home_goals_ft"] == row["away_goals_ft"] else 0),
        axis=1,
    )
    data["points_away"] = data.apply(
        lambda row: 3 if row["away_goals_ft"] > row["home_goals_ft"]
        else (1 if row["away_goals_ft"] == row["home_goals_ft"] else 0),
        axis=1,
    )

    # -----------------------------
    # Prepare team-level DataFrame
    # -----------------------------
    # Create a home team DataFrame
    home_df = data[
        ['country', 'season', 'date', 'home_team', 'away_team',
         'home_goals_ft', 'away_goals_ft', 'home_goals_ht', 'away_goals_ht']
    ].copy()
    home_df.rename(
        columns={
            'home_team': 'Team',
            'away_team': 'Opponent',
            'home_goals_ft': 'GoalsScored',
            'away_goals_ft': 'GoalsConceded',
            'home_goals_ht': 'FirstHalfGoalsScored',
            'away_goals_ht': 'FirstHalfGoalsConceded'
        },
        inplace=True,
    )
    home_df['is_home'] = 1

    # Create an away team DataFrame
    away_df = data[
        ['country', 'season', 'date', 'away_team', 'home_team',
         'away_goals_ft', 'home_goals_ft', 'away_goals_ht', 'home_goals_ht']
    ].copy()
    away_df.rename(
        columns={
            'away_team': 'Team',
            'home_team': 'Opponent',
            'away_goals_ft': 'GoalsScored',
            'home_goals_ft': 'GoalsConceded',
            'away_goals_ht': 'FirstHalfGoalsScored',
            'home_goals_ht': 'FirstHalfGoalsConceded'
        },
        inplace=True,
    )
    away_df['is_home'] = 0

    # Combine the home and away DataFrames into one team-level DataFrame
    team_df = pd.concat([home_df, away_df], ignore_index=True)
    team_df.sort_values(by=['country', 'season', 'Team', 'date'], inplace=True)

    # Define rolling window sizes
    window_long = 5  # e.g. last 5 matches for long-term trends
    window_short = 3  # e.g. last 3 matches for short-term momentum

    # -----------------------------
    # Rolling Feature Computation Functions
    # -----------------------------
    def compute_rolling(df_sub, prefix):
        """
        Compute rolling features on a given (sorted) DataFrame subset.
        The prefix will be used to name the computed columns.
        """
        df = df_sub.copy()

        # Full-Time Goals Rolling Features
        df[prefix + '_RollingGoalsScored_Mean'] = df['GoalsScored'].rolling(window=window_long,
                                                                            min_periods=1).mean().shift(1)
        df[prefix + '_RollingGoalsConceded_Mean'] = df['GoalsConceded'].rolling(window=window_long,
                                                                                min_periods=1).mean().shift(1)
        df[prefix + '_RollingGoalsScored_Std'] = df['GoalsScored'].rolling(window=window_long,
                                                                           min_periods=1).std().shift(1)
        df[prefix + '_RollingGoalsConceded_Std'] = df['GoalsConceded'].rolling(window=window_long,
                                                                               min_periods=1).std().shift(1)
        df[prefix + '_RollingGoalsScored_Mean_Short'] = df['GoalsScored'].rolling(window=window_short,
                                                                                  min_periods=1).mean().shift(1)
        df[prefix + '_Momentum_GoalsScored'] = df[prefix + '_RollingGoalsScored_Mean_Short'] - df[
            prefix + '_RollingGoalsScored_Mean']

        # First-Half Goals Rolling Features
        df[prefix + '_RollingFirstHalfGoalsScored_Mean'] = df['FirstHalfGoalsScored'].rolling(window=window_long,
                                                                                              min_periods=1).mean().shift(
            1)
        df[prefix + '_RollingFirstHalfGoalsConceded_Mean'] = df['FirstHalfGoalsConceded'].rolling(window=window_long,
                                                                                                  min_periods=1).mean().shift(
            1)
        df[prefix + '_RollingFirstHalfGoalsScored_Std'] = df['FirstHalfGoalsScored'].rolling(window=window_long,
                                                                                             min_periods=1).std().shift(
            1)
        df[prefix + '_RollingFirstHalfGoalsConceded_Std'] = df['FirstHalfGoalsConceded'].rolling(window=window_long,
                                                                                                 min_periods=1).std().shift(
            1)
        df[prefix + '_RollingFirstHalfGoalsScored_Mean_Short'] = df['FirstHalfGoalsScored'].rolling(window=window_short,
                                                                                                    min_periods=1).mean().shift(
            1)
        df[prefix + '_Momentum_FirstHalfGoalsScored'] = df[prefix + '_RollingFirstHalfGoalsScored_Mean_Short'] - df[
            prefix + '_RollingFirstHalfGoalsScored_Mean']

        # Function to compute trend slope using simple linear regression
        def compute_slope(x):
            if len(x) < 2:
                return np.nan
            xs = np.arange(len(x))
            return np.polyfit(xs, x, 1)[0]

        df[prefix + '_Trend_Slope_GoalsScored'] = df['GoalsScored'].rolling(window=window_long, min_periods=2).apply(
            compute_slope, raw=True).shift(1)
        df[prefix + '_Trend_Slope_FirstHalfGoalsScored'] = df['FirstHalfGoalsScored'].rolling(window=window_long,
                                                                                              min_periods=2).apply(
            compute_slope, raw=True).shift(1)

        computed_cols = [
            prefix + '_RollingGoalsScored_Mean',
            prefix + '_RollingGoalsConceded_Mean',
            prefix + '_RollingGoalsScored_Std',
            prefix + '_RollingGoalsConceded_Std',
            prefix + '_RollingGoalsScored_Mean_Short',
            prefix + '_Momentum_GoalsScored',
            prefix + '_Trend_Slope_GoalsScored',
            prefix + '_RollingFirstHalfGoalsScored_Mean',
            prefix + '_RollingFirstHalfGoalsConceded_Mean',
            prefix + '_RollingFirstHalfGoalsScored_Std',
            prefix + '_RollingFirstHalfGoalsConceded_Std',
            prefix + '_RollingFirstHalfGoalsScored_Mean_Short',
            prefix + '_Momentum_FirstHalfGoalsScored',
            prefix + '_Trend_Slope_FirstHalfGoalsScored'
        ]
        return df[computed_cols]

    def add_rolling_features_split(group):
        """
        For each team (grouped by country, season, and team), compute:
          - Overall rolling features (all matches)
          - Home-only rolling features (for matches where is_home == 1)
          - Away-only rolling features (for matches where is_home == 0)
        """
        group = group.sort_values(by='date').reset_index(drop=True)
        overall_features = compute_rolling(group, 'Overall')
        group = pd.concat([group, overall_features], axis=1)

        home_mask = group['is_home'] == 1
        if home_mask.sum() > 0:
            group_home = group.loc[home_mask].copy()
            home_features = compute_rolling(group_home, 'Home')
            for col in home_features.columns:
                group.loc[home_mask, col] = home_features[col].values

        away_mask = group['is_home'] == 0
        if away_mask.sum() > 0:
            group_away = group.loc[away_mask].copy()
            away_features = compute_rolling(group_away, 'Away')
            for col in away_features.columns:
                group.loc[away_mask, col] = away_features[col].values

            # ----- Additional Goal Threshold Percentages -----
            # Compute cumulative (season) and rolling (last 5 games) percentages for goals scored over thresholds.
            # ----- Additional Goal Threshold Percentages -----
            # For each threshold, compute season cumulative and rolling percentages.
        for threshold in [1.5, 2.5, 3.5]:
            # ----- Overall (Team as a Whole) -----
            overall_season_col = f'Overall_Percent_Over_{threshold}'
            overall_rolling_col = f'Overall_Rolling5_Percent_Over_{threshold}'
            indicator_overall = group['GoalsScored'].gt(threshold)
            # Using shift(1) to exclude the current match:
            group[overall_season_col] = indicator_overall.shift(1).expanding(min_periods=1).mean()
            group[overall_rolling_col] = indicator_overall.shift(1).rolling(window=5, min_periods=1).mean()

            # ----- Home Matches Only -----
            season_col_home = f'Home_Percent_Over_{threshold}'
            rolling_col_home = f'Home_Rolling5_Percent_Over_{threshold}'
            if home_mask.sum() > 0:
                indicator_home = group.loc[home_mask, 'GoalsScored'].gt(threshold)
                # Compute on the home subset and assign back to the group
                group.loc[home_mask, season_col_home] = indicator_home.shift(1).expanding(min_periods=1).mean()
                group.loc[home_mask, rolling_col_home] = indicator_home.shift(1).rolling(window=5, min_periods=1).mean()

            # ----- Away Matches Only -----
            season_col_away = f'Away_Percent_Over_{threshold}'
            rolling_col_away = f'Away_Rolling5_Percent_Over_{threshold}'
            if away_mask.sum() > 0:
                indicator_away = group.loc[away_mask, 'GoalsScored'].gt(threshold)
                group.loc[away_mask, season_col_away] = indicator_away.shift(1).expanding(min_periods=1).mean()
                group.loc[away_mask, rolling_col_away] = indicator_away.shift(1).rolling(window=5, min_periods=1).mean()

            # ----- Team-specific Match Outcome Percentages -----
            # These features capture the percentage of matches in which the team has scored over a given goal threshold.
        for threshold in [0.5, 1.5, 2.5, 3.5]:
            # Overall (all matches)
            overall_col = f'TeamPct_Over_{threshold}'
            indicator_overall = group['GoalsScored'].gt(threshold)
            # Use shift(1) to exclude the current match from its own calculation.
            group[overall_col] = indicator_overall.shift(1).expanding(min_periods=1).mean()

            # Home matches only
            home_col = f'Home_TeamPct_Over_{threshold}'
            if home_mask.sum() > 0:
                indicator_home = group.loc[home_mask, 'GoalsScored'].gt(threshold)
                group.loc[home_mask, home_col] = indicator_home.shift(1).expanding(min_periods=1).mean()

            # Away matches only
            away_col = f'Away_TeamPct_Over_{threshold}'
            if away_mask.sum() > 0:
                indicator_away = group.loc[away_mask, 'GoalsScored'].gt(threshold)
                group.loc[away_mask, away_col] = indicator_away.shift(1).expanding(min_periods=1).mean()

        return group

    # Apply rolling feature engineering group-wise
    team_df = team_df.groupby(['country', 'season', 'Team'], group_keys=False) \
        .apply(add_rolling_features_split) \
        .reset_index(drop=True)

    # -----------------------------
    # Process Home-Team Features
    # -----------------------------
    home_subset = team_df[team_df['is_home'] == 1].copy()
    home_subset = home_subset.drop(columns=['Opponent'])
    home_subset.rename(columns={'Team': 'home_team'}, inplace=True)
    home_key_cols = ['country', 'season', 'date', 'home_team', 'is_home']
    home_feature_cols = [col for col in home_subset.columns
                         if col not in home_key_cols and (col.startswith("Overall_") or col.startswith("Home_"))]
    home_features = home_subset[home_key_cols + home_feature_cols].copy()
    home_features.rename(columns={col: "home_" + col for col in home_feature_cols}, inplace=True)

    # -----------------------------
    # Process Away-Team Features
    # -----------------------------
    away_subset = team_df[team_df['is_home'] == 0].copy()
    away_subset = away_subset.drop(columns=['Opponent'])
    away_subset.rename(columns={'Team': 'away_team'}, inplace=True)
    away_key_cols = ['country', 'season', 'date', 'away_team', 'is_home']
    away_feature_cols = [col for col in away_subset.columns
                         if col not in away_key_cols and (col.startswith("Overall_") or col.startswith("Away_"))]
    away_features = away_subset[away_key_cols + away_feature_cols].copy()
    away_features.rename(columns={col: "away_" + col for col in away_feature_cols}, inplace=True)

    # -----------------------------
    # Merge Back into the Match-Level DataFrame
    # -----------------------------
    match_df = data.copy()
    match_df = match_df.merge(home_features, on=['country', 'season', 'date', 'home_team'], how='left')
    match_df = match_df.merge(away_features, on=['country', 'season', 'date', 'away_team'], how='left')

    # Clean up and finalise the match-level DataFrame
    match_df.dropna(inplace=True)
    match_df['total_goals'] = match_df['home_goals_ft'] + match_df['away_goals_ft']
    match_df['target'] = match_df['total_goals'].apply(lambda x: 1 if x > 2.5 else 0)

    return match_df

def pre_prepared_data(file_path):
    data = pd.read_csv(file_path,
                       low_memory=False)
    # Convert 'date' column to datetime object
    data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d", errors='coerce')
    data = data.sort_values(by='date')

    # Convert today's date to a pandas Timestamp for compatibility.
    today = pd.Timestamp(datetime.today().date())
    data = data[data['date'] <= today]

    # Clean up and finalise the match-level DataFrame
    data.dropna(inplace=True)
    #data['ht_score'] = data['home_goals_ht'].astype(str) + '-' + data['away_goals_ht'].astype(str)
    # 1 if away does not win (home win or draw), 0 if away wins
    data['target'] = (data['away_goals_ft'] >= data['home_goals_ft']).astype(int)
    return data




def extract_scores(directory: str) -> Tuple[str, ...]:
    """
    Scans `directory` for files named like
        model_metrics_('home-away',)_YYYYMMDD_HHMMSS.csv
    and returns the tuple of all the 'home-away' strings it finds.

    Args:
        directory (str): Path to the folder containing your CSVs.

    Returns:
        Tuple[str, ...]: All the HT score identifiers (e.g. "0-0", "1-2") already modelled.
    """
    pattern = re.compile(r"^model_metrics_\('([^']+)',\)_\d{8}_\d{6}\.csv$")
    search = os.path.join(directory, "model_metrics_*.csv")
    out = []
    for filepath in glob.glob(search):
        name = os.path.basename(filepath)
        m = pattern.match(name)
        if m:
            out.append(m.group(1))
    return tuple(out)

if __name__ == "__main__":
    start = time.time()

    matches = pre_prepared_data(r'C:\Users\leere\PycharmProjects\Football_ML3\engineered_master_data_ALL_2017+.csv')


    # Process each league separately
    leagues = matches[['country']].drop_duplicates().apply(tuple, axis=1)
    # ht_scores = matches[['ht_score']].drop_duplicates().apply(tuple, axis=1)

    matches = pd.get_dummies(matches, columns=['country'], prefix='country')
    dummy_cols = [col for col in matches.columns if col.startswith('country_')]
    features = features + dummy_cols

    # directory = r"C:\Users\leere\PycharmProjects\Football_ML3\Goals\2H_goal\ht_scoreline\best_models_by_ht_scoreline"
    # scoreline_tuple = extract_scores(directory)

    # CLASSIFY
    # fl.run_models_outcome(
    #     matches_filtered=matches,
    #     features=features,
    #     market="LAY_HOME",
    #
    #     # force CLASSIFY (not VALUE)
    #     use_value_for_lay=False,
    #     use_value_for_back=False,
    #
    #     # classify settings (lay side, use home odds for P/L)
    #     classify_side="lay",
    #     classify_odds_column="home_odds",  # REQUIRED for real P/L
    #
    #     # (fast) coarse search — adjust as you like
    #     thresholds=np.round(np.arange(0.20, 0.81, 0.05), 2),
    #     #classify_odds_min_grid=np.array([1.01]),  # collapse band sweep (fast)
    #     #classify_odds_max_grid=np.array([1000.0]),
    #     classify_odds_min_grid=np.round(np.arange(1.20, 6.01, 0.40), 2),
    #     classify_odds_max_grid=np.round(np.arange(1.80, 10.01, 0.40), 2),
    #
    #     # test BOTH lay staking variants during CLASSIFY
    #     classify_lay_flat_stake=1.0,  # flat stake per bet
    #     classify_lay_liability=1.0,  # flat liability per bet
    #
    #     # training/search controls
    #     base_model="xgb", search_mode="random",
    #     n_random_param_sets=5, cpu_jobs=6,
    #     min_samples=400, min_test_samples=400,
    #     precision_test_threshold=0.10,
    #     max_precision_drop=0.05,
    #
    #     # economics & outputs
    #     commission_rate=0.02,
    #     save_bets_csv=True, save_all_bets_csv=True, plot_pl=True
    # )

    # VALUE
    fl.run_models_outcome(
        matches_filtered=matches,
        features=features,
        market="LAY_HOME",

        # force VALUE (LAY)
        use_value_for_lay=True,
        use_value_for_back=False,

        # edge sweep: fair ≥ (1 + edge) × market
        value_edge_grid_lay=np.round(np.arange(0.00, 0.201, 0.01), 2),

        # (optional) search only flat-stake & flat-liability plans
        enable_staking_plan_search=True,
        staking_plan_lay_options=["flat_stake", "liability"],

        # staking parameters / bounds for LAY
        lay_flat_stake=1.0,
        liability_test=1.0,
        min_lay_stake=0.0, max_lay_stake=1.0,
        min_lay_liability=0.0, max_lay_liability=2.0,

        # training/search controls
        base_model="xgb", search_mode="random",
        n_random_param_sets=500, cpu_jobs=6,
        min_samples=400, min_test_samples=400,
        precision_test_threshold=0.10,
        max_precision_drop=0.05,

        # economics & outputs
        commission_rate=0.02,
        save_bets_csv=True, save_all_bets_csv=True, plot_pl=True
    )

    end = time.time()

    elapsed_time = end - start  # Calculate elapsed time in seconds

    # Print the elapsed time in seconds, minutes, and hours:
    print("Elapsed time in seconds: {:.2f}".format(elapsed_time))
    print("Elapsed time in minutes: {:.2f}".format(elapsed_time / 60))
    print("Elapsed time in hours:   {:.2f}".format(elapsed_time / 3600))
