import pandas as pd
from datetime import datetime
import numpy as np
import time
import function_library as fl

features = [
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
    'away_Away_Trend_Slope_FirstHalfGoalsScored'
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
    data['date'] = pd.to_datetime(data['date'], format="%d/%m/%Y", errors='coerce')
    data = data.sort_values(by='date')
    today = datetime.today().date()
    data = data[data['date'].dt.date <= today]

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
        df[prefix + '_RollingGoalsScored_Mean'] = df['GoalsScored'].rolling(window=window_long, min_periods=1).mean().shift(1)
        df[prefix + '_RollingGoalsConceded_Mean'] = df['GoalsConceded'].rolling(window=window_long, min_periods=1).mean().shift(1)
        df[prefix + '_RollingGoalsScored_Std'] = df['GoalsScored'].rolling(window=window_long, min_periods=1).std().shift(1)
        df[prefix + '_RollingGoalsConceded_Std'] = df['GoalsConceded'].rolling(window=window_long, min_periods=1).std().shift(1)
        df[prefix + '_RollingGoalsScored_Mean_Short'] = df['GoalsScored'].rolling(window=window_short, min_periods=1).mean().shift(1)
        df[prefix + '_Momentum_GoalsScored'] = df[prefix + '_RollingGoalsScored_Mean_Short'] - df[prefix + '_RollingGoalsScored_Mean']

        # First-Half Goals Rolling Features
        df[prefix + '_RollingFirstHalfGoalsScored_Mean'] = df['FirstHalfGoalsScored'].rolling(window=window_long, min_periods=1).mean().shift(1)
        df[prefix + '_RollingFirstHalfGoalsConceded_Mean'] = df['FirstHalfGoalsConceded'].rolling(window=window_long, min_periods=1).mean().shift(1)
        df[prefix + '_RollingFirstHalfGoalsScored_Std'] = df['FirstHalfGoalsScored'].rolling(window=window_long, min_periods=1).std().shift(1)
        df[prefix + '_RollingFirstHalfGoalsConceded_Std'] = df['FirstHalfGoalsConceded'].rolling(window=window_long, min_periods=1).std().shift(1)
        df[prefix + '_RollingFirstHalfGoalsScored_Mean_Short'] = df['FirstHalfGoalsScored'].rolling(window=window_short, min_periods=1).mean().shift(1)
        df[prefix + '_Momentum_FirstHalfGoalsScored'] = df[prefix + '_RollingFirstHalfGoalsScored_Mean_Short'] - df[prefix + '_RollingFirstHalfGoalsScored_Mean']

        # Function to compute trend slope using simple linear regression
        def compute_slope(x):
            if len(x) < 2:
                return np.nan
            xs = np.arange(len(x))
            return np.polyfit(xs, x, 1)[0]

        df[prefix + '_Trend_Slope_GoalsScored'] = df['GoalsScored'].rolling(window=window_long, min_periods=2).apply(compute_slope, raw=True).shift(1)
        df[prefix + '_Trend_Slope_FirstHalfGoalsScored'] = df['FirstHalfGoalsScored'].rolling(window=window_long, min_periods=2).apply(compute_slope, raw=True).shift(1)

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



if __name__ == "__main__":
    start = time.time()

    matches = prepare_data(r"C:\Users\leere\PycharmProjects\Football_ML3\Goals\cgmbetdatabase_top_5_2020+.csv")

    # Process each league separately
    leagues = matches[['country']].drop_duplicates().apply(tuple, axis=1)

    for league in leagues:
        print(league)
        matches_filtered = matches[(matches['country'] == league[0])]
        fl.run_models(matches_filtered, features, league, apply_pca=False)

    end = time.time()

    elapsed_time = end - start  # Calculate elapsed time in seconds

    # Print the elapsed time in seconds, minutes, and hours:
    print("Elapsed time in seconds: {:.2f}".format(elapsed_time))
    print("Elapsed time in minutes: {:.2f}".format(elapsed_time / 60))
    print("Elapsed time in hours:   {:.2f}".format(elapsed_time / 3600))