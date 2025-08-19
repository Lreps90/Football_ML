from datetime import datetime
import pandas as pd
import time
import function_library as fl

features = [
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
    'away_Trend_Slope_ShotsOnTarget_1h'
    ]


def prepare_data(file_path):
    data = None
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            print(f"Successfully read the file with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            print(f"Failed to decode with encoding: {encoding}")

    if data is None:
        raise ValueError("Could not read the file with any of the provided encodings.")

    # Convert 'date' column to datetime object
    data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d", errors='coerce')
    data = data.sort_values(by='date')

    # Convert today's date to a pandas Timestamp for compatibility.
    today = pd.Timestamp(datetime.today().date())
    data = data[data['date'] <= today]

    data['total_corners'] = data['corners_home'] + data['corners_away']
    data['target'] = data['total_corners'].apply(lambda x: 1 if x > 9.5 else 0)

    return data

if __name__ == "__main__":
    start = time.time()

    matches = prepare_data(r"C:\Users\leere\PycharmProjects\Football_ML3\engineered_master_data_2014.csv")

    # Process each league separately
    leagues = matches[['country']].drop_duplicates().apply(tuple, axis=1)

    for league in leagues:
        print(league)
        matches_filtered = matches[(matches['country'] == league[0])]
        fl.run_models(matches_filtered, features, league, min_samples=100)

    end = time.time()

    elapsed_time = end - start  # Calculate elapsed time in seconds

    # Print the elapsed time in seconds, minutes, and hours:
    print("Elapsed time in seconds: {:.2f}".format(elapsed_time))
    print("Elapsed time in minutes: {:.2f}".format(elapsed_time / 60))
    print("Elapsed time in hours:   {:.2f}".format(elapsed_time / 3600))


