import os
import random as rd
from datetime import datetime
from functools import reduce
from itertools import product
from operator import mul
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import function_library as fl
import traceback

# List of encodings to try
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

# Variables to store data
data1, data2 = None, None

# Attempt to read both files with each encoding
for encoding in encodings:
    try:
        data1 = pd.read_csv("GOAL_DATA_PAST_SEASON.CSV", encoding=encoding, low_memory=False)
        data2 = pd.read_csv("GOAL_DATA_PAST_5.CSV", encoding=encoding, low_memory=False)
        print(f"Successfully read the files with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Failed to decode with encoding: {encoding}")

col_dict_1 = {
    "country": "Country",
    "league": "League",
    "datameci": "Date",
    "etapa": "Round",
    "txtechipa1": "home_team",
    "txtechipa2": "away_team",
    "place1t": "Home_team_place_total",
    "place1a": "Home_team_place_home",
    "place2t": "Away_team_place_total",
    "place2d": "Away_team_place_away",
    "customh": "ELO_home",
    "customa": "ELO_away",
    "custom3": "FORM_home",
    "custom4": "FORM_away",
    "home_val": "home_win",
    "home_val_2": "home_win_15",
    "home_val_3": "home_o25",
    "home_val_4": "home_o35",
    "home_val_5": "home_scored",
    "away_val": "away_win",
    "away_val_2": "away_win_15",
    "away_val_3": "away_o25",
    "away_val_4": "away_o35",
    "away_val_5": "away_scored",
    "scor1": "home_goals",
    "scor2": "away_goals",
    "cotao": "o2.5_odds",
}

col_dict_2 = {
    "country": "Country",
    "league": "League",
    "datameci": "Date",
    "etapa": "Round",
    "txtechipa1": "home_team",
    "txtechipa2": "away_team",
    "place1t": "Home_team_place_total",
    "place1a": "Home_team_place_home",
    "place2t": "Away_team_place_total",
    "place2d": "Away_team_place_away",
    "customh": "ELO_home_past_5",
    "customa": "ELO_away_past_5",
    "custom3": "FORM_home_past_5",
    "custom4": "FORM_away_past_5",
    "home_val": "home_win_past_5",
    "home_val_2": "home_win_15_past_5",
    "home_val_3": "home_o25_past_5",
    "home_val_4": "home_o35_past_5",
    "home_val_5": "home_scored_past_5",
    "away_val": "away_win_past_5",
    "away_val_2": "away_win_15_past_5",
    "away_val_3": "away_o25_past_5",
    "away_val_4": "away_o35_past_5",
    "away_val_5": "away_scored_past_5",
    "scor1": "home_goals",
    "scor2": "away_goals",
    "cotao": "o2.5_odds",
}

data1 = data1.rename(columns=col_dict_1).filter(items=col_dict_1.values())
data2 = data2.rename(columns=col_dict_2).filter(items=col_dict_2.values())

# Merge data1 and data2 on all columns with matching names
data = pd.merge(
    data1,
    data2,
    how='inner'  # Use 'inner' join to keep only matching columns
)

# Convert Date column to datetime type
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Order by date
data = data.sort_values(by='Date')

data = data[data["Round"] >= 8]
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

data['total_goals'] = data['home_goals'] + data['away_goals']
data['over_2.5_goals'] = data['total_goals'].apply(lambda x: 1 if x > 2.5 else 0)

# Create unique tuples across 'Name' and 'City'
leagues = data[['Country', 'League']].drop_duplicates().apply(tuple, axis=1)
no_leagues = len(leagues)
league_counter = 0
runs = 1
# Probability thresholds to test
thresholds = np.arange(0.3, 0.8, 0.01)

data_ready = data.drop(columns=['home_team', 'away_team', 'home_goals', 'away_goals', 'total_goals', 'o2.5_odds','FORM_away_past_5', 'FORM_home_past_5'])
for league in leagues:
    league_counter += 1
    # Initialise counter
    test_counter = 0
    # Prepare to store results
    results = []
    for i in range(0, runs):
        # Format it as a string
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_filtered = data_ready[data_ready[["Country", "League"]].apply(tuple, axis=1) == league]
        if data_filtered.empty:
            pass

        result = fl.generate_train_and_three_test_sets(data_filtered, target_column = "over_2.5_goals")
        # Unpack the dictionary by accessing its keys
        train_data = result["train_data"]
        train_data_scaled = result["train_data_scaled"]
        y_train = result["y_train"]
        test_data1 = result["test_data1"]
        test_data1_scaled = result["test_data1_scaled"]
        y_test1 = result["y_test1"]
        test_data2 = result["test_data2"]
        test_data2_scaled = result["test_data2_scaled"]
        y_test2 = result["y_test2"]
        test_data3 = result["test_data3"]
        test_data3_scaled = result["test_data3_scaled"]
        y_test3 = result["y_test3"]
        test_data_master = result["test_data_master"]
        test_data_master_scaled = result["test_data_master_scaled"]
        y_test_master = result["y_test_master"]
        test_sets = [test_data1, test_data2]
        y_test_sets = [y_test1, y_test2]

        # Get parameter grid
        param_grid = fl.generate_param_grids(i=0, flag=False)  # Replace 0 with your desired value

        # Calculate total combinations
        total_combinations = 0
        for grid in param_grid:
            # Multiply the lengths of each parameter's value list and the number of thresholds
            grid_combinations = reduce(mul, [len(values) for values in grid.values()]) * len(thresholds)
            total_combinations += grid_combinations

        # Iterate over parameter grids
        for grid in param_grid:
            # Extract parameter names and values
            keys, values = zip(*grid.items())
            combinations = list(product(*values))  # Generate all combinations

            # Iterate over each parameter combination
            for combination in combinations:
                params = dict(zip(keys, combination))  # Create a parameter dictionary

                # Extract the model
                model = params.pop('model')  # Extract the model class

                try:
                    # Set parameters
                    model.set_params(**{k.split('__')[1]: v for k, v in params.items()})

                    # Fit the model
                    model.fit(train_data, y_train)

                    # Predict probabilities on test data
                    y_proba1 = model.predict_proba(test_data1_scaled)[:, 1]  # Probabilities for the positive class
                    y_proba2 = model.predict_proba(test_data2_scaled)[:, 1]  # Probabilities for the positive class
                    y_proba3 = model.predict_proba(test_data3_scaled)[:, 1]  # Probabilities for the positive class

                    # Test at different thresholds
                    for threshold in thresholds:
                        test_counter += 1  # Increment counter
                        y_pred1 = (y_proba1 >= threshold).astype(int)
                        y_pred2 = (y_proba2 >= threshold).astype(int)
                        y_pred3 = (y_proba3 >= threshold).astype(int)
                        y_pred_master = (y_proba3 >= threshold).astype(int)
                        y_preds = [y_pred1, y_pred2]

                        # Evaluate the model
                        score1 = accuracy_score(y_test1, y_pred1)
                        score2 = accuracy_score(y_test2, y_pred2)
                        score3 = accuracy_score(y_test3, y_pred3)
                        av_score = (score1 + score2) / 2
                        print(f"League: {league_counter}/{no_leagues}: {league}")
                        print(f"Run: {i + 1}/{runs}")
                        print(
                             f"Test {test_counter}/{total_combinations} | Threshold {threshold:.2f}: Accuracy Score: {av_score}: Params: {params}")

                        p_values = []
                        for j, y_pred in enumerate(y_preds):
                            # Calculate betting results on test data
                            profit_list = []
                            profit = 0
                            for index, pred in enumerate(y_pred):
                                if pred == 1:
                                    # Simulate placing a bet with potential profit or loss
                                    odds = data.iloc[test_sets[j].index[index]]['o2.5_odds']
                                    if y_test_sets[j].iloc[index] == 1:
                                        profit_value = (odds * 1) - 1  # Calculate profit for a win
                                    else:
                                        profit_value = -1  # Lose the bet
                                    profit_list.append(profit_value)
                                    profit += profit_value

                            # Total number of bets placed
                            total_stake = len(profit_list)
                            roi = (profit / total_stake) * 100 if total_stake > 0 else 0

                            # Perform one-sample t-test for the null hypothesis (mean profit < 0)
                            if (total_stake > 25) & (av_score > 0.5):
                                # Perform a one-sample t-test
                                t_stat, p_value = ttest_1samp(profit_list, 0)
                                # Adjust p-value for one-tailed test (H1: mean > 0)
                                if t_stat > 0:
                                    p_value /= 2  # Halve the p-value for one-tailed test
                                else:
                                    p_value = 1.0  # If t_stat <= 0, p-value is 1.0 (no evidence for H1)
                                p_values.append(p_value)
                            else:
                                print(f"  Threshold {threshold:.2f}: Not enough bets placed, skipping profit analysis.")
                        if len(p_values) == 2:
                            if (p_values[0] < 1) and (p_values[1] < 1): #and (p_values[2] <= 1):
                                profit_list = []
                                profit = 0
                                for index, pred in enumerate(y_pred_master):
                                    if pred == 1:
                                        # Simulate placing a bet with potential profit or loss
                                        odds = data.iloc[test_data_master.index[index]]['o2.5_odds']
                                        if y_test_master.iloc[index] == 1:
                                            profit_value = (odds * 1) - 1  # Calculate profit for a win
                                        else:
                                            profit_value = -1  # Lose the bet
                                        profit_list.append(profit_value)
                                        profit += profit_value

                                # Total number of bets placed
                                total_stake = len(profit_list)
                                roi = (profit / total_stake) * 100 if total_stake > 0 else 0
                                t_stat, p_value_master = ttest_1samp(profit_list, 0)
                                # Adjust p-value for one-tailed test (H1: mean > 0)
                                if t_stat > 0:
                                    p_value /= 2  # Halve the p-value for one-tailed test
                                else:
                                    p_value = 1.0  # If t_stat <= 0, p-value is 1.0 (no evidence for H1)

                                if total_stake > 100 and p_value < 1:
                                    results.append((league, av_score, p_value_master, threshold, total_stake, profit, roi, params))
                                    print(f"Results saved temporarily:")
                                    print(f"  Staked: {total_stake}")
                                    print(f"  Profit: £{profit:.2f}")
                                    print(f"  ROI: {roi:.1f}%")
                                    print(f"  p-value: {p_value:.4f}")
                                else:
                                    print(f"results are not good enough")
                        else:
                            print(f"Not enough p-values measured")

                except ValueError as e:
                    # Catch parameter compatibility errors
                    tb = traceback.extract_tb(e.__traceback__)
                    line_number = tb[-1].lineno  # Get the line number of the error
                    print(f"Test {test_counter}: Skipping due to incompatible parameters: {params}. Error: {e} on line {line_number}")
                except Exception as e:
                    # Catch all other errors
                    tb = traceback.extract_tb(e.__traceback__)
                    line_number = tb[-1].lineno  # Get the line number of the error
                    print(f"Test {test_counter}: Unexpected error: {e} on line: {line_number}")

        # Write sorted results to the text file
        if results:
            output_folder = f"Leagues_accuracy_score/{league[0]}_{league[1]}"
            # Ensure the output folder exists
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            # rand_tag = rd.randint(1, 10000)
            output_file = os.path.join(output_folder, f"results_{current_datetime}_{i + 1}.txt")
            results.sort(key=lambda x: x[2], reverse=False)
            results_100 = results[:100]
            with open(output_file, "w") as file:
                for result in results_100:
                    league, score, p_value, threshold, total_stake, profit, roi, params = result
                    file.write(
                        f"League: {league}, Accuracy Score: {score} p-value: {p_value:.4f}, Threshold: {threshold:.2f}, Staked: {total_stake}, Profit: £{profit:.2f}, ROI: {roi:.1f}%, Params: {params}\n")
            print(f"\nResults saved to {output_file}.")
        else:
            print("\nNo profitable results to save.")
