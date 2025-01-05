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

# List of common encodings to try
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

data = None
for encoding in encodings:
    try:
        data = pd.read_csv("GOAL_DATA_TOP_5.csv", encoding=encoding)
        print(f"Successfully read the file with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Failed to decode with encoding: {encoding}")

# if data is not None:
#     # View the data to ensure it was read correctly
#     print(data.head())
# else:
#     print("Unable to read the file with the tested encodings.")

col_dict = {
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

data = data.rename(columns=col_dict).filter(items=col_dict.values())
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
runs = 5
# Probability thresholds to test
thresholds = np.arange(0.5, 0.8, 0.01)

data_ready = data.drop(columns=['home_team', 'away_team', 'home_goals', 'away_goals', 'total_goals', 'o2.5_odds', ])
for league in leagues:
    league_counter += 1
    # Initialise counter
    test_counter = 0
    # Prepare to store results
    results = []
    for i in range(0, runs):
        # Get the current date and time
        current_datetime = datetime.now()
        # Format it as a string
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_filtered = data_ready[data_ready[["Country", "League"]].apply(tuple, axis=1) == league]
        if data_filtered.empty:
            pass
        # Split the data into training and testing sets by date
        cut_off_date = data_filtered['Date'].quantile(0.8)
        train_data = data_filtered[data_filtered['Date'] <= cut_off_date]
        y_train = train_data["over_2.5_goals"]
        train_data = train_data.drop(columns=['Date', 'over_2.5_goals', 'League', 'Country'])
        test_data = data_filtered[data_filtered['Date'] > cut_off_date]
        y_test = test_data["over_2.5_goals"]
        test_data = test_data.drop(columns=['Date', 'over_2.5_goals', 'League', 'Country'])
        # Initialize the scaler
        scaler = StandardScaler()

        # Fit the scaler on the training data
        scaler.fit(train_data)

        # Transform the training and testing data
        train_data_scaled = scaler.transform(train_data)
        test_data_scaled = scaler.transform(test_data)
        # Define parameter grid for multiple models

        # Define ranges for Random Forest parameters
        n_estimators_small = [rd.randint(10, 50), rd.randint(10, 50)]
        n_estimators_medium = [rd.randint(50, 100), rd.randint(50, 100)]
        n_estimators_large = [rd.randint(100, 200), rd.randint(100, 200)]

        max_depth_shallow = [rd.randint(1, 10), rd.randint(1, 10)]
        max_depth_medium = [rd.randint(10, 30), rd.randint(10, 30)]
        max_depth_deep = [rd.randint(30, 50), rd.randint(30, 50)]

        max_features_small = ['sqrt', 'log2', None]  # Considering small set of features
        max_features_large = ['auto', None]  # All features or None

        min_samples_split_small = [rd.randint(2, 10), rd.randint(2, 10)]
        min_samples_split_large = [rd.randint(10, 50), rd.randint(10, 50)]

        min_samples_leaf_small = [rd.randint(1, 5), rd.randint(1, 5)]
        min_samples_leaf_large = [rd.randint(5, 20), rd.randint(5, 20)]

        hidden_layer_small = [rd.randint(0, 100), rd.randint(0, 100), rd.randint(0, 100)]
        hidden_layer_medium = [rd.randint(100, 250), rd.randint(100, 250), rd.randint(100, 250)]
        hidden_layer_large = [rd.randint(250, 500), rd.randint(250, 500), rd.randint(250, 500)]
        hidden_layer_huge = [rd.randint(500, 1000), rd.randint(500, 1000), rd.randint(500, 1000)]
        alpha_tiny = [rd.uniform(0.0001, 0.001), rd.uniform(0.0001, 0.001)]
        alpha_small = [rd.uniform(0.001, 0.01), rd.uniform(0.001, 0.01)]
        alpha_medium = [rd.uniform(0.01, 0.1), rd.uniform(0.01, 0.1)]
        alpha_large = [rd.uniform(0.1, 1), rd.uniform(0.1, 1)]
        alpha_very_large = [rd.uniform(1, 10), rd.uniform(1, 10)]
        if i == 100:
            param_grid_1 = [
                {
                    'model': [LogisticRegression(random_state=42)],
                    'model__C': [0.01, 0.1, 1, 10],  # Adding more fine-grained values
                    'model__solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
                    # Covering all available solvers
                    'model__penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Additional penalties for regularisation
                    'model__max_iter': [500, 1000],  # Expanded iteration range
                    'model__class_weight': [None, 'balanced'],  # Considering imbalanced datasets
                },
                # {
                #     'model': [RandomForestClassifier(random_state=42)],
                #     'model__n_estimators': [50, 100, 200],  # Adding larger estimator ranges
                #     'model__max_depth': [None, 5, 10, 20, 50],  # Including smaller and larger depth values
                #     'model__min_samples_split': [2, 5, 10, 20],  # Including a wider range of splits
                #     'model__min_samples_leaf': [1, 2, 4, 10],  # Minimum samples in a leaf node
                #     'model__max_features': ['sqrt', 'log2', None],  # Number of features to consider for splitting
                #     'model__bootstrap': [True, False],  # Whether bootstrap samples are used
                #     'model__class_weight': [None, 'balanced', 'balanced_subsample'],  # Considering class weights
                # },
                {
                    'model': [MLPClassifier(random_state=42)],
                    'model__hidden_layer_sizes': [(100,), (50, 50)],
                    'model__activation': ['tanh', 'relu', 'tanh', 'logistic'],  #
                    'model__solver': ['adam', 'sgd'],  #
                    'model__alpha': [0.1, 1, 10],
                    'model__learning_rate': ['constant', 'adaptive'],  # ,
                    'model__max_iter': [5000],
                },
            ]
        else:
            param_grid_1 = []
        param_grid_2 = [
            {
                'model': [RandomForestClassifier(random_state=42)],
                'model__n_estimators': [
                    n_estimators_small[0], n_estimators_small[1],
                    n_estimators_medium[0], n_estimators_medium[1],
                    n_estimators_large[0], n_estimators_large[1],
                ],
                'model__max_depth': [
                    max_depth_shallow[0], max_depth_shallow[1],
                    max_depth_medium[0], max_depth_medium[1],
                    max_depth_deep[0], max_depth_deep[1],
                ],
                'model__max_features': max_features_small + max_features_large,
                'model__min_samples_split': [
                    min_samples_split_small[0], min_samples_split_small[1],
                    min_samples_split_large[0], min_samples_split_large[1],
                ],
                'model__min_samples_leaf': [
                    min_samples_leaf_small[0], min_samples_leaf_small[1],
                    min_samples_leaf_large[0], min_samples_leaf_large[1],
                ],
                'model__bootstrap': [True, False],
            },
            {
                'model': [MLPClassifier(random_state=42)],
                'model__hidden_layer_sizes': [
                    (hidden_layer_small[0], hidden_layer_small[1]),
                    (hidden_layer_medium[0], hidden_layer_medium[1]),
                    (hidden_layer_small[0], hidden_layer_medium[1]),
                    (hidden_layer_large[0]),
                    (hidden_layer_large[1]),
                    (hidden_layer_large[0], hidden_layer_medium[0]),
                ],
                'model__activation': ['tanh'],
                'model__solver': ['adam', 'sgd'],
                'model__alpha': [alpha_tiny[0], alpha_small[0], alpha_medium[0], alpha_large[0],
                                 alpha_medium[1], alpha_large[1],
                                 alpha_very_large[0], alpha_very_large[1]],
                'model__learning_rate': ['constant'],
                'model__max_iter': [5000],
            },
        ]
        param_grid = param_grid_1 + param_grid_2

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
                    model.fit(train_data_scaled, y_train)

                    # Predict probabilities on test data
                    y_proba = model.predict_proba(test_data_scaled)[:, 1]  # Probabilities for the positive class

                    # Test at different thresholds
                    for threshold in thresholds:
                        test_counter += 1  # Increment counter
                        y_pred = (y_proba >= threshold).astype(int)  # Apply threshold

                        # Evaluate the model
                        accuracy = accuracy_score(y_test, y_pred)
                        print(f"League: {league_counter}/{no_leagues}: {league}")
                        print(f"Run: {i + 1}/{runs}")
                        print(
                            f"Test {test_counter}/{total_combinations} | Threshold {threshold:.2f}: Params: {params}, Accuracy: {accuracy}")

                        # Calculate betting results on test data
                        profit_list = []
                        profit = 0
                        for index, pred in enumerate(y_pred):
                            if pred == 1:
                                # Simulate placing a bet with potential profit or loss
                                odds = data.iloc[test_data.index[index]]['o2.5_odds']
                                if y_test.iloc[index] == 1:
                                    profit_value = (odds * 1) - 1  # Calculate profit for a win
                                else:
                                    profit_value = -1  # Lose the bet
                                profit_list.append(profit_value)
                                profit += profit_value

                        # Total number of bets placed
                        total_stake = len(profit_list)
                        roi = (profit / total_stake) * 100 if total_stake > 0 else 0

                        # Perform one-sample t-test for the null hypothesis (mean profit < 0)
                        if (total_stake > 100) & (roi > 0):
                            # Perform a one-sample t-test
                            t_stat, p_value = ttest_1samp(profit_list, 0)

                            # Convert to one-tailed p-value
                            # p_value = p_value_two_tailed / 2
                            # if np.mean(profit_list) > 0:
                            # Add results to the list
                            results.append((league, p_value, threshold, total_stake, profit, roi, params))
                            print(f"Results saved temporarily:")
                            print(f"  Staked: {total_stake}")
                            print(f"  Profit: £{profit:.2f}")
                            print(f"  ROI: {roi:.1f}%")
                            print(f"  p-value: {p_value:.4f}")
                        # else:
                        #     print(f"  Total Stake: £{total_stake}")
                        #     print(f"  Net Profit: £{profit:.2f}")
                        #     print(f"  ROI: {roi:.1f}%")
                        #     print(f"  p-value: 1.0000")

                        else:
                            print(f"  Threshold {threshold:.2f}: Not enough bets placed, skipping profit analysis.")


                except ValueError as e:
                    # Catch parameter compatibility errors
                    print(f"Test {test_counter}: Skipping due to incompatible parameters: {params}. Error: {e}")
                except Exception as e:
                    # Catch all other errors
                    print(f"Test {test_counter}: Unexpected error: {e}")

        # Write sorted results to the text file
        if results:
            output_folder = f"Leagues_top_5/{league[0]}_{league[1]}"
            # Ensure the output folder exists
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            # rand_tag = rd.randint(1, 10000)
            output_file = os.path.join(output_folder, f"results_{current_datetime}_{i + 1}.txt")
            results.sort(key=lambda x: x[1], reverse=False)
            results_100 = results[:100]
            with open(output_file, "w") as file:
                for result in results_100:
                    league, p_value, threshold, total_stake, profit, roi, params = result
                    file.write(
                        f"League: {league}, p-value: {p_value:.4f}, Threshold: {threshold:.2f}, Staked: {total_stake}, Profit: £{profit:.2f}, ROI: {roi:.1f}%, Params: {params}\n")
            print(f"\nResults saved to {output_file}.")
        else:
            print("\nNo profitable results to save.")
