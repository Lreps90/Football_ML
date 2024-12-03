import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from itertools import product
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from itertools import product
from functools import reduce
from operator import mul
from scipy.stats import ttest_1samp
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import random as rd

# List of common encodings to try
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

data = None
for encoding in encodings:
    try:
        data = pd.read_csv("GOAL_DATA.csv", encoding=encoding)
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
    "customh": "o2.5_avg_season",
    "customa": "o2.5_avg_past5",
    "custom3": "o3.5_avg_season",
    "custom4": "o3.5_avg_past5",
    "home_val": "H_ELO_avg",
    "home_val_2": "H_ELO_avg_opp",
    "home_val_3": "H_win_avg",
    "home_val_4": "H_win_1.5_avg",
    "home_val_5": "H_gg_avg",
    "away_val": "A_ELO_avg",
    "away_val_2": "A_ELO_avg_opp",
    "away_val_3": "A_win_avg",
    "away_val_4": "A_win_1.5_avg",
    "away_val_5": "A_gg_avg",
    "scor1": "home_goals",
    "scor2": "away_goals",
    "cotao": "o2.5_odds",
}

data = data.rename(columns=col_dict).filter(items=col_dict.values())
# Convert Date column to datetime type
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Order by date
data = data.sort_values(by='Date')

data = data[data["Round"]>=8]
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

data['total_goals'] = data['home_goals'] + data['away_goals']
data['over_2.5_goals'] = data['total_goals'].apply(lambda x: 1 if x > 2.5 else 0)
data_ready = data.drop(columns=['home_team', 'away_team', 'home_goals', 'away_goals', 'total_goals', 'o2.5_odds'])
# Apply one-hot encoding to 'Country' and 'League'
data_ready = pd.get_dummies(data_ready, columns=['Country', 'League'])

# Split the data into training and testing sets by date
cut_off_date = data_ready['Date'].quantile(0.8)
train_data = data_ready[data_ready['Date'] <= cut_off_date]
y_train = train_data["over_2.5_goals"]
train_data = train_data.drop(columns=['Date', 'over_2.5_goals'])
test_data = data_ready[data_ready['Date'] > cut_off_date]
y_test = test_data["over_2.5_goals"]
test_data = test_data.drop(columns=['Date', 'over_2.5_goals'])
# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data
scaler.fit(train_data)

# Transform the training and testing data
train_data_scaled = scaler.transform(train_data)
test_data_scaled = scaler.transform(test_data)
# Define parameter grid for multiple models
param_grid = [
    # {
    #     'model': [LogisticRegression()],
    #     'model__C': [0.01, 0.1, 1, 10, 100, 1000],  # Adding more fine-grained values
    #     'model__solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],  # Covering all available solvers
    #     'model__penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Additional penalties for regularisation
    #     'model__max_iter': [100, 200, 500, 1000],  # Expanded iteration range
    #     'model__class_weight': [None, 'balanced'],  # Considering imbalanced datasets
    # },
    {
        'model': [RandomForestClassifier()],
        'model__n_estimators': [50, 100, 200],  # Adding larger estimator ranges
        'model__max_depth': [None, 5, 10, 20, 50],  # Including smaller and larger depth values
        'model__min_samples_split': [2, 5, 10, 20],  # Including a wider range of splits
        'model__min_samples_leaf': [1, 2, 4, 10],  # Minimum samples in a leaf node
        'model__max_features': ['sqrt', 'log2', None],  # Number of features to consider for splitting
        'model__bootstrap': [True, False],  # Whether bootstrap samples are used
        'model__class_weight': [None, 'balanced', 'balanced_subsample'],  # Considering class weights
    },
    {
        'model': [MLPClassifier()],
        'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'model__activation': ['relu', 'tanh', 'logistic'],
        'model__solver': ['adam', 'sgd'],
        'model__alpha': [0.0001, 0.001, 0.01],
        'model__learning_rate': ['constant', 'adaptive'],
        'model__max_iter': [1000],
    },

]

# Prepare to store results
results = []
output_folder = "goals"
# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Initialise counter
test_counter = 0

# Probability thresholds to test
thresholds = np.arange(0.5, 0.96, 0.01)

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
                print(f"Test {test_counter}/{total_combinations} | Threshold {threshold:.2f}: Params: {params}, Accuracy: {accuracy}")

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
                roi = (profit / total_stake)*100 if total_stake > 0 else 0

                # Perform one-sample t-test for the null hypothesis (mean profit < 0)
                if total_stake > 0:
                    t_stat, p_value = ttest_1samp(profit_list, 0)  # Null hypothesis: mean profit = 0
                    p_value /= 2  # One-tailed test
                    if (np.mean(profit_list) > 0) & (total_stake > 500):
                        # Add results to the list
                        results.append((p_value, threshold, total_stake, profit, roi, params))
                        print(f"Results saved temporarily:")
                        print(f"  Staked: {total_stake}")
                        print(f"  Profit: £{profit:.2f}")
                        print(f"  ROI: {roi:.1f}%")
                        print(f"  p-value: {p_value:.4f}")
                    else:
                        print(f"  Total Stake: £{total_stake}")
                        print(f"  Net Profit: £{profit:.2f}")
                        print(f"  ROI: {roi:.1f}%")
                        print(f"  p-value: 1.0000")

                    # Write sorted results to the text file
                    if results:
                        # rand_tag = rd.randint(1, 10000)
                        output_file = os.path.join(output_folder, f"results_{test_counter}.txt")
                        results.sort(key=lambda x: x[0])  # Sort by p_value (first element of the tuple)
                        with open(output_file, "w") as file:
                            for result in results:
                                p_value, threshold, total_stake, profit, roi, params = result
                                file.write(
                                    f"p-value: {p_value:.4f}, Threshold: {threshold:.2f}, Staked: {total_stake}, Profit: £{profit:.2f}, ROI: {roi:.1f}%, Params: {params}\n")
                        print(f"\nResults saved to {output_file}.")
                    else:
                        print("\nNo profitable results to save.")

                else:
                    print(f"  Threshold {threshold:.2f}: Not enough bets placed, skipping profit analysis.")

        except ValueError as e:
            # Catch parameter compatibility errors
            print(f"Test {test_counter}: Skipping due to incompatible parameters: {params}. Error: {e}")
        except Exception as e:
            # Catch all other errors
            print(f"Test {test_counter}: Unexpected error: {e}")




