import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, make_scorer
from xgboost import XGBRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import os
from xgboost import XGBClassifier
import time


def format_time(seconds):
    """Convert seconds into days, hours, minutes, and seconds for readability."""
    days = seconds // 86400
    seconds %= 86400
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    time_str = []
    if days > 0:
        time_str.append(f"{int(days)} days")
    if hours > 0:
        time_str.append(f"{int(hours)} hours")
    if minutes > 0:
        time_str.append(f"{int(minutes)} minutes")
    time_str.append(f"{seconds:.2f} seconds")

    return ', '.join(time_str)

start_time = time.time()

# Create directory for saving league results
output_dir = "league_goal_data"
os.makedirs(output_dir, exist_ok=True)

cv = 3

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
data.dropna(inplace=True)

data_ready = data.drop(columns=[ 'home_goals', 'away_goals', 'total_goals'])

# Process each league separately
leagues = data[['Country', 'League']].drop_duplicates().apply(tuple, axis=1)

for league in leagues:
    print(league)
    league_data = data_ready[(data_ready['Country'] == league[0]) & (data_ready['League'] == league[1])]

    # Check if the directory exists
    path = f"{output_dir}/{league[0]}_{league[1]}_{cv}.csv"
    if os.path.exists(path):
        print(f"Skipping league {league} as output directory already contains 5 or more files.")
        continue

    if league_data.shape[0] < 100:
        print(f"Skipping {league} due to insufficient data.")
        continue

    train_data = league_data.iloc[:int(0.8 * len(league_data))]
    test_data = league_data.iloc[int(0.8 * len(league_data)):]

    y_train = train_data["over_2.5_goals"]
    y_test = test_data["over_2.5_goals"]
    odds_train = train_data["o2.5_odds"]
    odds_test = test_data["o2.5_odds"]

    original_train_data = train_data.copy()
    original_test_data = test_data.copy()

    train_data = train_data.drop(columns=['Date', 'over_2.5_goals', 'League', 'Country', 'o2.5_odds', 'home_team', 'away_team',])
    test_data = test_data.drop(columns=['Date', 'over_2.5_goals', 'League', 'Country', 'o2.5_odds', 'home_team', 'away_team',])

    # Standardize features
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    # Define parameter grids
    models = {
        # "RandomForest": {
        #     "model": RandomForestClassifier(random_state=42),
        #     "param_grid": {
        #         'n_estimators': [50, 100, 300],  # Removed redundant values, kept a good range
        #         'max_depth': [10, 20, 30, None],  # Removed 50 (unnecessary complexity)
        #         'min_samples_split': [2, 5],  # 10 is rare in practice, removed to reduce combinations
        #         'min_samples_leaf': [1, 2, 4],  # Removed 8 (unlikely to be optimal)
        #         'max_features': ['sqrt'],  # 'log2' is rarely better than 'sqrt'
        #         'bootstrap': [True]  # Bootstrapping is usually better, False rarely helps
        #     }
        # },
            "RandomForest": {
                "model": RandomForestClassifier(random_state=42),
                "param_grid": {
                    'n_estimators': [10, 25, 50, 100, 200],  # Reduced upper limit (500 was excessive)
                    'max_depth': [5, 10, 20],  # Limited depth to prevent overfitting
                    'min_samples_split': [5, 10, 20],  # Increased min splits (discourages deep splits)
                    'min_samples_leaf': [3, 5, 10],  # Larger leaf nodes prevent small splits
                    'max_features': ['sqrt', 0.5],  # Limit features per split (was too flexible)
                    'bootstrap': [True]  # Avoid overfitting with bootstrap sampling
                }
            },

        # "XGBoost": {
        #     "model": XGBClassifier(random_state=42),
        #     "param_grid": {
        #         'n_estimators': [50, 200, 500],  # Removed 100 & 300 (kept meaningful spread)
        #         'max_depth': [3, 6, 10],  # Removed 15 (rarely useful in XGBoost)
        #         'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Removed 0.001, 0.3 (too slow, too aggressive)
        #         'subsample': [0.7, 0.8, 0.9],  # Removed extreme values (0.6 and 1.0)
        #         'colsample_bytree': [0.7, 0.8, 0.9],  # Removed 0.5 (too restrictive)
        #         'reg_lambda': [0.1, 1, 10],  # Removed 0 (almost never best)
        #         'reg_alpha': [0.1, 1],  # Removed extreme values (0, 10)
        #         'objective': ['binary:logistic'],  # Kept binary classification objective
        #         'eval_metric': ['logloss']  # Simplified evaluation
        #     }
        # },
        "XGBoost": {
            "model": XGBClassifier(random_state=42),
            "param_grid": {
                'n_estimators': [50, 100, 200],  # Lowered to prevent overfitting
                'max_depth': [3, 5, 7],  # Reduced to prevent overly complex trees
                'learning_rate': [0.01, 0.05, 0.1],  # Avoid too high LR that causes instability
                'subsample': [0.6, 0.7],  # Reduce sampling to prevent memorization
                'colsample_bytree': [0.6, 0.7],  # Limit features per tree
                'reg_lambda': [1, 5, 10],  # Increased L2 regularization (prevents overfitting)
                'reg_alpha': [0.5, 1, 5],  # Added L1 regularization (forces sparsity)
                'gamma': [0, 0.1, 0.2, 0.3],  # Minimum loss reduction required for a split
                'objective': ['binary:logistic'],  # Binary classification
                'eval_metric': ['logloss']  # Keep log loss as evaluation metric
            }
        },


        "NeuralNetwork": {
            "model": MLPClassifier(random_state=42, max_iter=10000),
            "param_grid": {
                'hidden_layer_sizes': [(25), (25, 25), (50,), (100,), (50, 100)],  # Kept best combinations
                'activation': ['relu', 'tanh'],  # 'tanh' is less reliable than 'relu'
                'solver': ['adam', 'sgd'],  # 'lbfgs' struggles on large datasets
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10],  # Removed 0.1 (rarely optimal)
            }
        },

        "SVM": {
            "model": SVC(probability=True),
            "param_grid": {
                'C': [0.01, 0.1, 1, 10, 100],  # Removed extreme values 0.01 & 100
                'kernel': ['linear', 'rbf'],  # Kept only most effective kernels
                'gamma': ['scale', 'auto', 0.01, 0.1],  # Removed extreme values
            }
        }

    }

    results = []
    for model_name, model_info in models.items():
        print(f"Training {model_name} for {league}...")
        grid_search = GridSearchCV(
            model_info["model"], model_info["param_grid"],
            scoring=make_scorer(accuracy_score),
            cv=cv, n_jobs=-1, verbose=10, return_train_score=True
        )
        grid_search.fit(train_data_scaled, y_train)

        # Retrieve top 100 parameter sets
        cv_results = pd.DataFrame(grid_search.cv_results_)
        top_results = cv_results.sort_values(by='mean_test_score', ascending=False)

        for idx, (_, row) in enumerate(top_results.iterrows()):
            print(f"⚙️ Training best model #{idx + 1}/{len(top_results)} for {model_name}...")

            # Get best model and update parameters
            best_model = grid_search.best_estimator_
            best_model.set_params(**row['params'])
            best_model.fit(train_data_scaled, y_train)

            train_preds = best_model.predict(train_data_scaled)
            test_preds = best_model.predict(test_data_scaled)

            train_accuracy = accuracy_score(y_train, train_preds)
            test_accuracy = accuracy_score(y_test, test_preds)
            train_test_diff = abs(train_accuracy - test_accuracy)

            train_profit = [(odds_train.iloc[i] - 1) if pred == 1 and y_train.iloc[i] == 1 else (0 if pred == 0 else -1) for i, pred in enumerate(train_preds)]
            test_profit = [(odds_test.iloc[i] - 1) if pred == 1 and y_test.iloc[i] == 1 else (0 if pred == 0 else -1) for i, pred in enumerate(test_preds)]

            train_roi = (sum(train_profit) / len(train_profit)) * 100 if train_profit else 0
            test_roi = (sum(test_profit) / len(test_profit)) * 100 if test_profit else 0

            p_value_train = ttest_1samp(train_profit, 0).pvalue if sum(train_profit) > 0 and len(train_profit) > 50 else 1
            p_value_test = ttest_1samp(test_profit, 0).pvalue if sum(test_profit) > 0 and len(test_profit) > 50 else 1

            num_bets_train = sum(train_preds)
            num_bets_test = sum(test_preds)
            total_profit_train = sum(train_profit)
            total_profit_test = sum(test_profit)

            if (train_test_diff < 0.1) & (p_value_train < 1) & (p_value_test < 1):
                results.append([
                    model_name, row['params'], train_accuracy, test_accuracy, train_test_diff,
                    p_value_train, p_value_test, train_roi, test_roi,
                    num_bets_train, num_bets_test, total_profit_train, total_profit_test
                ])



            # # Add predictions, profit/loss, and odds to train and test datasets
            # original_train_data["Predicted"] = train_preds
            # original_train_data["Profit_Loss"] = train_profit
            # original_test_data["Predicted"] = test_preds
            # original_test_data["Profit_Loss"] = test_profit
            #
            # # Ensure home and away team data are included
            # original_train_data["home_team"] = league_data.iloc[:int(0.8 * len(league_data))]["home_team"].values
            # original_train_data["away_team"] = league_data.iloc[:int(0.8 * len(league_data))]["away_team"].values
            # original_test_data["home_team"] = league_data.iloc[int(0.8 * len(league_data)):]["home_team"].values
            # original_test_data["away_team"] = league_data.iloc[int(0.8 * len(league_data)):]["away_team"].values
            # original_train_data["o2.5_odds"] = league_data.iloc[:int(0.8 * len(league_data))]["o2.5_odds"].values
            # original_test_data["o2.5_odds"] = league_data.iloc[int(0.8 * len(league_data)):]["o2.5_odds"].values
            #
            # # Save train and test datasets with predictions
            # train_filename = os.path_league.join(output_dir, f"{league[0]}_{league[1]}_{model_name}_train.csv")
            # test_filename = os.path_league.join(output_dir, f"{league[0]}_{league[1]}_{model_name}_test.csv")
            # original_train_data.to_csv(train_filename, index=False)
            # original_test_data.to_csv(test_filename, index=False)
            # print(f"Saved train data to {train_filename} and test data to {test_filename}")

    results_df = pd.DataFrame(results, columns=[
        'Model', 'Parameters', 'Train Accuracy', 'Test Accuracy',
        'Train-Test Difference', 'Train P-Value', 'Test P-Value',
        'Train ROI', 'Test ROI', 'Bets Placed (Train)', 'Bets Placed (Test)',
        'Total Profit (Train)', 'Total Profit (Test)'
    ])

    results_df = results_df.sort_values(by='Train-Test Difference', ascending=True)
    league_filename = os.path.join(output_dir, f"{league[0]}_{league[1]}_{cv}.csv")
    results_df.to_csv(league_filename, index=False)
    print(f"Saved results for {league} to {league_filename}")

# End time
end_time = time.time()
total_time = end_time - start_time

# Print formatted time
print(f"⏳ Total time taken: {format_time(total_time)}")

