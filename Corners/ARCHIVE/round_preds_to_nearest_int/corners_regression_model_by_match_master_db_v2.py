from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import time

start_time = time.time()  # Record start time
file_name = "master_db_eng1_20_25_by_match"
data = pd.read_csv(rf"/Corners/ARCHIVE/Data\{file_name}.csv")
data = data[data["round"]>7]
data["total_corners"]=data["corners_home"]+data["corners_away"]

# Define the columns to drop
columns_to_drop = [
    "shots_home", "shots_home_1h", "shots_home_2h",
    "shots_away", "shots_away_1h", "shots_away_2h",
    "shots_on_target_home", "shots_on_target_home_1h", "shots_on_target_home_2h",
    "shots_on_target_away", "shots_on_target_away_1h", "shots_on_target_away_2h",
    "corners_home", "corners_home_1h", "corners_home_2h",
    "corners_away", "corners_away_1h", "corners_away_2h",
    "fouls_home", "fouls_home_1h", "fouls_home_2h",
    "fouls_away", "fouls_away_1h", "fouls_away_2h",
    "yellow_cards_home", "yellow_cards_home_1h", "yellow_cards_home_2h",
    "yellow_cards_away", "yellow_cards_away_1h", "yellow_cards_away_2h",
    "possession_home", "possession_home_1h", "possession_home_2h",
    "possession_away", "possession_away_1h", "possession_away_2h",
    "goals_scored_total_home", "goals_conceded_total_home",
    "goals_scored_total_away", "goals_conceded_total_away", "country", "season", "date", "ko_time", "home_team", "away_team", "home_goals_ft", "away_goals_ft", "home_goals_ht", "away_goals_ht", "points_home", "points_away",
]

# Drop the specified columns
data = data.drop(columns=columns_to_drop, errors='ignore')

# Define target and features
y = data["total_corners"]
X = data.drop(columns=["total_corners"], errors="ignore")

# Fill missing values with median
X = X.fillna(X.median())

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    # "RandomForest": {
    #     "model": RandomForestRegressor(random_state=42),
    #     "param_grid": {
    #         'n_estimators': [50, 100, 200],
    #         'max_depth': [10, 20, None],
    #         'min_samples_split': [2, 5],
    #         'min_samples_leaf': [1, 2, 4]
    #     }
    # },
    # "XGBoost": {
    #     "model": XGBRegressor(random_state=42),
    #     "param_grid": {
    #         'n_estimators': [50, 100, 200],
    #         'max_depth': [3, 6, 10],
    #         'learning_rate': [0.01, 0.1, 0.2],
    #         'subsample': [0.8, 1.0],
    #         'colsample_bytree': [0.8, 1.0]
    #     }
    # },
    # "NeuralNetwork": {
    #     "model": MLPRegressor(random_state=42, max_iter=1000),
    #     "param_grid": {
    #         'hidden_layer_sizes': [(50,), (100,)],
    #         'activation': ['relu', 'tanh'],
    #         'solver': ['adam', 'lbfgs'],
    #         'alpha': [0.0001, 0.01, 0.1],
    # #     }
    # },
    "SVR": {
        "model": SVR(),
        "param_grid": {
            'C': [1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'epsilon': [0.1, 0.2],
        }
    },
    # "RandomForest": {
    #     "model": RandomForestRegressor(random_state=42),
    #     "param_grid": {
    #         'n_estimators': [50, 100, 200, 300, 500],  # More estimators
    #         'max_depth': [10, 20, 30, 50, None],  # Deeper trees
    #         'min_samples_split': [2, 5, 10],  # Split criteria
    #         'min_samples_leaf': [1, 2, 4, 8],  # Leaf node size
    #         'max_features': ['auto', 'sqrt', 'log2'],  # Feature selection
    #         'bootstrap': [True, False]  # Bootstrapping technique
    #     }
    # },
    #
    # "XGBoost": {
    #     "model": XGBRegressor(random_state=42),
    #     "param_grid": {
    #         'n_estimators': [50, 100, 200, 300, 500],  # More estimators
    #         'max_depth': [3, 6, 10, 15],  # Deep trees
    #         'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],  # More learning rates
    #         'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # Subsample ratios
    #         'colsample_bytree': [0.5, 0.7, 0.8, 0.9, 1.0],  # Feature selection
    #         'reg_lambda': [0, 0.1, 1, 10],  # L2 Regularization
    #         'reg_alpha': [0, 0.1, 1, 10]  # L1 Regularization
    #     }
    # },
    #
    # "NeuralNetwork": {
    #     "model": MLPRegressor(random_state=42, max_iter=2000),  # Increased iterations
    #     "param_grid": {
    #         'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100), (300, 150)],  # More hidden layers
    #         'activation': ['relu', 'tanh', 'logistic'],  # Add logistic for sigmoid-like function
    #         'solver': ['adam', 'lbfgs', 'sgd'],  # Add stochastic gradient descent (SGD)
    #         'alpha': [0.0001, 0.001, 0.01, 0.1, 1],  # More regularization options
    #         'learning_rate': ['constant', 'adaptive', 'invscaling'],  # More learning strategies
    #         'batch_size': ['auto', 16, 32, 64, 128],  # Mini-batch sizes
    #         'early_stopping': [True, False]  # Enable/disable early stopping
    #     }
    # },
    #
    # "SVR": {
    #     "model": SVR(),
    #     "param_grid": {
    #         'C': [0.1, 1, 10, 100, 1000],  # More values for regularization
    #         'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Additional kernels
    #         'epsilon': [0.01, 0.1, 0.2, 0.5],  # More precision control
    #         'degree': [2, 3, 4, 5],  # Polynomial kernel degrees
    #         'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10]  # Kernel coefficient
    #     }
    # }
}

def no_penalty_under_pred(y_true, y_pred, penalty=1.0):
    """
    Custom loss function that only penalises over-predictions.

    - No penalty if y_true >= y_pred (under-prediction is acceptable).
    - Applies a penalty (default = 1x) to over-predictions.

    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
        penalty (float): Multiplier for penalising over-predictions.

    Returns:
        float: The mean loss.
    """
    y_pred_rounded = np.round(y_pred)  # Round predictions to nearest whole number
    errors = y_pred_rounded - y_true  # Calculate errors
    loss = [
        0 if e <= 0 else penalty * abs(e)  # Penalise only over-predictions
        for e in errors
    ]
    return sum(loss) / len(loss)  # Return average loss

# Create a custom scorer for GridSearchCV
no_penalty_scorer = make_scorer(no_penalty_under_pred, greater_is_better=False)

# Store best models and results
best_models = {}
results = {}

# Run GridSearch for each model
for name, config in models.items():
    try:
        print(f"Training {name} with GridSearchCV...")

        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["param_grid"],
            cv=3,
            scoring=no_penalty_scorer,  # Custom scorer for over-predictions
            n_jobs=-1,
            verbose=10
        )

        grid_search.fit(X_train_scaled, y_train)
        best_models[name] = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"Best parameters for {name}: {best_params}")

        # Predict with the best model
        y_pred = best_models[name].predict(X_test_scaled)
        y_pred = np.round(y_pred)

        # Calculate adjusted metrics (only for over-predictions)
        adjusted_errors = np.where(y_test >= y_pred, 0, y_pred - y_test)
        adjusted_mae = np.mean(adjusted_errors)
        adjusted_rmse = np.sqrt(np.mean(adjusted_errors ** 2))
        adjusted_r2 = 1 - (np.sum(adjusted_errors ** 2) / np.sum((y_test - y_test.mean()) ** 2))

        # Calculate win rate (where predicted ≤ actual)
        win_rate = (y_pred <= y_test).sum() / len(y_test)

        # Store results
        results[name] = {
            "Best Parameters": best_params,
            "Adjusted MAE": adjusted_mae,
            "Adjusted RMSE": adjusted_rmse,
            "Adjusted R2 Score": adjusted_r2,
            "Win Rate": win_rate
        }

        # Save predictions to CSV
        test_results_df = X_test.copy()
        test_results_df["Actual_Corners"] = y_test.values
        test_results_df["Predicted_Corners"] = y_pred
        test_results_df["Difference"] = y_test.values - y_pred
        file_name = f"{name}_predictions.csv"
        test_results_df.to_csv(file_name, index=False)

    except Exception as e:
        print(f"Error training {name}: {e}")

# Convert results to DataFrame
df_results = pd.DataFrame.from_dict(results, orient='index')

# Save summary to CSV
summary_file_path = f"Expanded_Model_Summary_{file_name}"
df_results.to_csv(summary_file_path, index=True)

end_time = time.time()  # Record end time
elapsed_time = end_time - start_time  # Calculate elapsed time

print(f"Program executed in {elapsed_time / 60:.2f} minutes")

