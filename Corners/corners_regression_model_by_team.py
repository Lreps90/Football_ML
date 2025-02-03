import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import os
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
# List of common encodings to try
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

data = None
for encoding in encodings:
    try:
        data = pd.read_csv("Data/all_21_25_for_viz.csv", encoding=encoding, low_memory=False)
        print(f"Successfully read the file with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Failed to decode with encoding: {encoding}")


# Select relevant features for prediction
features = [
    "Rolling avg ball possession (Total)_3",
    "Rolling avg ball possession conceded (Total)_3",
    "Rolling avg points won (Total)_3",
    "elo difference",
    "Rolling avg corners (Total)_3",
    "Rolling avg shots (Total)_3",
    "Rolling avg shots on target (Total)_3",
    "Rolling avg corners conceded (Total)_3",
    "Team elo",
    "Opponent elo",
    "Rolling avg ball possession (Total)_5",
    "Rolling avg ball possession conceded (Total)_5",
    "Rolling avg points won (Total)_5",
    "Rolling avg corners (Total)_5",
    "Rolling avg shots (Total)_5",
    "Rolling avg shots on target (Total)_5",
    "Rolling avg corners conceded (Total)_5",
    "Rolling avg ball possession (Total)_7",
    "Rolling avg ball possession conceded (Total)_7",
    "Rolling avg points won (Total)_7",
    "Rolling avg corners (Total)_7",
    "Rolling avg shots (Total)_7",
    "Rolling avg shots on target (Total)_7",
    "Rolling avg corners conceded (Total)_7",
]

# Target variable
target = "Total match corners"

# Keep only the features and the target column
data = data[features + ["Season", "Round", "Team", target]]

# Shift rolling averages for each team
rolling_avg_features = [
    "Rolling avg ball possession (Total)_3",
    "Rolling avg ball possession conceded (Total)_3",
    "Rolling avg points won (Total)_3",
    "Rolling avg corners (Total)_3",
    "Rolling avg shots (Total)_3",
    "Rolling avg shots on target (Total)_3",
    "Rolling avg corners conceded (Total)_3",
    "Rolling avg ball possession (Total)_5",
    "Rolling avg ball possession conceded (Total)_5",
    "Rolling avg points won (Total)_5",
    "Rolling avg corners (Total)_5",
    "Rolling avg shots (Total)_5",
    "Rolling avg shots on target (Total)_5",
    "Rolling avg corners conceded (Total)_5",
    "Rolling avg ball possession (Total)_7",
    "Rolling avg ball possession conceded (Total)_7",
    "Rolling avg points won (Total)_7",
    "Rolling avg corners (Total)_7",
    "Rolling avg shots (Total)_7",
    "Rolling avg shots on target (Total)_7",
    "Rolling avg corners conceded (Total)_7",
]
data[rolling_avg_features] = data.groupby(["Team", "Season"])[rolling_avg_features].shift(1)

# Drop rows with missing values caused by shifting
data = data.dropna()

# Define features (X) and target (y)
X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialise the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data only
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Define a custom scoring function
def over_corners_scorer(y_true, y_pred):
    return (y_pred < y_true).sum() / len(y_true)


def asymmetric_loss(y_true, y_pred, penalty=0.5):
    """
    Custom loss function to penalise over-predictions more heavily than under-predictions.

    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
        penalty (float): Multiplier for penalising over-predictions.

    Returns:
        float: The asymmetric loss.
    """
    errors = y_pred - y_true  # Calculate prediction errors
    loss = [
        abs(e) if e <= 0 else penalty * abs(e)  # Apply penalty only to over-predictions
        for e in errors
    ]
    return sum(loss) / len(loss)  # Return the average loss


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

# Create a custom scorer for use in GridSearchCV
asymmetric_scorer = make_scorer(asymmetric_loss, greater_is_better=False)

# Create the scorer
under_penalty_scorer = make_scorer(over_corners_scorer, greater_is_better=True)

#mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

def mae_no_penalty_under_pred(y_true, y_pred):
    """
    Custom MAE function that does not penalise under-predictions.

    - If y_pred < y_true (under-prediction), no penalty (error = 0).
    - If y_pred > y_true (over-prediction), compute absolute error as usual.

    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        float: The adjusted MAE.
    """
    y_pred_rounded = np.round(y_pred)  # Round predictions to nearest whole number
    errors = np.maximum(y_pred_rounded - y_true, 0)  # Keep error only for over-predictions
    return np.mean(errors)  # Return the mean error


models = {
    # "LinearRegression": {
    #     "model": LinearRegression(),
    #     "param_grid": {
    #         "fit_intercept": [True, False]
    #     }
    # },
    # "Ridge": {
    #     "model": Ridge(),
    #     "param_grid": {
    #         "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],  # Wider range
    #         "solver": ["auto", "lsqr", "saga", "svd", "cholesky"],  # More solvers
    #     }
    # },
    # "Lasso": {
    #     "model": Lasso(),
    #     "param_grid": {
    #         "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],  # Wider range
    #         "max_iter": [1000, 2000, 5000],  # Increase iterations for convergence
    #     }
    # },
    # "ElasticNet": {
    #     "model": ElasticNet(),
    #     "param_grid": {
    #         "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],  # Wider range
    #         "l1_ratio": [0.01, 0.1, 0.5, 0.7, 0.9, 0.99],  # More fine-grained values
    #         "max_iter": [1000, 2000, 5000],  # Increase iterations
    #     }
    # },
    # "PolynomialRegression": {
    #     "model": Pipeline([
    #         ('poly', PolynomialFeatures()),
    #         ('linear', LinearRegression())
    #     ]),
    #     "param_grid": {
    #         "poly__degree": [2, 3],
    #         "linear__fit_intercept": [True, False]
    #     }
    # },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42),
        "param_grid": {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42),
        "param_grid": {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    },
    "NeuralNetwork": {
        "model": MLPRegressor(random_state=42, max_iter=1000),
        "param_grid": {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.01, 0.1],
        }
    },
    "SVR": {
        "model": SVR(),
        "param_grid": {
            'C': [1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'epsilon': [0.1, 0.2],
        }
    },
}

best_models = {}

folder_name = "model_results_penalise_under"

# Create a directory to save the model results
os.makedirs(folder_name, exist_ok=True)

# Initialize a summary DataFrame to save results
summary_results = []

# Train and evaluate each model
for model_name, model_info in models.items():
    print(f"\nTraining {model_name}...")

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=model_info["model"],
        param_grid=model_info["param_grid"],
        cv=3,
        scoring=under_penalty_scorer,
        n_jobs=-1,
        verbose=10
    )

    # Fit the model
    grid_search.fit(X_train_scaled, y_train)
    best_models[model_name] = grid_search.best_estimator_

    # Make predictions
    y_pred = grid_search.best_estimator_.predict(X_test_scaled)

    # Apply the custom MAE calculation
    mae = mae_no_penalty_under_pred(y_test, y_pred)

    over_corner_wins = (y_pred < y_test).sum()
    total_matches = len(y_test)
    win_rate = over_corner_wins / total_matches

    # Print evaluation results
    print(f"{model_name} Best Hyperparameters: {grid_search.best_params_}")
    print(f"{model_name} CV Score: {grid_search.best_score_:.4f}")
    print(f"{model_name} Over Corner Market Win Rate: {win_rate:.2%}")
    print(f"{model_name} Mean Absolute Error (MAE): {mae:.4f}")

    # Prepare and append the summary row
    summary_results.append({
        "Model": model_name,
        "Best Hyperparameters": str(grid_search.best_params_),  # Convert dict to string for CSV
        "CV Score:": str(grid_search.best_score_),
        "Win Rate": f"{win_rate:.2%}",  # Save as percentage for readability
        "MAE": mae
    })

    # Create a summary DataFrame
    summary_df = pd.DataFrame(summary_results)

    # Sort the DataFrame by MAE in ascending order
    summary_df = summary_df.sort_values(by="MAE", ascending=True)

    # Save the summary results to a CSV
    summary_file = f"{folder_name}/summary_results_{model_name}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary results saved to {summary_file}")

    # Create a DataFrame with actual values, predictions, and evaluation metrics
    prediction_df = pd.DataFrame({
        "Season": data.loc[y_test.index, "Season"].values,
        "Round": data.loc[y_test.index, "Round"].values,
        "Team": data.loc[y_test.index, "Team"].values,
        "Actual Total Corners": y_test.values,
        "Predicted Total Corners": np.round(y_pred),
        "Mean Absolute Error (MAE)": mae,  # Add MAE to every row for reference
        "Win Rate": win_rate,  # Add win rate to every row for reference
        "CV Score":grid_search.best_score_  # Add CV score for reference
    })

    # Save the predictions to a CSV file
    prediction_file = f"{folder_name}/predictions_{model_name}.csv"
    prediction_df.to_csv(prediction_file, index=False)
    print(f"\nPredictions saved to {prediction_file}")

