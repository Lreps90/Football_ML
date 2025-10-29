# import datetime
import random as rd
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_score
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score
from xgboost import XGBClassifier
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("XGBOOST_VERBOSITY", "0")

from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score
)
from sklearn.neural_network import MLPClassifier

try:
    import xgboost as xgb

    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def create_import_file(df, output_file_path, provider="", market_name="", selection_name=""):
    """
    Creates a CSV file with match details formatted for import.

    Args:
        df (pd.DataFrame): DataFrame containing 'home_team' and 'away_team' columns.
        output_file_path (str): Path to save the output CSV file.
        provider (str): Name of the provider. Default is an empty string.
        market_name (str): Name of the market. Default is an empty string.
        selection_name (str): Name of the selection. Default is an empty string.
    """
    try:
        # Create Match column
        match_df = df['home_team'] + ' v ' + df['away_team']

        # Create the output DataFrame
        output_df = pd.DataFrame({
            'EventName': match_df,
            'Provider': provider,
            'MarketName': market_name,
            'SelectionName': selection_name
        })

        # Save to CSV file
        output_df.to_csv(output_file_path, index=False)

        # Print success message
        print(f"File created and saved successfully at: {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


def team_name_map(dataframe, home_col="home_team", away_col="away_team"):
    team_name_dict_csv = r"C:\Users\leere\PycharmProjects\Football_ML3\team_name_map.csv"
    team_name_mapping = pd.read_csv(team_name_dict_csv)
    # Convert the mapping into a dictionary
    mapping_dict = dict(zip(team_name_mapping["Original Name"], team_name_mapping["Mapped Name"]))
    dataframe[home_col] = dataframe[home_col].map(mapping_dict).fillna(dataframe[home_col])
    dataframe[away_col] = dataframe[away_col].map(mapping_dict).fillna(dataframe[away_col])
    return dataframe


def generate_param_grids(i, flag=True):
    # Define random ranges for parameters
    n_estimators_small = [rd.randint(10, 50), rd.randint(10, 50)]
    n_estimators_medium = [rd.randint(50, 100), rd.randint(50, 100)]
    n_estimators_large = [rd.randint(100, 200), rd.randint(100, 200)]

    max_depth_shallow = [rd.randint(1, 10), rd.randint(1, 10)]
    max_depth_medium = [rd.randint(10, 30), rd.randint(10, 30)]
    max_depth_deep = [rd.randint(30, 50), rd.randint(30, 50)]

    max_features_small = ['sqrt', 'log2', None]
    max_features_large = ['auto', None]

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

    # Define parameter grids
    if i == 0:
        param_grid_1 = [
            {
                'model': [RandomForestClassifier(random_state=42)],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 5, 10, 20, 50],
                'model__min_samples_split': [2, 5, 10, 20],
                'model__min_samples_leaf': [1, 2, 4, 10],
                'model__max_features': ['sqrt', 'log2', None],
                'model__bootstrap': [True, False],
                'model__class_weight': [None, 'balanced', 'balanced_subsample'],
            },
            {
                'model': [MLPClassifier(random_state=42)],
                'model__hidden_layer_sizes': [(100,), (100, 50), (150, 100, 50)],
                'model__activation': ['tanh', 'relu'],
                'model__solver': ['adam', 'sgd'],
                'model__alpha': [0.01, 0.1, 1],
                'model__learning_rate': ['constant', 'adaptive'],
                'model__max_iter': [5000],
            },
        ]
    else:
        param_grid_1 = []

    if flag:
        param_grid_2 = [
            {
                'model': [RandomForestClassifier(random_state=42)],
                'model__n_estimators': (
                        n_estimators_small + n_estimators_medium + n_estimators_large
                ),
                'model__max_depth': (
                        max_depth_shallow + max_depth_medium + max_depth_deep
                ),
                'model__max_features': max_features_small + max_features_large,
                'model__min_samples_split': (
                        min_samples_split_small + min_samples_split_large
                ),
                'model__min_samples_leaf': (
                        min_samples_leaf_small + min_samples_leaf_large
                ),
                'model__bootstrap': [True, False],
            },
            {
                'model': [MLPClassifier(random_state=42)],
                'model__hidden_layer_sizes': [
                    (hidden_layer_small[0], hidden_layer_small[1]),
                    (hidden_layer_medium[0], hidden_layer_medium[1]),
                    (hidden_layer_large[0],),
                    (hidden_layer_large[1],),
                    (hidden_layer_huge[0], hidden_layer_huge[1]),
                ],
                'model__activation': ['tanh', 'relu'],
                'model__solver': ['adam', 'sgd'],
                'model__alpha': (
                        alpha_tiny + alpha_small + alpha_medium + alpha_large + alpha_very_large
                ),
                'model__learning_rate': ['constant'],
                'model__max_iter': [5000],
            },
        ]
    else:
        param_grid_2 = []

    # Combine parameter grids
    return param_grid_1 + param_grid_2


def generate_train_and_three_test_sets(data_filtered, target_column):
    # Ensure the data is sorted by Date
    data_filtered = data_filtered.sort_values(by='Date')

    # Calculate cut-off dates for splitting
    cut_off_date1 = data_filtered['Date'].quantile(0.7)  # 70% for training
    cut_off_date2 = data_filtered['Date'].quantile(0.8)  # Next 10% for test set 1
    cut_off_date3 = data_filtered['Date'].quantile(0.9)  # Next 10% for test set 2

    # Split into training and test sets
    train_data = data_filtered[data_filtered['Date'] <= cut_off_date1]
    y_train = train_data[target_column]
    train_data = train_data.drop(columns=['Date', target_column, 'League', 'Country'])

    test_data1 = data_filtered[
        (data_filtered['Date'] > cut_off_date1) & (data_filtered['Date'] <= cut_off_date2)
        ]
    y_test1 = test_data1[target_column]
    test_data1 = test_data1.drop(columns=['Date', target_column, 'League', 'Country'])

    test_data2 = data_filtered[
        (data_filtered['Date'] > cut_off_date2) & (data_filtered['Date'] <= cut_off_date3)
        ]
    y_test2 = test_data2[target_column]
    test_data2 = test_data2.drop(columns=['Date', target_column, 'League', 'Country'])

    test_data3 = data_filtered[data_filtered['Date'] > cut_off_date3]
    y_test3 = test_data3[target_column]
    test_data3 = test_data3.drop(columns=['Date', target_column, 'League', 'Country'])

    test_data_master = data_filtered[data_filtered['Date'] > cut_off_date1]
    y_test_master = test_data_master[target_column]
    test_data_master = test_data_master.drop(columns=['Date', target_column, 'League', 'Country'])

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler on the training data
    scaler.fit(train_data)

    # Scale the training and test data
    train_data_scaled = scaler.transform(train_data)
    test_data1_scaled = scaler.transform(test_data1)
    test_data2_scaled = scaler.transform(test_data2)
    test_data3_scaled = scaler.transform(test_data3)
    test_data_master_scaled = scaler.transform(test_data_master)

    return {
        "train_data": train_data,
        "train_data_scaled": train_data_scaled,
        "y_train": y_train,
        "test_data1": test_data1,
        "test_data1_scaled": test_data1_scaled,
        "y_test1": y_test1,
        "test_data2": test_data2,
        "test_data2_scaled": test_data2_scaled,
        "y_test2": y_test2,
        "test_data3": test_data3,
        "test_data3_scaled": test_data3_scaled,
        "y_test3": y_test3,
        "test_data_master": test_data_master,
        "test_data_master_scaled": test_data_master_scaled,
        "y_test_master": y_test_master,
    }


def select_optimal_pca_components(X, variance_threshold=0.95):
    # """
    # Fit PCA on X and return the smallest number of components needed
    # to capture at least `variance_threshold` (default 90%) of the variance.
    # """

    # Standardize the features.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)

    # Perform PCA with all components.
    pca = PCA()
    pca.fit(scaled_features)

    # Calculate explained variance.
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Determine optimal number of components (e.g., components to explain 95% of variance).
    optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    # print(f"Optimal number of components to explain 95% variance: {optimal_components}")
    return optimal_components


def build_pipelines(apply_pca=True):
    """
    Build model pipelines and parameter grids.
    If apply_pca is True, each pipeline includes a PCA step.
    Returns:
      pipelines: a dict of pipelines keyed by model name.
      param_grids: a dict of parameter grids keyed by model name.
    """
    pipelines = dict()
    param_grids = dict()

    if apply_pca:
        # XGBoost pipeline with PCA
        pipelines['XGBoost'] = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(svd_solver='randomized', random_state=42)),
            ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
        ])
        param_grids['XGBoost'] = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.01, 0.05],
            'classifier__subsample': [0.7, 0.8],
            'classifier__colsample_bytree': [0.7, 0.8],
            'classifier__gamma': [0, 0.1],
            'classifier__min_child_weight': [1, 3],
            'classifier__reg_lambda': [1, 5],
            'classifier__reg_alpha': [0, 0.1]
        }

        # RandomForest
        pipelines['RandomForest'] = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(svd_solver='randomized', random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])
        param_grids['RandomForest'] = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__bootstrap': [True],
            'classifier__criterion': ['gini'],
            'classifier__max_features': ['sqrt']
        }

        # MLP
        pipelines['MLP'] = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(svd_solver='randomized', random_state=42)),
            ('classifier', MLPClassifier(random_state=42, max_iter=10000))
        ])
        param_grids['MLP'] = {
            'classifier__hidden_layer_sizes': [(100,), (100, 50), (250,), (250, 100)],
            'classifier__activation': ['relu'],
            'classifier__solver': ['adam'],
            'classifier__alpha': [0.00001, 0.0001, 0.001],
            'classifier__learning_rate': ['constant', 'adaptive'],
            'classifier__early_stopping': [True],
            'classifier__batch_size': [32, 64]
        }

        # # SVC
        # pipelines['SVC'] = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('pca', PCA(svd_solver='randomized', random_state=42)),
        #     ('classifier', SVC(probability=True))
        # ])
        # param_grids['SVC'] = {
        #     'classifier__C': [0.1, 1, 10],
        #     'classifier__kernel': ['rbf', 'linear'],
        #     'classifier__gamma': ['scale', 'auto']
        # }

        # # Logistic Regression
        # pipelines['LogisticRegression'] = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('pca', PCA(svd_solver='randomized', random_state=42)),
        #     ('classifier', LogisticRegression(max_iter=10000))
        # ])
        # param_grids['LogisticRegression'] = {
        #     'classifier__C': [0.1, 1, 10],
        #     'classifier__solver': ['lbfgs', 'saga']
        # }
        #
        # # K-Nearest Neighbours
        # pipelines['KNN'] = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('pca', PCA(svd_solver='randomized', random_state=42)),
        #     ('classifier', KNeighborsClassifier())
        # ])
        # param_grids['KNN'] = {
        #     'classifier__n_neighbors': [3, 5, 7],
        #     'classifier__weights': ['uniform', 'distance']
        # }
        #
        # # AdaBoost
        # pipelines['AdaBoost'] = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('pca', PCA(svd_solver='randomized', random_state=42)),
        #     ('classifier', AdaBoostClassifier(random_state=42, algorithm='SAMME'))
        # ])
        # param_grids['AdaBoost'] = {
        #     'classifier__n_estimators': [50, 100],
        #     'classifier__learning_rate': [0.5, 1.0]
        # }
        #
        # # Stacking Ensembles
        # ensemble_configs = {
        #     "StackingEnsemble1": [
        #         ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
        #         ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
        #         ('ada', AdaBoostClassifier(random_state=42, algorithm='SAMME'))
        #     ],
        #     "StackingEnsemble2": [
        #         ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
        #         ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
        #         ('mlp', MLPClassifier(random_state=42, max_iter=10000))
        #     ],
        #     "StackingEnsemble3": [
        #         ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
        #         ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
        #         ('svc', SVC(probability=True))
        #     ],
        #     "StackingEnsemble4": [
        #         ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
        #         ('ada', AdaBoostClassifier(random_state=42, algorithm='SAMME')),
        #         ('knn', KNeighborsClassifier())
        #     ],
        #     "StackingEnsemble5": [
        #         ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
        #         ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
        #         ('ada', AdaBoostClassifier(random_state=42, algorithm='SAMME')),
        #         ('mlp', MLPClassifier(random_state=42, max_iter=10000))
        #     ],
        # }
        # for ens_name, estimators in ensemble_configs.items():
        #     stacking = StackingClassifier(
        #         estimators=estimators,
        #         final_estimator=LogisticRegression(max_iter=10000)
        #     )
        #     pipelines[ens_name] = Pipeline([
        #         ('scaler', StandardScaler()),
        #         ('pca', PCA(svd_solver='randomized', random_state=42)),
        #         ('classifier', stacking)
        #     ])
        #     param_grids[ens_name] = {
        #         'classifier__final_estimator__C': [0.1, 1, 10]
        #     }

    else:
        # Build pipelines without PCA
        pipelines['XGBoost'] = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
        ])
        param_grids['XGBoost'] = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.01, 0.05],
            'classifier__subsample': [0.7, 0.8],
            'classifier__colsample_bytree': [0.7, 0.8],
            'classifier__gamma': [0, 0.1],
            'classifier__min_child_weight': [1, 3],
            'classifier__reg_lambda': [1, 5],
            'classifier__reg_alpha': [0, 0.1]
        }

        # pipelines['RandomForest'] = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
        # ])
        # param_grids['RandomForest'] = {
        #     'classifier__n_estimators': [100, 200],
        #     'classifier__max_depth': [None, 10],
        #     'classifier__min_samples_split': [2, 5],
        #     'classifier__min_samples_leaf': [1, 2],
        #     'classifier__bootstrap': [True],
        #     'classifier__criterion': ['gini'],
        #     'classifier__max_features': ['sqrt']
        # }

        pipelines['MLP'] = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(random_state=42, max_iter=10000))
        ])
        param_grids['MLP'] = {
            'classifier__hidden_layer_sizes': [(100,), (100, 50), (250,), (250, 100)],
            'classifier__activation': ['relu'],
            'classifier__solver': ['adam'],
            'classifier__alpha': [0.00001, 0.0001, 0.001],
            'classifier__learning_rate': ['constant', 'adaptive'],
            'classifier__early_stopping': [True],
            'classifier__batch_size': [32, 64]
        }

        # pipelines['SVC'] = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('classifier', SVC(probability=True))
        # ])
        # param_grids['SVC'] = {
        #     'classifier__C': [0.1, 1, 10],
        #     'classifier__kernel': ['rbf', 'linear'],
        #     'classifier__gamma': ['scale', 'auto']
        # }

        # pipelines['LogisticRegression'] = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('classifier', LogisticRegression(max_iter=10000))
        # ])
        # param_grids['LogisticRegression'] = {
        #     'classifier__C': [0.1, 1, 10],
        #     'classifier__solver': ['lbfgs', 'saga']
        # }
        #
        # pipelines['KNN'] = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('classifier', KNeighborsClassifier())
        # ])
        # param_grids['KNN'] = {
        #     'classifier__n_neighbors': [3, 5, 7],
        #     'classifier__weights': ['uniform', 'distance']
        # }
        #
        # pipelines['AdaBoost'] = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('classifier', AdaBoostClassifier(random_state=42, algorithm='SAMME'))
        # ])
        # param_grids['AdaBoost'] = {
        #     'classifier__n_estimators': [50, 100],
        #     'classifier__learning_rate': [0.5, 1.0]
        # }
        #
        # ensemble_configs = {
        #     "StackingEnsemble1": [
        #         ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
        #         ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
        #         ('ada', AdaBoostClassifier(random_state=42, algorithm='SAMME'))
        #     ],
        #     "StackingEnsemble2": [
        #         ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
        #         ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
        #         ('mlp', MLPClassifier(random_state=42, max_iter=10000))
        #     ],
        #     "StackingEnsemble3": [
        #         ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
        #         ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
        #         ('svc', SVC(probability=True))
        #     ],
        #     "StackingEnsemble4": [
        #         ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
        #         ('ada', AdaBoostClassifier(random_state=42, algorithm='SAMME')),
        #         ('knn', KNeighborsClassifier())
        #     ],
        #     "StackingEnsemble5": [
        #         ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
        #         ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
        #         ('ada', AdaBoostClassifier(random_state=42, algorithm='SAMME')),
        #         ('mlp', MLPClassifier(random_state=42, max_iter=10000))
        #     ],
        # }
        # for ens_name, estimators in ensemble_configs.items():
        #     stacking = StackingClassifier(
        #         estimators=estimators,
        #         final_estimator=LogisticRegression(max_iter=10000)
        #     )
        #     pipelines[ens_name] = Pipeline([
        #         ('scaler', StandardScaler()),
        #         ('classifier', stacking)
        #     ])
        #     param_grids[ens_name] = {
        #         'classifier__final_estimator__C': [0.1, 1, 10]
        #     }

    return pipelines, param_grids


def run_models(data, features, filename_feature, min_samples=100, apply_pca=True, precision_test_threshold=0.5):
    """
    Run grid-search experiments over different models.
    If apply_pca is True, PCA is applied using various variance thresholds
    to dynamically determine the number of components.

    A master counter is computed that shows the total number of individual tests
    (across all parameter combinations and probability thresholds) to be performed.

    For each outer run (a unique combination of variance threshold (if applicable),
    SMOTE level, and model pipeline), a concise update is printed showing:
      - The outer run number (out of total outer runs)
      - The model name and SMOTE level
      - If PCA is applied, the optimal number of PCA components for the current variance threshold
      - And the total number of parameter combinations being searched for that model.

    The following metrics are computed and saved (for those runs meeting performance criteria):
      - MCC (train, test, ratio)
      - Accuracy (train, test, ratio)
      - F1 Score (train, test, ratio)
      - AUC (train, test, ratio)
      - Precision (train, test, ratio)
      - Recall (train, test, ratio)
    """

    # Print basic dataset info
    print("Data length:", len(data))
    print("Total positive targets:", data['target'].sum())

    # Separate features and target
    X = data[features]
    y = data['target']

    # Time-series split: first 80% for training, last 20% for testing
    train_size = int(len(data) * 0.8)
    X_train_full = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train_full = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    # Compute class distribution for SMOTE
    counts = Counter(y_train_full)
    minority_class = min(counts, key=counts.get)
    majority_class = max(counts, key=counts.get)
    current_ratio = counts[minority_class] / counts[majority_class]
    print("Current minority/majority ratio:", current_ratio)

    # Define SMOTE strategies and probability thresholds
    # lower_bound = max(np.ceil(current_ratio * 100) / 100, 0.55)
    upper_bound = 0.95  # Adjust as necessary
    # smote_strategies = [round(x, 2) for x in np.arange(lower_bound, upper_bound, 0.05)] + [None]
    smote_strategies = [round(x, 2) for x in np.arange(current_ratio + 0.01, upper_bound, 0.05)] + [None]
    probability_thresholds = [round(x, 2) for x in np.arange(0.2, 0.81, 0.01)]

    metrics_list = []

    for apply_pca in [False]:
        # If PCA is not applied, use a dummy variance threshold list.
        if apply_pca:
            var_thresholds = [0.9, 0.92, 0.94, 0.96, 0.98]
        else:
            var_thresholds = [1]

        # Compute master total tests (each test is one evaluation for a given probability threshold).
        temp_pipelines, temp_param_grids = build_pipelines(apply_pca=apply_pca)
        master_total_tests = 0
        for model in temp_param_grids:
            num_param = len(list(ParameterGrid(temp_param_grids[model])))
            master_total_tests += len(var_thresholds) * len(smote_strategies) * num_param * len(probability_thresholds)
        print(f"Total tests to be performed: {master_total_tests}")

        # Also compute total outer runs (each unique combination of variance threshold, SMOTE level, and model)
        if apply_pca:
            total_outer_runs = len(var_thresholds) * len(smote_strategies) * len(temp_pipelines)
        else:
            total_outer_runs = len(smote_strategies) * len(temp_pipelines)
        outer_run_counter = 0
        print(f"Total grid search outer runs: {total_outer_runs}")

        master_test_counter = 0  # Counter for individual tests

        if apply_pca:
            for var_threshold in var_thresholds:
                # Determine optimal PCA components for current variance threshold.
                optimal_n_components = select_optimal_pca_components(X_train_full, variance_threshold=var_threshold)
                print(f"PCA: {optimal_n_components} components for {var_threshold * 100:.0f}% variance")

                # Build pipelines with PCA and update the PCA step.
                pipelines, param_grids = build_pipelines(apply_pca=True)
                for model_name, pipeline in pipelines.items():
                    pipeline.set_params(pca__n_components=optimal_n_components)

                # Loop over SMOTE strategies.
                for sample_st in smote_strategies:
                    if sample_st is not None:
                        smote = SMOTE(sampling_strategy=sample_st, random_state=42)
                        X_train_res, y_train_res = smote.fit_resample(X_train_full, y_train_full)
                        smote_label = sample_st
                    else:
                        X_train_res, y_train_res = X_train_full, y_train_full
                        smote_label = "None"
                    print(f"SMOTE level: {smote_label}")

                    # For each model pipeline (outer run)
                    for model_name in pipelines.keys():
                        outer_run_counter += 1
                        num_params = len(list(ParameterGrid(param_grids[model_name])))
                        print(
                            f"Run {outer_run_counter}/{total_outer_runs} - Model: {model_name}, SMOTE: {smote_label}, Params to search: {num_params}")

                        # Loop over parameter combinations (evaluated silently)
                        for params in ParameterGrid(param_grids[model_name]):
                            pipeline = pipelines[model_name]
                            pipeline.set_params(**params)
                            pipeline.fit(X_train_res, y_train_res)
                            train_probs = pipeline.predict_proba(X_train_res)[:, 1]
                            test_probs = pipeline.predict_proba(X_test)[:, 1]

                            # Loop over probability thresholds
                            for thresh in probability_thresholds:
                                master_test_counter += 1
                                train_pred = (train_probs >= thresh).astype(int)
                                test_pred = (test_probs >= thresh).astype(int)

                                # Only record the result if at least 100 samples are predicted positive in the test set.
                                if np.sum(test_pred) < min_samples:
                                    continue

                                # Calculate sample sizes
                                train_sample_size = np.sum(train_pred)
                                test_sample_size = np.sum(test_pred)

                                # Compute MCC and its ratio
                                train_mcc = matthews_corrcoef(y_train_res, train_pred)
                                test_mcc = matthews_corrcoef(y_test, test_pred)
                                mcc_ratio = test_mcc / (train_mcc + 1e-10)
                                if mcc_ratio > 1:
                                    mcc_ratio = train_mcc / (test_mcc + 1e-10)

                                # Compute Accuracy and its ratio
                                train_acc = accuracy_score(y_train_res, train_pred)
                                test_acc = accuracy_score(y_test, test_pred)
                                acc_ratio = test_acc / (train_acc + 1e-10)
                                if acc_ratio > 1:
                                    acc_ratio = train_acc / (test_acc + 1e-10)

                                # Compute F1 Score and its ratio
                                train_f1 = f1_score(y_train_res, train_pred)
                                test_f1 = f1_score(y_test, test_pred)
                                f1_ratio = test_f1 / (train_f1 + 1e-10)
                                if f1_ratio > 1:
                                    f1_ratio = train_f1 / (test_f1 + 1e-10)

                                # Compute AUC and its ratio
                                train_auc = roc_auc_score(y_train_res, train_probs)
                                test_auc = roc_auc_score(y_test, test_probs)
                                auc_ratio = test_auc / (train_auc + 1e-10)
                                if auc_ratio > 1:
                                    auc_ratio = train_auc / (test_auc + 1e-10)

                                # Compute Precision and its ratio (with zero_division handling)
                                train_precision = precision_score(y_train_res, train_pred, zero_division=0)
                                test_precision = precision_score(y_test, test_pred, zero_division=0)
                                precision_ratio = test_precision / (train_precision + 1e-10)
                                if precision_ratio > 1:
                                    precision_ratio = train_precision / (test_precision + 1e-10)

                                # Compute Recall and its ratio (with zero_division handling)
                                train_recall = recall_score(y_train_res, train_pred, zero_division=0)
                                test_recall = recall_score(y_test, test_pred, zero_division=0)
                                recall_ratio = test_recall / (train_recall + 1e-10)
                                if recall_ratio > 1:
                                    recall_ratio = train_recall / (test_recall + 1e-10)

                                # Save metrics if performance criteria are met
                                if (precision_ratio >= 0.9) and (test_precision >= precision_test_threshold):
                                    metrics_list.append({
                                        'Model': model_name,
                                        'SMOTE': smote_label,
                                        'Probability_Threshold': thresh,
                                        'AUC_Train': round(train_auc, 4),
                                        'AUC_Test': round(test_auc, 4),
                                        'AUC_Test/Train_Ratio': round(auc_ratio, 4),
                                        'Precision_Train': round(train_precision, 4),
                                        'Precision_Test': round(test_precision, 4),
                                        'Precision_Test/Train_Ratio': round(precision_ratio, 4),
                                        'MCC_Train': round(train_mcc, 4),
                                        'MCC_Test': round(test_mcc, 4),
                                        'MCC_Test/Train_Ratio': round(mcc_ratio, 4),
                                        'ACC_Train': round(train_acc, 4),
                                        'ACC_Test': round(test_acc, 4),
                                        'ACC_Test/Train_Ratio': round(acc_ratio, 4),
                                        'F1_Train': round(train_f1, 4),
                                        'F1_Test': round(test_f1, 4),
                                        'F1_Test/Train_Ratio': round(f1_ratio, 4),
                                        'Recall_Train': round(train_recall, 4),
                                        'Recall_Test': round(test_recall, 4),
                                        'Recall_Test/Train_Ratio': round(recall_ratio, 4),
                                        'Train_Sample_Size': train_sample_size,
                                        'Test_Sample_Size': test_sample_size,
                                        'Var_Threshold': None,
                                        'Params': params
                                    })
        else:
            print("Running models without PCA")
            pipelines, param_grids = build_pipelines(apply_pca=False)
            for sample_st in smote_strategies:
                if sample_st is not None:
                    smote = SMOTE(sampling_strategy=sample_st, random_state=42)
                    X_train_res, y_train_res = smote.fit_resample(X_train_full, y_train_full)
                    smote_label = sample_st
                else:
                    X_train_res, y_train_res = X_train_full, y_train_full
                    smote_label = "None"
                print(f"SMOTE level: {smote_label}")

                for model_name in pipelines.keys():
                    outer_run_counter += 1
                    num_params = len(list(ParameterGrid(param_grids[model_name])))
                    print(
                        f"Run {outer_run_counter}/{total_outer_runs} - Model: {model_name}, SMOTE: {smote_label}, Params to search: {num_params}")

                    for params in ParameterGrid(param_grids[model_name]):
                        pipeline = pipelines[model_name]
                        pipeline.set_params(**params)
                        pipeline.fit(X_train_res, y_train_res)
                        train_probs = pipeline.predict_proba(X_train_res)[:, 1]
                        test_probs = pipeline.predict_proba(X_test)[:, 1]

                        # Loop over probability thresholds
                        for thresh in probability_thresholds:
                            master_test_counter += 1
                            train_pred = (train_probs >= thresh).astype(int)
                            test_pred = (test_probs >= thresh).astype(int)

                            # Only record the result if at least 100 samples are predicted positive in the test set.
                            if np.sum(test_pred) < min_samples:
                                continue

                            # Calculate sample sizes
                            train_sample_size = np.sum(train_pred)
                            test_sample_size = np.sum(test_pred)

                            # Compute MCC and its ratio
                            train_mcc = matthews_corrcoef(y_train_res, train_pred)
                            test_mcc = matthews_corrcoef(y_test, test_pred)
                            mcc_ratio = test_mcc / (train_mcc + 1e-10)
                            if mcc_ratio > 1:
                                mcc_ratio = train_mcc / (test_mcc + 1e-10)

                            # Compute Accuracy and its ratio
                            train_acc = accuracy_score(y_train_res, train_pred)
                            test_acc = accuracy_score(y_test, test_pred)
                            acc_ratio = test_acc / (train_acc + 1e-10)
                            if acc_ratio > 1:
                                acc_ratio = train_acc / (test_acc + 1e-10)

                            # Compute F1 Score and its ratio
                            train_f1 = f1_score(y_train_res, train_pred)
                            test_f1 = f1_score(y_test, test_pred)
                            f1_ratio = test_f1 / (train_f1 + 1e-10)
                            if f1_ratio > 1:
                                f1_ratio = train_f1 / (test_f1 + 1e-10)

                            # Compute AUC and its ratio
                            train_auc = roc_auc_score(y_train_res, train_probs)
                            test_auc = roc_auc_score(y_test, test_probs)
                            auc_ratio = test_auc / (train_auc + 1e-10)
                            if auc_ratio > 1:
                                auc_ratio = train_auc / (test_auc + 1e-10)

                            # Compute Precision and its ratio (with zero_division handling)
                            train_precision = precision_score(y_train_res, train_pred, zero_division=0)
                            test_precision = precision_score(y_test, test_pred, zero_division=0)
                            precision_ratio = test_precision / (train_precision + 1e-10)
                            if precision_ratio > 1:
                                precision_ratio = train_precision / (test_precision + 1e-10)

                            # Compute Recall and its ratio (with zero_division handling)
                            train_recall = recall_score(y_train_res, train_pred, zero_division=0)
                            test_recall = recall_score(y_test, test_pred, zero_division=0)
                            recall_ratio = test_recall / (train_recall + 1e-10)
                            if recall_ratio > 1:
                                recall_ratio = train_recall / (test_recall + 1e-10)

                            # Save metrics if performance criteria are met
                            if (precision_ratio >= 0.9) and (test_precision >= precision_test_threshold):
                                metrics_list.append({
                                    'Model': model_name,
                                    'SMOTE': smote_label,
                                    'Probability_Threshold': thresh,
                                    'AUC_Train': round(train_auc, 4),
                                    'AUC_Test': round(test_auc, 4),
                                    'AUC_Test/Train_Ratio': round(auc_ratio, 4),
                                    'Precision_Train': round(train_precision, 4),
                                    'Precision_Test': round(test_precision, 4),
                                    'Precision_Test/Train_Ratio': round(precision_ratio, 4),
                                    'MCC_Train': round(train_mcc, 4),
                                    'MCC_Test': round(test_mcc, 4),
                                    'MCC_Test/Train_Ratio': round(mcc_ratio, 4),
                                    'ACC_Train': round(train_acc, 4),
                                    'ACC_Test': round(test_acc, 4),
                                    'ACC_Test/Train_Ratio': round(acc_ratio, 4),
                                    'F1_Train': round(train_f1, 4),
                                    'F1_Test': round(test_f1, 4),
                                    'F1_Test/Train_Ratio': round(f1_ratio, 4),
                                    'Recall_Train': round(train_recall, 4),
                                    'Recall_Test': round(test_recall, 4),
                                    'Recall_Test/Train_Ratio': round(recall_ratio, 4),
                                    'Train_Sample_Size': train_sample_size,
                                    'Test_Sample_Size': test_sample_size,
                                    # 'Var_Threshold': var_threshold,
                                    'Params': params
                                })

    print(f"Total tests performed: {master_test_counter}")
    # Save the results to a CSV file with a timestamp in the filename that includes the filename_feature.
    filename_out = f"model_metrics_{filename_feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    metrics_df = pd.DataFrame(metrics_list)
    # Sort by Precision_Test (greatest at the top)
    if not metrics_df.empty:
        metrics_df.sort_values(by='Test_Sample_Size', ascending=False, inplace=True)
        metrics_df.to_csv(filename_out, index=False)
        print(f"Model metrics saved to {filename_out}")
    else:
        metrics_df.to_csv(filename_out, index=False)
        print(f"Model metrics saved to {filename_out}")
        # print(f"No metrics were recorded for {filename_feature}.")


def run_models_v2(
        matches_filtered: pd.DataFrame,
        features: list,
        ht_score: tuple | str,
        min_samples: int = 200,  # validation gate: min predicted positives across folds
        min_test_samples: int = 100,  # test gate: min predicted positives on hold-out test
        precision_test_threshold: float = 0.80,
        base_model: str = "xgb",  # "xgb" or "mlp"
        search_mode: str = "random",  # "random" or "grid"
        n_random_param_sets: int = 10,
        cpu_jobs: int = 6,
        top_k: int = 10,
        thresholds: np.ndarray | None = None,
        out_dir: str | None = None,
        # --- anti-overfitting knobs ---
        val_conf_level: float = 0.95,  # Wilson-LCB confidence for validation precision
        max_precision_drop: float = 0.10,  # allow at most 10pp drop val → test
        # --- failure handling ---
        on_fail: str = "return",  # "return" | "warn" | "raise"
        save_diagnostics_on_fail: bool = True,
):
    """
    Rolling time-ordered CV (no leakage) with calibration and robust failure handling.

    Random every run:
      - Hyper-parameter samples change each call (fresh entropy).
      - Model random_state changes each call (stable within-run per param set).

    Selection:
      1) Evaluate candidates on rolling validation across thresholds; track TP/FP.
      2) Rank by Wilson LCB of validation precision, then mean val precision, n_preds_val, val_accuracy.
      3) Take Top-K by that ranking. Fit on pre-test, calibrate on small final-val window,
         evaluate on TEST at each chosen threshold.
      4) TEST GATE: keep ONLY rows with
            n_preds_test ≥ min_test_samples
         AND test_precision ≥ max(precision_test_threshold, val_precision - max_precision_drop).
      5) Save CSV with ONLY survivors (includes 'model_pkl' for the top row), and save PKL for the top survivor.

    Failure case (no survivors):
      - If save_diagnostics_on_fail=True, writes a *_FAILED.csv with all Top-K test results and 'fail_reason'.
      - Behaviour controlled by `on_fail`: "return" (default), "warn", or "raise".
    """
    # ------------------ Imports & setup ------------------
    import os, secrets, hashlib
    from datetime import datetime
    import numpy as np
    import pandas as pd
    from itertools import product
    from math import sqrt
    from sklearn.model_selection import ParameterSampler
    from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, log_loss, brier_score_loss
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from joblib import Parallel, delayed, parallel_backend
    from tqdm import tqdm
    from tqdm_joblib import tqdm_joblib
    import joblib

    # xgboost (optional)
    try:
        import xgboost as xgb
        _HAS_XGB_LOCAL = True
    except Exception:
        _HAS_XGB_LOCAL = False

    # fallbacks for distributions if not defined at module scope
    try:
        _randint  # noqa: F821
        _uniform  # noqa: F821
        _loguniform  # noqa: F821
    except NameError:
        from scipy.stats import randint as _randint
        from scipy.stats import uniform as _uniform
        from scipy.stats import loguniform as _loguniform

    # normal quantile for Wilson LCB
    try:
        from scipy.stats import norm
        _Z = lambda conf: float(norm.ppf(1 - (1 - conf) / 2))
    except Exception:
        _Z = lambda conf: 1.96 if abs(conf - 0.95) < 1e-6 else 1.64  # crude fallback

    def _wilson_lcb(tp: int, fp: int, conf: float) -> float:
        n = tp + fp
        if n <= 0:
            return 0.0
        p = tp / n
        z = _Z(conf)
        denom = 1.0 + (z * z) / n
        centre = p + (z * z) / (2 * n)
        rad = z * sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
        return max(0.0, (centre - rad) / denom)

    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)

    ht_tag = ht_score[0] if isinstance(ht_score, tuple) else str(ht_score)
    out_dir = out_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    # honour external _HAS_XGB if present, else use local probe
    _HAS_XGB = globals().get("_HAS_XGB", _HAS_XGB_LOCAL)
    if base_model == "xgb" and not _HAS_XGB:
        raise ImportError("XGBoost not available; set base_model='mlp' or install xgboost.")

    # Fresh run seed → different every invocation
    RUN_SEED = secrets.randbits(32)

    def _seed_from(*vals) -> int:
        """Derive a stable (within-run) 31-bit positive seed from RUN_SEED and provided values."""
        h = hashlib.blake2b(digest_size=8)
        h.update(int(RUN_SEED).to_bytes(8, 'little', signed=False))
        for v in vals:
            h.update(str(v).encode('utf-8'))
        return int.from_bytes(h.digest(), 'little') & 0x7FFFFFFF

    # ------------------ Data prep ------------------
    req_cols = {'date', 'target'}
    missing = req_cols - set(matches_filtered.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = matches_filtered.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    cols_needed = list(set(features) | {'target'})
    df = df.dropna(subset=cols_needed).reset_index(drop=True)

    X = df[features].copy()
    y = df['target'].astype(int).reset_index(drop=True)

    n = len(X)
    if n < max(min_samples * 3, 500):
        raise RuntimeError(f"Not enough rows for HT score {ht_tag}: {n}")

    # ------------------ Temporal split ------------------
    test_start = int(0.85 * n)
    pretest_end = test_start

    X_test = X.iloc[test_start:].reset_index(drop=True)
    y_test = y.iloc[test_start:].reset_index(drop=True)

    # Rolling folds inside [0, pretest_end)
    N_FOLDS = 5
    total_val_len = max(1, int(0.15 * n))
    val_len = max(1, total_val_len // N_FOLDS)
    fold_val_ends = [pretest_end - total_val_len + (i + 1) * val_len for i in range(N_FOLDS)]
    fold_val_starts = [end - val_len for end in fold_val_ends]
    if fold_val_ends:
        fold_val_ends[-1] = min(fold_val_ends[-1], pretest_end)
        fold_val_starts[-1] = max(0, fold_val_ends[-1] - val_len)

    # Final small validation slice (for calibration before test)
    final_val_len = max(1, val_len)
    final_val_start = max(0, test_start - final_val_len)
    X_train_final = X.iloc[:final_val_start]
    y_train_final = y.iloc[:final_val_start]
    X_val_final = X.iloc[final_val_start:test_start]
    y_val_final = y.iloc[final_val_start:test_start]

    # ------------------ Hyper-parameter spaces ------------------
    xgb_param_grid = {
        'n_estimators': [200],
        'max_depth': [5],
        'learning_rate': [0.1],
        'subsample': [0.7],
        'colsample_bytree': [1.0],
        'min_child_weight': [5],
        'reg_lambda': [1.0],
    }
    xgb_param_distributions = {
        'n_estimators': _randint(100, 1001),
        'max_depth': _randint(3, 8),
        'learning_rate': _loguniform(0.01, 0.2),
        'min_child_weight': _randint(3, 13),
        'subsample': _uniform(0.7, 0.3),
        'colsample_bytree': _uniform(0.6, 0.4),
        'reg_lambda': _loguniform(0.1, 10.0),
    }
    mlp_param_grid = {
        'hidden_layer_sizes': [(128,), (256,), (128, 64)],
        'alpha': [1e-4],
        'learning_rate_init': [1e-3],
        'batch_size': ['auto'],
        'max_iter': [200],
    }
    mlp_param_distributions = {
        'hidden_layer_sizes': [(64,), (128,), (256,), (128, 64), (256, 128)],
        'alpha': _loguniform(1e-5, 1e-2),
        'learning_rate_init': _loguniform(5e-4, 5e-2),
        'batch_size': _randint(32, 257),
        'max_iter': _randint(150, 401),
    }

    if search_mode.lower() == "grid":
        grid, dists = (xgb_param_grid, None) if base_model == "xgb" else (mlp_param_grid, None)
        all_param_dicts = [dict(zip(grid.keys(), combo)) for combo in product(*grid.values())]
    else:
        grid, dists = (xgb_param_grid, xgb_param_distributions) if base_model == "xgb" else (mlp_param_grid,
                                                                                             mlp_param_distributions)
        # Random every run
        sampler_seed = _seed_from("sampler")
        all_param_dicts = list(ParameterSampler(dists, n_iter=n_random_param_sets, random_state=sampler_seed))

    # ------------------ Helpers ------------------
    def cast_params(p: dict) -> dict:
        q = dict(p)
        if base_model == "xgb":
            for k in ['n_estimators', 'max_depth', 'min_child_weight']:
                if k in q: q[k] = int(q[k])
            for k in ['learning_rate', 'subsample', 'colsample_bytree', 'reg_lambda']:
                if k in q: q[k] = float(q[k])
        else:
            if 'max_iter' in q: q['max_iter'] = int(q['max_iter'])
            if 'batch_size' in q and q['batch_size'] != 'auto':
                q['batch_size'] = int(q['batch_size'])
            if 'alpha' in q: q['alpha'] = float(q['alpha'])
            if 'learning_rate_init' in q: q['learning_rate_init'] = float(q['learning_rate_init'])
            if 'hidden_layer_sizes' in q:
                h = q['hidden_layer_sizes']
                q['hidden_layer_sizes'] = tuple(h) if not isinstance(h, tuple) else h
        return q

    def _final_step_name(estimator):
        try:
            if isinstance(estimator, Pipeline):
                return estimator.steps[-1][0]
        except Exception:
            pass
        return None

    def build_model(params: dict, spw: float):
        # model seed depends on RUN_SEED and params → changes each run
        model_seed = _seed_from("model", base_model, tuple(sorted(params.items())))
        if base_model == "xgb":
            return xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=model_seed,
                scale_pos_weight=spw,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
                **params
            )
        else:
            mlp = MLPClassifier(
                random_state=model_seed,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
                solver="adam",
                **params
            )
            return make_pipeline(StandardScaler(with_mean=True, with_std=True), mlp)

    def fit_model(model, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None):
        if base_model == "xgb":
            try:
                model.set_params(verbosity=0, early_stopping_rounds=50)
                if X_va is not None and y_va is not None:
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                else:
                    model.fit(X_tr, y_tr, verbose=False)
            except Exception:
                model.fit(X_tr, y_tr, verbose=False)
        else:
            fit_kwargs = {}
            if sample_weight is not None:
                stepname = _final_step_name(model)
                if stepname is not None:
                    fit_kwargs[f"{stepname}__sample_weight"] = sample_weight
            try:
                model.fit(X_tr, y_tr, **fit_kwargs)
            except TypeError:
                model.fit(X_tr, y_tr)

    def fit_calibrator(fitted, X_va, y_va):
        try:
            from sklearn.calibration import FrozenEstimator
            frozen = FrozenEstimator(fitted)
            cal = CalibratedClassifierCV(frozen, method='sigmoid', cv=None)
            cal.fit(X_va, y_va)
            return cal
        except Exception:
            try:
                cal = CalibratedClassifierCV(fitted, method='sigmoid', cv='prefit')
                cal.fit(X_va, y_va)
                return cal
            except Exception:
                return fitted

    def predict_proba_1(model_or_cal, X_):
        proba = model_or_cal.predict_proba(X_)
        return proba[:, 1].astype(np.float32) if proba.ndim == 2 else np.asarray(proba, dtype=np.float32)

    # ------------------ Rolling-CV evaluator (VALIDATION) ------------------
    def evaluate_param_set(param_dict, task_id=None, total_tasks=None):
        safe = cast_params(param_dict)
        rows = []
        val_prob_all, val_true_all = [], []

        for vstart, vend in zip(fold_val_starts, fold_val_ends):
            if vstart is None or vend is None or vstart <= 0 or vend <= vstart:
                continue

            X_tr, y_tr = X.iloc[:vstart], y.iloc[:vstart]
            X_va, y_va = X.iloc[vstart:vend], y.iloc[vstart:vend]

            pos = int(y_tr.sum());
            neg = len(y_tr) - pos
            spw = (neg / pos) if pos > 0 else 1.0

            sample_weight = None
            if base_model == "mlp":
                w_pos = spw
                sample_weight = np.where(y_tr.values == 1, w_pos, 1.0).astype(np.float32)

            model = build_model(safe, spw)
            fit_model(model, X_tr, y_tr, X_va, y_va, sample_weight=sample_weight)

            cal = fit_calibrator(model, X_va, y_va)
            proba_va = predict_proba_1(cal, X_va)

            val_prob_all.append(proba_va)
            y_true = y_va.values.astype(np.uint8)
            val_true_all.append(y_true)

            for thr in thresholds:
                y_pred = (proba_va >= thr).astype(np.uint8)
                n_preds = int(y_pred.sum())
                tp = int(((y_true == 1) & (y_pred == 1)).sum())
                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                prc = precision_score(y_va, y_pred, zero_division=0)
                acc = accuracy_score(y_va, y_pred)

                rows.append({
                    **safe,
                    'threshold': float(thr),
                    'fold_vstart': int(vstart),
                    'fold_vend': int(vend),
                    'n_preds_val': n_preds,
                    'tp_val': tp,
                    'fp_val': fp,
                    'val_precision': float(prc),
                    'val_accuracy': float(acc),
                })

        # pooled diagnostics (optional)
        if val_prob_all:
            vp = np.concatenate(val_prob_all, axis=0)
            vt = np.concatenate(val_true_all, axis=0)
            try:
                val_auc = float(roc_auc_score(vt, vp))
            except Exception:
                val_auc = np.nan
            try:
                val_ll = float(log_loss(vt, vp, labels=[0, 1]))
            except Exception:
                val_ll = np.nan
            try:
                val_bri = float(brier_score_loss(vt, vp))
            except Exception:
                val_bri = np.nan
        else:
            val_auc = val_ll = val_bri = np.nan

        for r in rows:
            r['val_auc'] = val_auc
            r['val_logloss'] = val_ll
            r['val_brier'] = val_bri

        return rows

    # ------------------ Parallel parameter search ------------------
    total_tasks = len(all_param_dicts)
    if base_model == "mlp":
        eff_jobs = min(max(1, cpu_jobs), 4)
        prefer = "threads";
        backend = "threading";
        pre_dispatch = eff_jobs
        ctx = parallel_backend(backend, n_jobs=eff_jobs)
    else:
        eff_jobs = max(1, min(cpu_jobs, 4)) if cpu_jobs != -1 else 4
        prefer = "processes";
        backend = "loky";
        pre_dispatch = f"{2 * eff_jobs}"
        ctx = parallel_backend(backend, n_jobs=eff_jobs, inner_max_num_threads=1)

    with ctx:
        try:
            with tqdm_joblib(tqdm(total=total_tasks, desc=f"Param search ({search_mode}, {base_model})")) as _:
                out = Parallel(
                    n_jobs=eff_jobs, batch_size=1, prefer=prefer, pre_dispatch=pre_dispatch
                )(
                    delayed(evaluate_param_set)(pd_, i, total_tasks)
                    for i, pd_ in enumerate(all_param_dicts)
                )
        except OSError as e:
            print(f"[WARN] Parallel failed with {e}. Falling back to serial search...")
            out = []
            for i, pd_ in enumerate(tqdm(all_param_dicts, desc=f"Param search (serial, {base_model})")):
                out.append(evaluate_param_set(pd_, i, total_tasks))

    val_rows = [r for sub in out for r in sub]
    if not val_rows:
        raise RuntimeError("No validation rows produced (check folds and input data).")
    val_df = pd.DataFrame(val_rows)

    # ------------------ Aggregate across folds (VALIDATION) ------------------
    param_keys = list((xgb_param_grid if base_model == "xgb" else mlp_param_grid).keys())
    group_cols = param_keys + ['threshold']
    agg = val_df.groupby(group_cols, as_index=False).agg({
        'n_preds_val': 'sum',
        'tp_val': 'sum',
        'fp_val': 'sum',
        'val_precision': 'mean',
        'val_accuracy': 'mean',
        'val_auc': 'mean',
        'val_logloss': 'mean',
        'val_brier': 'mean',
    })

    # Wilson-LCB & pooled precision
    agg['val_precision_pooled'] = agg.apply(
        lambda r: (r['tp_val'] / max(1, (r['tp_val'] + r['fp_val']))), axis=1
    )
    agg['val_precision_lcb'] = agg.apply(
        lambda r: _wilson_lcb(int(r['tp_val']), int(r['fp_val']), conf=val_conf_level), axis=1
    )

    # Validation gates (mean precision + count)
    qual = agg[
        (agg['val_precision'] >= float(precision_test_threshold)) &
        (agg['n_preds_val'] >= int(min_samples))
        ].copy()
    if qual.empty:
        # nothing qualifies even on validation → treat as failure early
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_diagnostics_on_fail:
            fail_path = os.path.join(out_dir, f"model_metrics_{repr(ht_tag)}_{timestamp}_FAILED.csv")
            (agg.sort_values(['val_precision_lcb', 'val_precision', 'n_preds_val', 'val_accuracy'],
                             ascending=[False, False, False, False])
             .assign(fail_reason="failed_validation_gate")
             .to_csv(fail_path, index=False))
        msg = (f"No strategy met validation gates (precision ≥ {precision_test_threshold} "
               f"and n_preds_val ≥ {min_samples}) for HT {ht_tag}.")
        if on_fail == "raise":
            raise RuntimeError(msg)
        if on_fail == "warn":
            print("[WARN]", msg)
        return {
            'status': 'failed_validation_gate',
            'csv': fail_path if save_diagnostics_on_fail else None,
            'model_pkl': None,
            'summary_df': None,
            'validation_table': agg.sort_values(['val_precision_lcb', 'val_precision', 'n_preds_val', 'val_accuracy'],
                                                ascending=[False, False, False, False]).reset_index(drop=True)
        }

    # STRICT validation ordering (conservative first)
    ranked = qual.sort_values(
        by=['val_precision_lcb', 'val_precision', 'n_preds_val', 'val_accuracy'],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    topk_val = ranked.head(top_k).reset_index(drop=True)

    # ------------------ Evaluate ALL Top-K on TEST ------------------
    candidates = []
    for _, row in topk_val.iterrows():
        candidates.append({
            'params': {k: row[k] for k in param_keys if k in row.index},
            'threshold': float(row['threshold']),
            'val_precision': float(row['val_precision']),
            'val_precision_lcb': float(row['val_precision_lcb']),
            'val_accuracy': float(row['val_accuracy']),
            'n_preds_val': int(row['n_preds_val']),
        })

    # Evaluate each Top-K candidate on TEST
    records_all = []  # every candidate with test metrics + pass/fail reason
    for cand in candidates:
        best_params = cast_params(cand['params'])
        pos = int(y_train_final.sum());
        neg = len(y_train_final) - pos
        spw_final = (neg / pos) if pos > 0 else 1.0

        final_model = build_model(best_params, spw_final)
        final_sample_weight = None
        if base_model == "mlp":
            w_pos = spw_final
            final_sample_weight = np.where(y_train_final.values == 1, w_pos, 1.0).astype(np.float32)

        fit_model(final_model, X_train_final, y_train_final, X_val_final, y_val_final,
                  sample_weight=final_sample_weight)
        final_calibrator = fit_calibrator(final_model, X_val_final, y_val_final)

        y_test_proba = predict_proba_1(final_calibrator, X_test)
        thr = cand['threshold']
        y_pred = (y_test_proba >= thr).astype(np.uint8)
        n_preds_test = int(y_pred.sum())
        prc_test = precision_score(y_test, y_pred, zero_division=0)
        acc_test = accuracy_score(y_test, y_pred)

        # TEST GATE checks + reason
        enough = n_preds_test >= int(min_test_samples)
        not_collapsed = prc_test >= max(float(precision_test_threshold),
                                        float(cand['val_precision']) - float(max_precision_drop))
        pass_gate = bool(enough and not_collapsed)
        reason = ""
        if not pass_gate:
            if not enough and not not_collapsed:
                reason = "insufficient_test_preds_and_precision_collapse"
            elif not enough:
                reason = "insufficient_test_preds"
            else:
                reason = "precision_collapse"

        records_all.append({
            **cand['params'],
            'threshold': thr,
            'val_precision_lcb': cand['val_precision_lcb'],
            'val_precision': cand['val_precision'],
            'val_accuracy': cand['val_accuracy'],
            'n_preds_val': cand['n_preds_val'],
            'n_preds_test': n_preds_test,
            'test_precision': float(prc_test),
            'test_accuracy': float(acc_test),
            'pass_test_gate': pass_gate,
            'fail_reason': reason,
            'model_obj': final_calibrator if pass_gate else None,
        })

    survivors_df = pd.DataFrame(records_all)
    passers = survivors_df[survivors_df['pass_test_gate']].copy()

    # ------------------ Persist outputs ------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "xgb" if base_model == "xgb" else "mlp"

    if passers.empty:
        # No survivor: write diagnostics, optionally return or warn instead of raising
        if save_diagnostics_on_fail:
            diag = (survivors_df
                    .drop(columns=['model_obj'])
                    .sort_values(by=['val_precision_lcb', 'val_precision', 'n_preds_val', 'val_accuracy'],
                                 ascending=[False, False, False, False])
                    )
            fail_csv = os.path.join(out_dir, f"model_metrics_{repr(ht_tag)}_{timestamp}_FAILED.csv")
            diag.to_csv(fail_csv, index=False)
        msg = (f"All Top-{len(candidates)} failed the TEST gate "
               f"(n_preds_test ≥ {min_test_samples} and precision not collapsing) for HT {ht_tag}.")
        if on_fail == "raise":
            raise RuntimeError(msg)
        if on_fail == "warn":
            print("[WARN]", msg)
        return {
            'status': 'failed_test_gate',
            'csv': fail_csv if save_diagnostics_on_fail else None,
            'model_pkl': None,
            'summary_df': diag if save_diagnostics_on_fail else survivors_df.drop(columns=['model_obj']),
            'validation_table': ranked,
        }

    # At least one survivor → choose best & save PKL + CSV (only passers)
    passers_sorted = passers.sort_values(
        by=['val_precision_lcb', 'val_precision', 'test_precision', 'n_preds_test', 'val_accuracy'],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)

    pkl_path = os.path.join(
        r'C:\Users\leere\PycharmProjects\Football_ML3\Goals\2H_goal\ht_scoreline\path_ht_score',
        f"best_model_{repr(ht_tag)}_{tag}_calibrated_{timestamp}.pkl"
    )

    # Prepare CSV (include the PKL path for the winning row only)
    csv_df = passers_sorted.drop(columns=['model_obj']).copy()
    csv_df['model_pkl'] = ""
    csv_df.loc[0, 'model_pkl'] = pkl_path

    csv_path = os.path.join(out_dir, f"model_metrics_{repr(ht_tag)}_{timestamp}.csv")
    csv_df.to_csv(csv_path, index=False)

    # Save PKL for the top row
    top_row = passers_sorted.iloc[0]
    chosen_model = top_row['model_obj']
    chosen_params = {k: top_row[k] for k in param_keys if k in passers_sorted.columns}
    chosen_threshold = float(top_row['threshold'])

    joblib.dump(
        {
            'model': chosen_model,
            'threshold': chosen_threshold,
            'features': features,
            'base_model': base_model,
            'best_params': chosen_params,
            'precision_test_threshold': float(precision_test_threshold),
            'min_samples': int(min_samples),
            'min_test_samples': int(min_test_samples),
            'val_conf_level': float(val_conf_level),
            'max_precision_drop': float(max_precision_drop),
            'ht_score': ht_tag,
            'notes': (
                'CSV includes only candidates passing test gate; ranked by '
                'val_precision_lcb → val_precision → test_precision → n_preds_test → val_accuracy. '
                'Seeds are random each run.'
            ),
            'run_seed': int(RUN_SEED),  # for traceability
        },
        pkl_path
    )

    return {
        'status': 'ok',
        'csv': csv_path,
        'model_pkl': pkl_path,
        'summary_df': csv_df,  # passers only, with model_pkl set on row 0
        'validation_table': ranked,  # full validation ranking (post-gates)
    }


def run_models_25(
        matches_filtered: pd.DataFrame,
        features: list,
        min_samples: int = 200,  # validation gate: min predicted positives across folds
        min_test_samples: int = 100,  # test gate: min predicted positives on hold-out test
        precision_test_threshold: float = 0.80,
        base_model: str = "xgb",  # "xgb" or "mlp"
        search_mode: str = "random",  # "random" or "grid"
        n_random_param_sets: int = 10,
        cpu_jobs: int = 6,
        top_k: int = 10,
        thresholds: np.ndarray | None = None,
        out_dir: str | None = None,
        # --- anti-overfitting knobs ---
        val_conf_level: float = 0.99,  # Wilson-LCB confidence for validation precision
        max_precision_drop: float = 0.02,  # allow at most 10pp drop val → test
        # --- failure handling ---
        on_fail: str = "return",  # "return" | "warn" | "raise"
        save_diagnostics_on_fail: bool = True,
        market: str = "OVER"
):
    """
    Rolling time-ordered CV (no leakage) with calibration and robust failure handling.

    Random every run:
      - Hyper-parameter samples change each call (fresh entropy).
      - Model random_state changes each call (stable within-run per param set).

    Selection:
      1) Evaluate candidates on rolling validation across thresholds; track TP/FP.
      2) Rank by Wilson LCB of validation precision, then mean val precision, n_preds_val, val_accuracy.
      3) Take Top-K by that ranking. Fit on pre-test, calibrate on small final-val window,
         evaluate on TEST at each chosen threshold.
      4) TEST GATE: keep ONLY rows with
            n_preds_test ≥ min_test_samples
         AND test_precision ≥ max(precision_test_threshold, val_precision - max_precision_drop).
      5) Save CSV with ONLY survivors (includes 'model_pkl' for the top row), and save PKL for the top survivor.

    Failure case (no survivors):
      - If save_diagnostics_on_fail=True, writes a *_FAILED.csv with all Top-K test results and 'fail_reason'.
      - Behaviour controlled by `on_fail`: "return" (default), "warn", or "raise".
    """
    # ------------------ Imports & setup ------------------
    import os, secrets, hashlib
    from datetime import datetime
    import numpy as np
    import pandas as pd
    from itertools import product
    from math import sqrt
    from sklearn.model_selection import ParameterSampler
    from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, log_loss, brier_score_loss
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from joblib import Parallel, delayed, parallel_backend
    from tqdm import tqdm
    from tqdm_joblib import tqdm_joblib
    import joblib

    # xgboost (optional)
    try:
        import xgboost as xgb
        _HAS_XGB_LOCAL = True
    except Exception:
        _HAS_XGB_LOCAL = False

    # fallbacks for distributions if not defined at module scope
    try:
        _randint  # noqa: F821
        _uniform  # noqa: F821
        _loguniform  # noqa: F821
    except NameError:
        from scipy.stats import randint as _randint
        from scipy.stats import uniform as _uniform
        from scipy.stats import loguniform as _loguniform

    # normal quantile for Wilson LCB
    try:
        from scipy.stats import norm
        _Z = lambda conf: float(norm.ppf(1 - (1 - conf) / 2))
    except Exception:
        _Z = lambda conf: 1.96 if abs(conf - 0.95) < 1e-6 else 1.64  # crude fallback

    def _wilson_lcb(tp: int, fp: int, conf: float) -> float:
        n = tp + fp
        if n <= 0:
            return 0.0
        p = tp / n
        z = _Z(conf)
        denom = 1.0 + (z * z) / n
        centre = p + (z * z) / (2 * n)
        rad = z * sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
        return max(0.0, (centre - rad) / denom)

    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)

    out_dir = out_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    # honour external _HAS_XGB if present, else use local probe
    _HAS_XGB = globals().get("_HAS_XGB", _HAS_XGB_LOCAL)
    if base_model == "xgb" and not _HAS_XGB:
        raise ImportError("XGBoost not available; set base_model='mlp' or install xgboost.")

    # Fresh run seed → different every invocation
    RUN_SEED = secrets.randbits(32)

    def _seed_from(*vals) -> int:
        """Derive a stable (within-run) 31-bit positive seed from RUN_SEED and provided values."""
        h = hashlib.blake2b(digest_size=8)
        h.update(int(RUN_SEED).to_bytes(8, 'little', signed=False))
        for v in vals:
            h.update(str(v).encode('utf-8'))
        return int.from_bytes(h.digest(), 'little') & 0x7FFFFFFF

    # ------------------ Data prep ------------------
    req_cols = {'date', 'target'}
    missing = req_cols - set(matches_filtered.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = matches_filtered.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    cols_needed = list(set(features) | {'target'})
    df = df.dropna(subset=cols_needed).reset_index(drop=True)

    X = df[features].copy()
    y = df['target'].astype(int).reset_index(drop=True)

    n = len(X)
    if n < max(min_samples * 3, 500):
        raise RuntimeError(f"Not enough rows: {n}")

    # ------------------ Temporal split ------------------
    test_start = int(0.85 * n)
    pretest_end = test_start

    X_test = X.iloc[test_start:].reset_index(drop=True)
    y_test = y.iloc[test_start:].reset_index(drop=True)

    # Rolling folds inside [0, pretest_end)
    N_FOLDS = 5
    total_val_len = max(1, int(0.15 * n))
    val_len = max(1, total_val_len // N_FOLDS)
    fold_val_ends = [pretest_end - total_val_len + (i + 1) * val_len for i in range(N_FOLDS)]
    fold_val_starts = [end - val_len for end in fold_val_ends]
    if fold_val_ends:
        fold_val_ends[-1] = min(fold_val_ends[-1], pretest_end)
        fold_val_starts[-1] = max(0, fold_val_ends[-1] - val_len)

    # Final small validation slice (for calibration before test)
    final_val_len = max(1, val_len)
    final_val_start = max(0, test_start - final_val_len)
    X_train_final = X.iloc[:final_val_start]
    y_train_final = y.iloc[:final_val_start]
    X_val_final = X.iloc[final_val_start:test_start]
    y_val_final = y.iloc[final_val_start:test_start]

    # ------------------ Hyper-parameter spaces ------------------
    xgb_param_grid = {
        'n_estimators': [200],
        'max_depth': [5],
        'learning_rate': [0.1],
        'subsample': [0.7],
        'colsample_bytree': [1.0],
        'min_child_weight': [5],
        'reg_lambda': [1.0],
    }
    xgb_param_distributions = {
        'n_estimators': _randint(100, 1001),
        'max_depth': _randint(3, 8),
        'learning_rate': _loguniform(0.01, 0.2),
        'min_child_weight': _randint(3, 13),
        'subsample': _uniform(0.7, 0.3),
        'colsample_bytree': _uniform(0.6, 0.4),
        'reg_lambda': _loguniform(0.1, 10.0),
    }
    mlp_param_grid = {
        'hidden_layer_sizes': [(128,), (256,), (128, 64)],
        'alpha': [1e-4],
        'learning_rate_init': [1e-3],
        'batch_size': ['auto'],
        'max_iter': [200],
    }
    mlp_param_distributions = {
        'hidden_layer_sizes': [(64,), (128,), (256,), (128, 64), (256, 128)],
        'alpha': _loguniform(1e-5, 1e-2),
        'learning_rate_init': _loguniform(5e-4, 5e-2),
        'batch_size': _randint(32, 257),
        'max_iter': _randint(150, 401),
    }

    if search_mode.lower() == "grid":
        grid, dists = (xgb_param_grid, None) if base_model == "xgb" else (mlp_param_grid, None)
        all_param_dicts = [dict(zip(grid.keys(), combo)) for combo in product(*grid.values())]
    else:
        grid, dists = (xgb_param_grid, xgb_param_distributions) if base_model == "xgb" else (mlp_param_grid,
                                                                                             mlp_param_distributions)
        # Random every run
        sampler_seed = _seed_from("sampler")
        all_param_dicts = list(ParameterSampler(dists, n_iter=n_random_param_sets, random_state=sampler_seed))

    # ------------------ Helpers ------------------
    def cast_params(p: dict) -> dict:
        q = dict(p)
        if base_model == "xgb":
            for k in ['n_estimators', 'max_depth', 'min_child_weight']:
                if k in q: q[k] = int(q[k])
            for k in ['learning_rate', 'subsample', 'colsample_bytree', 'reg_lambda']:
                if k in q: q[k] = float(q[k])
        else:
            if 'max_iter' in q: q['max_iter'] = int(q['max_iter'])
            if 'batch_size' in q and q['batch_size'] != 'auto':
                q['batch_size'] = int(q['batch_size'])
            if 'alpha' in q: q['alpha'] = float(q['alpha'])
            if 'learning_rate_init' in q: q['learning_rate_init'] = float(q['learning_rate_init'])
            if 'hidden_layer_sizes' in q:
                h = q['hidden_layer_sizes']
                q['hidden_layer_sizes'] = tuple(h) if not isinstance(h, tuple) else h
        return q

    def _final_step_name(estimator):
        try:
            if isinstance(estimator, Pipeline):
                return estimator.steps[-1][0]
        except Exception:
            pass
        return None

    def build_model(params: dict, spw: float):
        # model seed depends on RUN_SEED and params → changes each run
        model_seed = _seed_from("model", base_model, tuple(sorted(params.items())))
        if base_model == "xgb":
            return xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=model_seed,
                scale_pos_weight=spw,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
                **params
            )
        else:
            mlp = MLPClassifier(
                random_state=model_seed,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
                solver="adam",
                **params
            )
            return make_pipeline(StandardScaler(with_mean=True, with_std=True), mlp)

    def fit_model(model, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None):
        if base_model == "xgb":
            try:
                model.set_params(verbosity=0, early_stopping_rounds=50)
                if X_va is not None and y_va is not None:
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                else:
                    model.fit(X_tr, y_tr, verbose=False)
            except Exception:
                model.fit(X_tr, y_tr, verbose=False)
        else:
            fit_kwargs = {}
            if sample_weight is not None:
                stepname = _final_step_name(model)
                if stepname is not None:
                    fit_kwargs[f"{stepname}__sample_weight"] = sample_weight
            try:
                model.fit(X_tr, y_tr, **fit_kwargs)
            except TypeError:
                model.fit(X_tr, y_tr)

    def fit_calibrator(fitted, X_va, y_va):
        try:
            from sklearn.calibration import FrozenEstimator
            frozen = FrozenEstimator(fitted)
            cal = CalibratedClassifierCV(frozen, method='sigmoid', cv=None)
            cal.fit(X_va, y_va)
            return cal
        except Exception:
            try:
                cal = CalibratedClassifierCV(fitted, method='sigmoid', cv='prefit')
                cal.fit(X_va, y_va)
                return cal
            except Exception:
                return fitted

    def predict_proba_1(model_or_cal, X_):
        proba = model_or_cal.predict_proba(X_)
        return proba[:, 1].astype(np.float32) if proba.ndim == 2 else np.asarray(proba, dtype=np.float32)

    # ------------------ Rolling-CV evaluator (VALIDATION) ------------------
    def evaluate_param_set(param_dict, task_id=None, total_tasks=None):
        safe = cast_params(param_dict)
        rows = []
        val_prob_all, val_true_all = [], []

        for vstart, vend in zip(fold_val_starts, fold_val_ends):
            if vstart is None or vend is None or vstart <= 0 or vend <= vstart:
                continue

            X_tr, y_tr = X.iloc[:vstart], y.iloc[:vstart]
            X_va, y_va = X.iloc[vstart:vend], y.iloc[vstart:vend]

            pos = int(y_tr.sum());
            neg = len(y_tr) - pos
            spw = (neg / pos) if pos > 0 else 1.0

            sample_weight = None
            if base_model == "mlp":
                w_pos = spw
                sample_weight = np.where(y_tr.values == 1, w_pos, 1.0).astype(np.float32)

            model = build_model(safe, spw)
            fit_model(model, X_tr, y_tr, X_va, y_va, sample_weight=sample_weight)

            cal = fit_calibrator(model, X_va, y_va)
            proba_va = predict_proba_1(cal, X_va)

            val_prob_all.append(proba_va)
            y_true = y_va.values.astype(np.uint8)
            val_true_all.append(y_true)

            for thr in thresholds:
                y_pred = (proba_va >= thr).astype(np.uint8)
                n_preds = int(y_pred.sum())
                tp = int(((y_true == 1) & (y_pred == 1)).sum())
                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                prc = precision_score(y_va, y_pred, zero_division=0)
                acc = accuracy_score(y_va, y_pred)

                rows.append({
                    **safe,
                    'threshold': float(thr),
                    'fold_vstart': int(vstart),
                    'fold_vend': int(vend),
                    'n_preds_val': n_preds,
                    'tp_val': tp,
                    'fp_val': fp,
                    'val_precision': float(prc),
                    'val_accuracy': float(acc),
                })

        # pooled diagnostics (optional)
        if val_prob_all:
            vp = np.concatenate(val_prob_all, axis=0)
            vt = np.concatenate(val_true_all, axis=0)
            try:
                val_auc = float(roc_auc_score(vt, vp))
            except Exception:
                val_auc = np.nan
            try:
                val_ll = float(log_loss(vt, vp, labels=[0, 1]))
            except Exception:
                val_ll = np.nan
            try:
                val_bri = float(brier_score_loss(vt, vp))
            except Exception:
                val_bri = np.nan
        else:
            val_auc = val_ll = val_bri = np.nan

        for r in rows:
            r['val_auc'] = val_auc
            r['val_logloss'] = val_ll
            r['val_brier'] = val_bri

        return rows

    # ------------------ Parallel parameter search ------------------
    total_tasks = len(all_param_dicts)
    if base_model == "mlp":
        eff_jobs = min(max(1, cpu_jobs), 4)
        prefer = "threads";
        backend = "threading";
        pre_dispatch = eff_jobs
        ctx = parallel_backend(backend, n_jobs=eff_jobs)
    else:
        eff_jobs = max(1, min(cpu_jobs, 4)) if cpu_jobs != -1 else 4
        prefer = "processes";
        backend = "loky";
        pre_dispatch = f"{2 * eff_jobs}"
        ctx = parallel_backend(backend, n_jobs=eff_jobs, inner_max_num_threads=1)

    with ctx:
        try:
            with tqdm_joblib(tqdm(total=total_tasks, desc=f"Param search ({search_mode}, {base_model})")) as _:
                out = Parallel(
                    n_jobs=eff_jobs, batch_size=1, prefer=prefer, pre_dispatch=pre_dispatch
                )(
                    delayed(evaluate_param_set)(pd_, i, total_tasks)
                    for i, pd_ in enumerate(all_param_dicts)
                )
        except OSError as e:
            print(f"[WARN] Parallel failed with {e}. Falling back to serial search...")
            out = []
            for i, pd_ in enumerate(tqdm(all_param_dicts, desc=f"Param search (serial, {base_model})")):
                out.append(evaluate_param_set(pd_, i, total_tasks))

    val_rows = [r for sub in out for r in sub]
    if not val_rows:
        raise RuntimeError("No validation rows produced (check folds and input data).")
    val_df = pd.DataFrame(val_rows)

    # ------------------ Aggregate across folds (VALIDATION) ------------------
    param_keys = list((xgb_param_grid if base_model == "xgb" else mlp_param_grid).keys())
    group_cols = param_keys + ['threshold']
    agg = val_df.groupby(group_cols, as_index=False).agg({
        'n_preds_val': 'sum',
        'tp_val': 'sum',
        'fp_val': 'sum',
        'val_precision': 'mean',
        'val_accuracy': 'mean',
        'val_auc': 'mean',
        'val_logloss': 'mean',
        'val_brier': 'mean',
    })

    # Wilson-LCB & pooled precision
    agg['val_precision_pooled'] = agg.apply(
        lambda r: (r['tp_val'] / max(1, (r['tp_val'] + r['fp_val']))), axis=1
    )
    agg['val_precision_lcb'] = agg.apply(
        lambda r: _wilson_lcb(int(r['tp_val']), int(r['fp_val']), conf=val_conf_level), axis=1
    )

    # Validation gates (mean precision + count)
    qual = agg[
        (agg['val_precision'] >= float(precision_test_threshold)) &
        (agg['n_preds_val'] >= int(min_samples))
        ].copy()
    if qual.empty:
        # nothing qualifies even on validation → treat as failure early
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_diagnostics_on_fail:
            fail_path = os.path.join(out_dir, f"model_metrics_{timestamp}_FAILED.csv")
            (agg.sort_values(['val_precision_lcb', 'val_precision', 'n_preds_val', 'val_accuracy'],
                             ascending=[False, False, False, False])
             .assign(fail_reason="failed_validation_gate")
             .to_csv(fail_path, index=False))
        msg = (f"No strategy met validation gates (precision ≥ {precision_test_threshold} "
               f"and n_preds_val ≥ {min_samples}).")
        if on_fail == "raise":
            raise RuntimeError(msg)
        if on_fail == "warn":
            print("[WARN]", msg)
        return {
            'status': 'failed_validation_gate',
            'csv': fail_path if save_diagnostics_on_fail else None,
            'model_pkl': None,
            'summary_df': None,
            'validation_table': agg.sort_values(['val_precision_lcb', 'val_precision', 'n_preds_val', 'val_accuracy'],
                                                ascending=[False, False, False, False]).reset_index(drop=True)
        }

    # STRICT validation ordering (conservative first)
    ranked = qual.sort_values(
        by=['val_precision_lcb', 'val_precision', 'n_preds_val', 'val_accuracy'],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    topk_val = ranked.head(top_k).reset_index(drop=True)

    # ------------------ Evaluate ALL Top-K on TEST ------------------
    candidates = []
    for _, row in topk_val.iterrows():
        candidates.append({
            'params': {k: row[k] for k in param_keys if k in row.index},
            'threshold': float(row['threshold']),
            'val_precision': float(row['val_precision']),
            'val_precision_lcb': float(row['val_precision_lcb']),
            'val_accuracy': float(row['val_accuracy']),
            'n_preds_val': int(row['n_preds_val']),
        })

    # Evaluate each Top-K candidate on TEST
    records_all = []  # every candidate with test metrics + pass/fail reason
    for cand in candidates:
        best_params = cast_params(cand['params'])
        pos = int(y_train_final.sum());
        neg = len(y_train_final) - pos
        spw_final = (neg / pos) if pos > 0 else 1.0

        final_model = build_model(best_params, spw_final)
        final_sample_weight = None
        if base_model == "mlp":
            w_pos = spw_final
            final_sample_weight = np.where(y_train_final.values == 1, w_pos, 1.0).astype(np.float32)

        fit_model(final_model, X_train_final, y_train_final, X_val_final, y_val_final,
                  sample_weight=final_sample_weight)
        final_calibrator = fit_calibrator(final_model, X_val_final, y_val_final)

        y_test_proba = predict_proba_1(final_calibrator, X_test)
        thr = cand['threshold']
        y_pred = (y_test_proba >= thr).astype(np.uint8)
        n_preds_test = int(y_pred.sum())
        prc_test = precision_score(y_test, y_pred, zero_division=0)
        acc_test = accuracy_score(y_test, y_pred)

        # TEST GATE checks + reason
        enough = n_preds_test >= int(min_test_samples)
        not_collapsed = prc_test >= max(float(precision_test_threshold),
                                        float(cand['val_precision']) - float(max_precision_drop))
        pass_gate = bool(enough and not_collapsed)
        reason = ""
        if not pass_gate:
            if not enough and not not_collapsed:
                reason = "insufficient_test_preds_and_precision_collapse"
            elif not enough:
                reason = "insufficient_test_preds"
            else:
                reason = "precision_collapse"

        records_all.append({
            **cand['params'],
            'threshold': thr,
            'val_precision_lcb': cand['val_precision_lcb'],
            'val_precision': cand['val_precision'],
            'val_accuracy': cand['val_accuracy'],
            'n_preds_val': cand['n_preds_val'],
            'n_preds_test': n_preds_test,
            'test_precision': float(prc_test),
            'test_accuracy': float(acc_test),
            'pass_test_gate': pass_gate,
            'fail_reason': reason,
            'model_obj': final_calibrator if pass_gate else None,
        })

    survivors_df = pd.DataFrame(records_all)
    passers = survivors_df[survivors_df['pass_test_gate']].copy()

    # ------------------ Persist outputs ------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "xgb" if base_model == "xgb" else "mlp"

    if passers.empty:
        # No survivor: write diagnostics, optionally return or warn instead of raising
        if save_diagnostics_on_fail:
            diag = (survivors_df
                    .drop(columns=['model_obj'])
                    .sort_values(by=['val_precision_lcb', 'val_precision', 'n_preds_val', 'val_accuracy'],
                                 ascending=[False, False, False, False])
                    )
            fail_csv = os.path.join(out_dir, f"model_metrics_{timestamp}_FAILED.csv")
            diag.to_csv(fail_csv, index=False)
        msg = (f"All Top-{len(candidates)} failed the TEST gate "
               f"(n_preds_test ≥ {min_test_samples} and precision not collapsing).")
        if on_fail == "raise":
            raise RuntimeError(msg)
        if on_fail == "warn":
            print("[WARN]", msg)
        return {
            'status': 'failed_test_gate',
            'csv': fail_csv if save_diagnostics_on_fail else None,
            'model_pkl': None,
            'summary_df': diag if save_diagnostics_on_fail else survivors_df.drop(columns=['model_obj']),
            'validation_table': ranked,
        }

    # At least one survivor → choose best & save PKL + CSV (only passers)
    passers_sorted = passers.sort_values(
        by=['val_precision_lcb', 'val_precision', 'test_precision', 'n_preds_test', 'val_accuracy'],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)

    # Save PKL next to CSVs
    if market == 'OVER':
        pkl_path = os.path.join(r"C:\Users\leere\PycharmProjects\Football_ML3\Goals\Over_2_5\model_file",
                                f"best_model_{tag}_calibrated_{timestamp}.pkl")
    else:
        pkl_path = os.path.join(r"C:\Users\leere\PycharmProjects\Football_ML3\Goals\Under_2_5\model_file",
                                f"best_model_{tag}_calibrated_{timestamp}.pkl")

    # Prepare CSV (include the PKL path for the winning row only)
    csv_df = passers_sorted.drop(columns=['model_obj']).copy()
    csv_df['model_pkl'] = ""
    csv_df.loc[0, 'model_pkl'] = pkl_path

    csv_path = os.path.join(out_dir, f"model_metrics_{timestamp}.csv")
    csv_df.to_csv(csv_path, index=False)

    # Save PKL for the top row
    top_row = passers_sorted.iloc[0]
    chosen_model = top_row['model_obj']
    chosen_params = {k: top_row[k] for k in param_keys if k in passers_sorted.columns}
    chosen_threshold = float(top_row['threshold'])

    joblib.dump(
        {
            'model': chosen_model,
            'threshold': chosen_threshold,
            'features': features,
            'base_model': base_model,
            'best_params': chosen_params,
            'precision_test_threshold': float(precision_test_threshold),
            'min_samples': int(min_samples),
            'min_test_samples': int(min_test_samples),
            'val_conf_level': float(val_conf_level),
            'max_precision_drop': float(max_precision_drop),
            'notes': (
                'CSV includes only candidates passing test gate; ranked by '
                'val_precision_lcb → val_precision → test_precision → n_preds_test → val_accuracy. '
                'Seeds are random each run.'
            ),
            'run_seed': int(RUN_SEED),  # for traceability
        },
        pkl_path
    )

    return {
        'status': 'ok',
        'csv': csv_path,
        'model_pkl': pkl_path,
        'summary_df': csv_df,  # passers only, with model_pkl set on row 0
        'validation_table': ranked,  # full validation ranking (post-gates)
    }


# ---- Simple ersatz rvs helpers (NumPy 2.x compatible) ----
def _as_rng(random_state):
    """Return a NumPy Generator, accepting int | RandomState | Generator | None."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        # Pull a 32-bit value from the legacy MT19937 state (safe for seeding)
        seed = int(np.uint32(random_state.get_state()[1][0]))
        return np.random.default_rng(seed)
    if random_state is None or isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(None if random_state is None else int(random_state))
    # Fallback: hash arbitrary objects into a 32-bit seed
    return np.random.default_rng(abs(hash(random_state)) % (2 ** 32))


def _randint(low, high):
    class _R:
        def rvs(self, size=None, random_state=None):
            rng = _as_rng(random_state)
            return rng.integers(low, high, size=size)

    return _R()


def _uniform(a, b):
    # draws from [a, a+b)
    class _R:
        def rvs(self, size=None, random_state=None):
            rng = _as_rng(random_state)
            return rng.uniform(a, a + b, size=size)

    return _R()


def _loguniform(a, b):
    # log-uniform on [a, b]
    class _R:
        def rvs(self, size=None, random_state=None):
            rng = _as_rng(random_state)
            lo, hi = np.log(a), np.log(b)
            return np.exp(rng.uniform(lo, hi, size=size))

    return _R()


# def run_models_with_probs(data, features, filename_feature, min_samples=100, apply_calibration=True):
#     """
#     Run grid-search experiments over different models.
#     If apply_pca is True, PCA is applied using various variance thresholds.
#     If apply_calibration is True, probability calibration is applied to each fitted model
#     using the sigmoid method (Platt scaling).
#     The metrics computed include standard evaluation metrics plus summary statistics
#     of the calibrated probabilities.
#     """
#     print("Data length:", len(data))
#     print("Total positive targets:", data['target'].sum())
#
#     # Separate features and target
#     X = data[features]
#     y = data['target']
#
#     # Time-series split: first 80% for training, last 20% for testing
#     train_size = int(len(data) * 0.8)
#     X_train_full = X.iloc[:train_size]
#     X_test = X.iloc[train_size:]
#     y_train_full = y.iloc[:train_size]
#     y_test = y.iloc[train_size:]
#
#     counts = Counter(y_train_full)
#     minority_class = min(counts, key=counts.get)
#     majority_class = max(counts, key=counts.get)
#     current_ratio = counts[minority_class] / counts[majority_class]
#     print("Current minority/majority ratio:", current_ratio)
#
#     # Define SMOTE strategies and probability thresholds.
#     upper_bound = 0.95  # adjust as needed
#     smote_strategies = [round(x, 2) for x in np.arange(current_ratio + 0.01, upper_bound, 0.05)] + [None]
#     probability_thresholds = [round(x, 2) for x in np.arange(0.2, 0.81, 0.01)]
#
#     metrics_list = []
#
#     for apply_pca in [True, False]:
#         if apply_pca:
#             var_thresholds = [0.9, 0.92, 0.94, 0.96, 0.98]
#         else:
#             var_thresholds = [1]
#
#         # Compute master total tests.
#         temp_pipelines, temp_param_grids = build_pipelines(apply_pca=apply_pca)
#         master_total_tests = 0
#         for model in temp_param_grids:
#             num_param = len(list(ParameterGrid(temp_param_grids[model])))
#             master_total_tests += len(var_thresholds) * len(smote_strategies) * num_param * len(probability_thresholds)
#         print(f"Total tests to be performed: {master_total_tests}")
#
#         if apply_pca:
#             total_outer_runs = len(var_thresholds) * len(smote_strategies) * len(temp_pipelines)
#         else:
#             total_outer_runs = len(smote_strategies) * len(temp_pipelines)
#         outer_run_counter = 0
#         print(f"Total grid search outer runs: {total_outer_runs}")
#
#         master_test_counter = 0
#
#         # Loop over whether to use PCA or not.
#         for pca_flag in [apply_pca]:
#             if pca_flag:
#                 for var_threshold in var_thresholds:
#                     optimal_n_components = select_optimal_pca_components(X_train_full, variance_threshold=var_threshold)
#                     print(f"PCA: {optimal_n_components} components for {var_threshold * 100:.0f}% variance")
#                     pipelines, param_grids = build_pipelines(apply_pca=True)
#                     for model_name, pipeline in pipelines.items():
#                         pipeline.set_params(pca__n_components=optimal_n_components)
#                     for sample_st in smote_strategies:
#                         if sample_st is not None:
#                             smote = SMOTE(sampling_strategy=sample_st, random_state=42)
#                             X_train_res, y_train_res = smote.fit_resample(X_train_full, y_train_full)
#                             smote_label = sample_st
#                         else:
#                             X_train_res, y_train_res = X_train_full, y_train_full
#                             smote_label = "None"
#                         print(f"SMOTE level: {smote_label}")
#                         for model_name in pipelines.keys():
#                             outer_run_counter += 1
#                             num_params = len(list(ParameterGrid(param_grids[model_name])))
#                             print(
#                                 f"Run {outer_run_counter}/{total_outer_runs} - Model: {model_name}, SMOTE: {smote_label}, Params: {num_params}")
#                             for params in ParameterGrid(param_grids[model_name]):
#                                 pipeline = pipelines[model_name]
#                                 pipeline.set_params(**params)
#                                 pipeline.fit(X_train_res, y_train_res)
#
#                                 # Calibrate probabilities if requested.
#                                 if apply_calibration:
#                                     calibrated_pipeline = CalibratedClassifierCV(estimator=pipeline,
#                                                                                  method='sigmoid',
#                                                                                  cv='prefit')
#                                     calibrated_pipeline.fit(X_train_res, y_train_res)
#                                     current_pipeline = calibrated_pipeline
#                                 else:
#                                     current_pipeline = pipeline
#
#                                 train_probs = current_pipeline.predict_proba(X_train_res)[:, 1]
#                                 test_probs = current_pipeline.predict_proba(X_test)[:, 1]
#                                 cal_train_mean = np.mean(train_probs)
#                                 cal_test_mean = np.mean(test_probs)
#
#                                 for thresh in probability_thresholds:
#                                     master_test_counter += 1
#                                     train_pred = (train_probs >= thresh).astype(int)
#                                     test_pred = (test_probs >= thresh).astype(int)
#                                     if np.sum(test_pred) < min_samples:
#                                         continue
#                                     train_sample_size = np.sum(train_pred)
#                                     test_sample_size = np.sum(test_pred)
#
#                                     train_mcc = matthews_corrcoef(y_train_res, train_pred)
#                                     test_mcc = matthews_corrcoef(y_test, test_pred)
#                                     mcc_ratio = test_mcc / (train_mcc + 1e-10)
#                                     if mcc_ratio > 1:
#                                         mcc_ratio = train_mcc / (test_mcc + 1e-10)
#
#                                     train_acc = accuracy_score(y_train_res, train_pred)
#                                     test_acc = accuracy_score(y_test, test_pred)
#                                     acc_ratio = test_acc / (train_acc + 1e-10)
#                                     if acc_ratio > 1:
#                                         acc_ratio = train_acc / (test_acc + 1e-10)
#
#                                     train_f1 = f1_score(y_train_res, train_pred)
#                                     test_f1 = f1_score(y_test, test_pred)
#                                     f1_ratio = test_f1 / (train_f1 + 1e-10)
#                                     if f1_ratio > 1:
#                                         f1_ratio = train_f1 / (test_f1 + 1e-10)
#
#                                     train_auc = roc_auc_score(y_train_res, train_probs)
#                                     test_auc = roc_auc_score(y_test, test_probs)
#                                     auc_ratio = test_auc / (train_auc + 1e-10)
#                                     if auc_ratio > 1:
#                                         auc_ratio = train_auc / (test_auc + 1e-10)
#
#                                     train_precision = precision_score(y_train_res, train_pred, zero_division=0)
#                                     test_precision = precision_score(y_test, test_pred, zero_division=0)
#                                     precision_ratio = test_precision / (train_precision + 1e-10)
#                                     if precision_ratio > 1:
#                                         precision_ratio = train_precision / (test_precision + 1e-10)
#
#                                     train_recall = recall_score(y_train_res, train_pred, zero_division=0)
#                                     test_recall = recall_score(y_test, test_pred, zero_division=0)
#                                     recall_ratio = test_recall / (train_recall + 1e-10)
#                                     if recall_ratio > 1:
#                                         recall_ratio = train_recall / (test_recall + 1e-10)
#
#                                     if (auc_ratio > 0.8 and precision_ratio > 0.9):
#                                         metrics_list.append({
#                                             'Model': model_name,
#                                             'SMOTE': smote_label,
#                                             'Probability_Threshold': thresh,
#                                             'AUC_Train': round(train_auc, 4),
#                                             'AUC_Test': round(test_auc, 4),
#                                             'AUC_Ratio': round(auc_ratio, 4),
#                                             'Precision_Train': round(train_precision, 4),
#                                             'Precision_Test': round(test_precision, 4),
#                                             'Precision_Ratio': round(precision_ratio, 4),
#                                             'MCC_Train': round(train_mcc, 4),
#                                             'MCC_Test': round(test_mcc, 4),
#                                             'MCC_Ratio': round(mcc_ratio, 4),
#                                             'ACC_Train': round(train_acc, 4),
#                                             'ACC_Test': round(test_acc, 4),
#                                             'ACC_Ratio': round(acc_ratio, 4),
#                                             'F1_Train': round(train_f1, 4),
#                                             'F1_Test': round(test_f1, 4),
#                                             'F1_Ratio': round(f1_ratio, 4),
#                                             'Recall_Train': round(train_recall, 4),
#                                             'Recall_Test': round(test_recall, 4),
#                                             'Recall_Ratio': round(recall_ratio, 4),
#                                             'Train_Sample_Size': train_sample_size,
#                                             'Test_Sample_Size': test_sample_size,
#                                             'Calibrated_Train_Prob_Mean': round(cal_train_mean, 4),
#                                             'Calibrated_Test_Prob_Mean': round(cal_test_mean, 4),
#                                             'Var_Threshold': var_threshold,
#                                             'Params': params
#                                         })
#             else:
#                 # Running models without PCA (similar logic; calibration can be applied similarly)
#                 print("Running models without PCA not shown here for brevity.")
#
#     print(f"Total tests performed: {master_test_counter}")
#     filename_out = f"model_metrics_{filename_feature}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
#     metrics_df = pd.DataFrame(metrics_list)
#     metrics_df.sort_values(by='Precision_Test', ascending=False, inplace=True)
#     metrics_df.to_csv(filename_out, index=False)
#     print(f"Model metrics saved to {filename_out}")
#
#
# def running_ttest_p_profit(x):
#     """
#     One-sided (greater) t-test on sample x against popmean=0.
#     Returns p-value for H1: mean > 0.
#     """
#     x = np.asarray(x)
#     if len(x) < 2:
#         return np.nan
#     t_stat, p_two = stats.ttest_1samp(x, popmean=0, nan_policy='omit')
#     return (p_two / 2) if (t_stat > 0) else (1 - p_two / 2)
#
#
# models_params = [
#     (
#         'XGBoost',
#         XGBClassifier,
#         {
#             # Trees / learning‐rate trade‐off
#             'n_estimators': [100, 300],
#             'learning_rate': [0.05, 0.1],
#             # Shallow vs moderate depth
#             'max_depth': [3, 5],
#             # Row & column sampling
#             'subsample': [0.8, 1.0],
#             'colsample_bytree': [0.8],
#             # Light regularisation
#             'gamma': [0.0, 0.1],
#             'reg_alpha': [0.0, 0.1],
#             'reg_lambda': [1.0],
#             # for imbalanced data you can compute:
#             # 'scale_pos_weight': [sum(neg)/sum(pos)]
#         }
#     ),
#     (
#         'MLP',
#         MLPClassifier,
#         {
#             # small vs medium single‐layer nets
#             'hidden_layer_sizes': [(50,), (100,)],
#             # L2 penalty
#             'alpha': [1e-4, 1e-3, 1e-2],
#             # fixed learning rate
#             'learning_rate_init': [1e-3],
#             'max_iter': [1000],
#             'early_stopping': [True],
#             # you get a 10%‐validation split for the stopping criterion by default
#         }
#     )
# ]
#
#
# def run_models_outcome(
#     matches_filtered: pd.DataFrame,
#     features: list,
#     min_samples: int = 200,            # validation gate: min predicted positives across folds
#     min_test_samples: int = 100,       # test gate: min predicted positives on hold-out test
#     precision_test_threshold: float = 0.80,
#     base_model: str = "xgb",           # "xgb" or "mlp"
#     search_mode: str = "random",       # "random" or "grid"
#     n_random_param_sets: int = 10,
#     cpu_jobs: int = 6,
#     top_k: int = 10,
#     thresholds: np.ndarray | None = None,
#     out_dir: str | None = None,        # if None, CSV saved to per-market metrics dir
#     # --- anti-overfitting knobs ---
#     val_conf_level: float = 0.99,      # Wilson-LCB confidence for validation precision
#     max_precision_drop: float = 0.05,  # allow at most 2pp drop val → test
#     # --- failure handling ---
#     on_fail: str = "return",           # "return" | "warn" | "raise"
#     save_diagnostics_on_fail: bool = True,
#     # --- market selection ---
#     market: str = "LAY_AWAY",          # LAY_HOME | LAY_AWAY | LAY_DRAW | BACK_HOME | BACK_AWAY | BACK_DRAW | (optional: OVER | UNDER)
# ):
#     """
#     Rolling time-ordered CV (no leakage) with calibration and robust failure handling.
#
#     Target encoding (you must prepare df['target'] accordingly):
#       LAY_AWAY:  target=1 if AWAY does NOT win (H or D), else 0
#       LAY_HOME:  target=1 if HOME does NOT win (A or D), else 0
#       LAY_DRAW:  target=1 if NOT a draw (H or A), else 0
#
#       BACK_AWAY: target=1 if AWAY wins, else 0
#       BACK_HOME: target=1 if HOME wins, else 0
#       BACK_DRAW: target=1 if DRAW, else 0
#
#       (Optional)
#       OVER:      target=1 if goals > 2.5, else 0
#       UNDER:     target=1 if goals < 2.5, else 0
#     """
#     # ------------------ Imports & setup ------------------
#     import os, secrets, hashlib
#     from datetime import datetime
#     import numpy as np
#     import pandas as pd
#     from itertools import product
#     from math import sqrt
#     from sklearn.model_selection import ParameterSampler
#     from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, log_loss, brier_score_loss
#     from sklearn.calibration import CalibratedClassifierCV
#     from sklearn.pipeline import Pipeline, make_pipeline
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.neural_network import MLPClassifier
#     from joblib import Parallel, delayed, parallel_backend
#     from tqdm import tqdm
#     from tqdm_joblib import tqdm_joblib
#     import joblib
#
#     # ---------- Fixed save locations per market ----------
#     BASE = r"C:\Users\leere\PycharmProjects\Football_ML3\FT Results"
#     PKL_DIRS = {
#         "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "model_file"),
#         "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "model_file"),
#         "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "model_file"),
#         "BACK_HOME": os.path.join(BASE, "Back_Home", "model_file"),
#         "BACK_AWAY": os.path.join(BASE, "Back_Away", "model_file"),
#         "BACK_DRAW": os.path.join(BASE, "Back_Draw", "model_file"),
#         # keep goals markets if you still use them
#         "OVER":      os.path.join(BASE, "Over_2_5",  "model_file"),
#         "UNDER":     os.path.join(BASE, "Under_2_5", "model_file"),
#     }
#     CSV_DIRS = {
#         "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "best_model_metrics"),
#         "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "best_model_metrics"),
#         "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "best_model_metrics"),
#         "BACK_HOME": os.path.join(BASE, "Back_Home", "best_model_metrics"),
#         "BACK_AWAY": os.path.join(BASE, "Back_Away", "best_model_metrics"),
#         "BACK_DRAW": os.path.join(BASE, "Back_Draw", "best_model_metrics"),
#         "OVER":      os.path.join(BASE, "Over_2_5",  "best_model_metrics"),
#         "UNDER":     os.path.join(BASE, "Under_2_5", "best_model_metrics"),
#     }
#     market = str(market).upper().strip()
#     if market not in PKL_DIRS or market not in CSV_DIRS:
#         raise ValueError(f"Unsupported market '{market}'. Use one of: {sorted(PKL_DIRS)}")
#
#     # xgboost (optional)
#     try:
#         import xgboost as xgb
#         _HAS_XGB_LOCAL = True
#     except Exception:
#         _HAS_XGB_LOCAL = False
#
#     # fallbacks for distributions if not defined at module scope
#     try:
#         _randint  # noqa: F821
#         _uniform  # noqa: F821
#         _loguniform  # noqa: F821
#     except NameError:
#         from scipy.stats import randint as _randint
#         from scipy.stats import uniform as _uniform
#         from scipy.stats import loguniform as _loguniform
#
#     # normal quantile for Wilson LCB
#     try:
#         from scipy.stats import norm
#         _Z = lambda conf: float(norm.ppf(1 - (1 - conf) / 2))
#     except Exception:
#         _Z = lambda conf: 1.96 if abs(conf - 0.95) < 1e-6 else 1.64  # crude fallback
#
#     def _wilson_lcb(tp: int, fp: int, conf: float) -> float:
#         n = tp + fp
#         if n <= 0:
#             return 0.0
#         p = tp / n
#         z = _Z(conf)
#         denom = 1.0 + (z*z)/n
#         centre = p + (z*z)/(2*n)
#         rad = z * sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
#         return max(0.0, (centre - rad) / denom)
#
#     if thresholds is None:
#         thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)
#
#     # CSV save dir defaults to per-market folder unless out_dir explicitly provided
#     csv_save_dir = out_dir if (out_dir is not None and len(str(out_dir)) > 0) else CSV_DIRS[market]
#     os.makedirs(csv_save_dir, exist_ok=True)
#     model_dir = PKL_DIRS[market]
#     os.makedirs(model_dir, exist_ok=True)
#
#     # honour external _HAS_XGB if present, else use local probe
#     _HAS_XGB = globals().get("_HAS_XGB", _HAS_XGB_LOCAL)
#     if base_model == "xgb" and not _HAS_XGB:
#         raise ImportError("XGBoost not available; set base_model='mlp' or install xgboost.")
#
#     # Fresh run seed → different every invocation
#     RUN_SEED = secrets.randbits(32)
#
#     def _seed_from(*vals) -> int:
#         """Derive a stable (within-run) 31-bit positive seed from RUN_SEED and provided values."""
#         h = hashlib.blake2b(digest_size=8)
#         h.update(int(RUN_SEED).to_bytes(8, 'little', signed=False))
#         for v in vals:
#             h.update(str(v).encode('utf-8'))
#         return int.from_bytes(h.digest(), 'little') & 0x7FFFFFFF
#
#     # ------------------ Data prep ------------------
#     req_cols = {'date', 'target'}
#     missing = req_cols - set(matches_filtered.columns)
#     if missing:
#         raise ValueError(f"Missing required columns: {missing}")
#
#     df = matches_filtered.copy()
#     df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     df = df.sort_values('date').reset_index(drop=True)
#
#     cols_needed = list(set(features) | {'target'})
#     df = df.dropna(subset=cols_needed).reset_index(drop=True)
#
#     X = df[features].copy()
#     y = df['target'].astype(int).reset_index(drop=True)
#
#     n = len(X)
#     if n < max(min_samples * 3, 500):
#         raise RuntimeError(f"Not enough rows: {n}")
#
#     # ------------------ Temporal split ------------------
#     test_start = int(0.85 * n)
#     pretest_end = test_start
#
#     X_test = X.iloc[test_start:].reset_index(drop=True)
#     y_test = y.iloc[test_start:].reset_index(drop=True)
#
#     # Rolling folds inside [0, pretest_end)
#     N_FOLDS = 5
#     total_val_len = max(1, int(0.15 * n))
#     val_len = max(1, total_val_len // N_FOLDS)
#     fold_val_ends = [pretest_end - total_val_len + (i + 1) * val_len for i in range(N_FOLDS)]
#     fold_val_starts = [end - val_len for end in fold_val_ends]
#     if fold_val_ends:
#         fold_val_ends[-1] = min(fold_val_ends[-1], pretest_end)
#         fold_val_starts[-1] = max(0, fold_val_ends[-1] - val_len)
#
#     # Final small validation slice (for calibration before test)
#     final_val_len = max(1, val_len)
#     final_val_start = max(0, test_start - final_val_len)
#     X_train_final = X.iloc[:final_val_start]
#     y_train_final = y.iloc[:final_val_start]
#     X_val_final   = X.iloc[final_val_start:test_start]
#     y_val_final   = y.iloc[final_val_start:test_start]
#
#     # ------------------ Hyper-parameter spaces ------------------
#     xgb_param_grid = {
#         'n_estimators':      [200],
#         'max_depth':         [5],
#         'learning_rate':     [0.1],
#         'subsample':         [0.7],
#         'colsample_bytree':  [1.0],
#         'min_child_weight':  [5],
#         'reg_lambda':        [1.0],
#     }
#     xgb_param_distributions = {
#         'n_estimators':      _randint(100, 1001),
#         'max_depth':         _randint(3, 8),
#         'learning_rate':     _loguniform(0.01, 0.2),
#         'min_child_weight':  _randint(3, 13),
#         'subsample':         _uniform(0.7, 0.3),
#         'colsample_bytree':  _uniform(0.6, 0.4),
#         'reg_lambda':        _loguniform(0.1, 10.0),
#     }
#     mlp_param_grid = {
#         'hidden_layer_sizes': [(128,), (256,), (128, 64)],
#         'alpha':              [1e-4],
#         'learning_rate_init': [1e-3],
#         'batch_size':         ['auto'],
#         'max_iter':           [200],
#     }
#     mlp_param_distributions = {
#         'hidden_layer_sizes': [(64,), (128,), (256,), (128, 64), (256, 128)],
#         'alpha':              _loguniform(1e-5, 1e-2),
#         'learning_rate_init': _loguniform(5e-4, 5e-2),
#         'batch_size':         _randint(32, 257),
#         'max_iter':           _randint(150, 401),
#     }
#
#     if search_mode.lower() == "grid":
#         grid = xgb_param_grid if base_model == "xgb" else mlp_param_grid
#         all_param_dicts = [dict(zip(grid.keys(), combo)) for combo in product(*grid.values())]
#     else:
#         dists = xgb_param_distributions if base_model == "xgb" else mlp_param_distributions
#         sampler_seed = _seed_from("sampler")
#         all_param_dicts = list(ParameterSampler(dists, n_iter=n_random_param_sets, random_state=sampler_seed))
#
#     # ------------------ Helpers ------------------
#     def cast_params(p: dict) -> dict:
#         q = dict(p)
#         if base_model == "xgb":
#             for k in ['n_estimators', 'max_depth', 'min_child_weight']:
#                 if k in q: q[k] = int(q[k])
#             for k in ['learning_rate', 'subsample', 'colsample_bytree', 'reg_lambda']:
#                 if k in q: q[k] = float(q[k])
#         else:
#             if 'max_iter' in q: q['max_iter'] = int(q['max_iter'])
#             if 'batch_size' in q and q['batch_size'] != 'auto':
#                 q['batch_size'] = int(q['batch_size'])
#             if 'alpha' in q: q['alpha'] = float(q['alpha'])
#             if 'learning_rate_init' in q: q['learning_rate_init'] = float(q['learning_rate_init'])
#             if 'hidden_layer_sizes' in q:
#                 h = q['hidden_layer_sizes']
#                 q['hidden_layer_sizes'] = tuple(h) if not isinstance(h, tuple) else h
#         return q
#
#     def _final_step_name(estimator):
#         try:
#             if isinstance(estimator, Pipeline):
#                 return estimator.steps[-1][0]
#         except Exception:
#             pass
#         return None
#
#     def build_model(params: dict, spw: float):
#         model_seed = _seed_from("model", base_model, tuple(sorted(params.items())))
#         if base_model == "xgb":
#             return xgb.XGBClassifier(
#                 objective='binary:logistic',
#                 eval_metric='auc',
#                 random_state=model_seed,
#                 scale_pos_weight=spw,
#                 n_jobs=1,
#                 tree_method="hist",
#                 verbosity=0,
#                 **params
#             )
#         else:
#             mlp = MLPClassifier(
#                 random_state=model_seed,
#                 early_stopping=True,
#                 n_iter_no_change=20,
#                 validation_fraction=0.1,
#                 solver="adam",
#                 **params
#             )
#             return make_pipeline(StandardScaler(with_mean=True, with_std=True), mlp)
#
#     def fit_model(model, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None):
#         if base_model == "xgb":
#             try:
#                 model.set_params(verbosity=0, early_stopping_rounds=50)
#                 if X_va is not None and y_va is not None:
#                     model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
#                 else:
#                     model.fit(X_tr, y_tr, verbose=False)
#             except Exception:
#                 model.fit(X_tr, y_tr, verbose=False)
#         else:
#             fit_kwargs = {}
#             if sample_weight is not None:
#                 stepname = _final_step_name(model)
#                 if stepname is not None:
#                     fit_kwargs[f"{stepname}__sample_weight"] = sample_weight
#             try:
#                 model.fit(X_tr, y_tr, **fit_kwargs)
#             except TypeError:
#                 model.fit(X_tr, y_tr)
#
#     def fit_calibrator(fitted, X_va, y_va):
#         try:
#             from sklearn.calibration import FrozenEstimator
#             frozen = FrozenEstimator(fitted)
#             cal = CalibratedClassifierCV(frozen, method='sigmoid', cv=None)
#             cal.fit(X_va, y_va)
#             return cal
#         except Exception:
#             try:
#                 cal = CalibratedClassifierCV(fitted, method='sigmoid', cv='prefit')
#                 cal.fit(X_va, y_va)
#                 return cal
#             except Exception:
#                 return fitted
#
#     # --- robust proba extraction (handles single-class fits) ---
#     def _unwrap_estimator(est):
#         if isinstance(est, Pipeline):
#             return est.steps[-1][1]
#         return est
#
#     def predict_proba_1(model_or_cal, X_):
#         proba = model_or_cal.predict_proba(X_)
#         if proba.ndim == 2:
#             n_classes = proba.shape[1]
#             # try to locate column for class '1'
#             pos_idx = None
#             classes = getattr(model_or_cal, "classes_", None)
#             if classes is None:
#                 base = _unwrap_estimator(model_or_cal)
#                 classes = getattr(base, "classes_", None)
#             if classes is not None:
#                 try:
#                     idxs = np.where(np.asarray(classes) == 1)[0]
#                     if len(idxs):
#                         pos_idx = int(idxs[0])
#                 except Exception:
#                     pos_idx = None
#             if n_classes == 2 and pos_idx is not None:
#                 return proba[:, pos_idx].astype(np.float32)
#             if n_classes == 1:
#                 only = None
#                 if classes is not None and len(classes) == 1:
#                     only = int(classes[0])
#                 return (np.ones(proba.shape[0], dtype=np.float32)
#                         if only == 1 else
#                         np.zeros(proba.shape[0], dtype=np.float32))
#             return proba[:, min(1, n_classes - 1)].astype(np.float32)
#         return np.asarray(proba, dtype=np.float32)
#
#     # ------------------ Rolling-CV evaluator (VALIDATION) ------------------
#     def evaluate_param_set(param_dict, task_id=None, total_tasks=None):
#         safe = cast_params(param_dict)
#         rows = []
#         val_prob_all, val_true_all = [], []
#
#         for vstart, vend in zip(fold_val_starts, fold_val_ends):
#             if vstart is None or vend is None or vstart <= 0 or vend <= vstart:
#                 continue
#
#             X_tr, y_tr = X.iloc[:vstart], y.iloc[:vstart]
#             X_va, y_va = X.iloc[vstart:vend], y.iloc[vstart:vend]
#
#             # Skip this fold if training slice is single-class
#             if y_tr.nunique() < 2:
#                 continue
#
#             pos = int(y_tr.sum()); neg = len(y_tr) - pos
#             spw = (neg / pos) if pos > 0 else 1.0
#
#             sample_weight = None
#             if base_model == "mlp":
#                 w_pos = spw
#                 sample_weight = np.where(y_tr.values == 1, w_pos, 1.0).astype(np.float32)
#
#             model = build_model(safe, spw)
#             fit_model(model, X_tr, y_tr, X_va, y_va, sample_weight=sample_weight)
#
#             cal = fit_calibrator(model, X_va, y_va)
#             proba_va = predict_proba_1(cal, X_va)
#
#             val_prob_all.append(proba_va)
#             y_true = y_va.values.astype(np.uint8)
#             val_true_all.append(y_true)
#
#             for thr in thresholds:
#                 y_pred = (proba_va >= thr).astype(np.uint8)
#                 n_preds = int(y_pred.sum())
#                 tp = int(((y_true == 1) & (y_pred == 1)).sum())
#                 fp = int(((y_true == 0) & (y_pred == 1)).sum())
#                 prc = precision_score(y_va, y_pred, zero_division=0)
#                 acc = accuracy_score(y_va, y_pred)
#
#                 rows.append({
#                     **safe,
#                     'threshold': float(thr),
#                     'fold_vstart': int(vstart),
#                     'fold_vend': int(vend),
#                     'n_preds_val': n_preds,
#                     'tp_val': tp,
#                     'fp_val': fp,
#                     'val_precision': float(prc),
#                     'val_accuracy': float(acc),
#                 })
#
#         # pooled diagnostics
#         if val_prob_all:
#             vp = np.concatenate(val_prob_all, axis=0)
#             vt = np.concatenate(val_true_all, axis=0)
#             try: val_auc = float(roc_auc_score(vt, vp))
#             except Exception: val_auc = np.nan
#             try: val_ll  = float(log_loss(vt, vp, labels=[0, 1]))
#             except Exception: val_ll  = np.nan
#             try: val_bri = float(brier_score_loss(vt, vp))
#             except Exception: val_bri = np.nan
#         else:
#             val_auc = val_ll = val_bri = np.nan
#
#         for r in rows:
#             r['val_auc'] = val_auc
#             r['val_logloss'] = val_ll
#             r['val_brier'] = val_bri
#
#         return rows
#
#     # ------------------ Parallel parameter search ------------------
#     total_tasks = len(all_param_dicts)
#     if base_model == "mlp":
#         eff_jobs = min(max(1, cpu_jobs), 4)
#         prefer = "threads"; backend = "threading"; pre_dispatch = eff_jobs
#         ctx = parallel_backend(backend, n_jobs=eff_jobs)
#     else:
#         eff_jobs = max(1, min(cpu_jobs, 4)) if cpu_jobs != -1 else 4
#         prefer = "processes"; backend = "loky"; pre_dispatch = f"{2*eff_jobs}"
#         ctx = parallel_backend(backend, n_jobs=eff_jobs, inner_max_num_threads=1)
#
#     with ctx:
#         try:
#             with tqdm_joblib(tqdm(total=total_tasks, desc=f"Param search ({search_mode}, {base_model})")) as _:
#                 out = Parallel(
#                     n_jobs=eff_jobs, batch_size=1, prefer=prefer, pre_dispatch=pre_dispatch
#                 )(
#                     delayed(evaluate_param_set)(pd_, i, total_tasks)
#                     for i, pd_ in enumerate(all_param_dicts)
#                 )
#         except OSError as e:
#             print(f"[WARN] Parallel failed with {e}. Falling back to serial search...")
#             out = []
#             for i, pd_ in enumerate(tqdm(all_param_dicts, desc=f"Param search (serial, {base_model})")):
#                 out.append(evaluate_param_set(pd_, i, total_tasks))
#
#     val_rows = [r for sub in out for r in sub]
#     if not val_rows:
#         raise RuntimeError("No validation rows produced (check folds and input data).")
#     val_df = pd.DataFrame(val_rows)
#
#     # ------------------ Aggregate across folds (VALIDATION) ------------------
#     if base_model == "xgb":
#         param_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']
#     else:
#         param_keys = ['hidden_layer_sizes','alpha','learning_rate_init','batch_size','max_iter']
#
#     group_cols = param_keys + ['threshold']
#     agg = val_df.groupby(group_cols, as_index=False).agg({
#         'n_preds_val': 'sum',
#         'tp_val': 'sum',
#         'fp_val': 'sum',
#         'val_precision': 'mean',
#         'val_accuracy': 'mean',
#         'val_auc': 'mean',
#         'val_logloss': 'mean',
#         'val_brier': 'mean',
#     })
#
#     # Wilson-LCB & pooled precision
#     agg['val_precision_pooled'] = agg.apply(
#         lambda r: (r['tp_val'] / max(1, (r['tp_val'] + r['fp_val']))), axis=1
#     )
#     agg['val_precision_lcb'] = agg.apply(
#         lambda r: _wilson_lcb(int(r['tp_val']), int(r['fp_val']), conf=val_conf_level), axis=1
#     )
#
#     # Validation gates
#     qual = agg[
#         (agg['val_precision'] >= float(precision_test_threshold)) &
#         (agg['n_preds_val'] >= int(min_samples))
#     ].copy()
#
#     if qual.empty:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         fail_csv = None
#         if save_diagnostics_on_fail:
#             diag = (agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                                     ascending=[False, False, False, False])
#                     .assign(fail_reason="failed_validation_gate", market=market))
#             fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv")
#             os.makedirs(csv_save_dir, exist_ok=True)
#             diag.to_csv(fail_csv, index=False)
#         msg = (f"No strategy met validation gates (precision ≥ {precision_test_threshold} "
#                f"and n_preds_val ≥ {min_samples}).")
#         if on_fail == "raise":
#             raise RuntimeError(msg)
#         if on_fail == "warn":
#             print("[WARN]", msg)
#         return {
#             'status': 'failed_validation_gate',
#             'csv': fail_csv,
#             'model_pkl': None,
#             'summary_df': None,
#             'validation_table': agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                                                 ascending=[False, False, False, False]).reset_index(drop=True)
#         }
#
#     ranked = qual.sort_values(
#         by=['val_precision_lcb', 'val_precision', 'n_preds_val', 'val_accuracy'],
#         ascending=[False, False, False, False]
#     ).reset_index(drop=True)
#
#     topk_val = ranked.head(top_k).reset_index(drop=True)
#
#     # ------------------ Evaluate ALL Top-K on TEST ------------------
#     candidates = []
#     for _, row in topk_val.iterrows():
#         candidates.append({
#             'params': {k: row[k] for k in param_keys if k in row.index},
#             'threshold': float(row['threshold']),
#             'val_precision': float(row['val_precision']),
#             'val_precision_lcb': float(row['val_precision_lcb']),
#             'val_accuracy': float(row['val_accuracy']),
#             'n_preds_val': int(row['n_preds_val']),
#         })
#
#     records_all = []
#     for cand in candidates:
#         best_params = cast_params(cand['params'])
#         pos = int(y_train_final.sum()); neg = len(y_train_final) - pos
#         spw_final = (neg / pos) if pos > 0 else 1.0
#
#         final_model = build_model(best_params, spw_final)
#         final_sample_weight = None
#         if base_model == "mlp":
#             w_pos = spw_final
#             final_sample_weight = np.where(y_train_final.values == 1, w_pos, 1.0).astype(np.float32)
#
#         fit_model(final_model, X_train_final, y_train_final, X_val_final, y_val_final,
#                   sample_weight=final_sample_weight)
#         final_calibrator = fit_calibrator(final_model, X_val_final, y_val_final)
#
#         y_test_proba = predict_proba_1(final_calibrator, X_test)
#         thr = cand['threshold']
#         y_pred = (y_test_proba >= thr).astype(np.uint8)
#         n_preds_test = int(y_pred.sum())
#         prc_test = precision_score(y_test, y_pred, zero_division=0)
#         acc_test = accuracy_score(y_test, y_pred)
#
#         enough = n_preds_test >= int(min_test_samples)
#         not_collapsed = prc_test >= max(float(precision_test_threshold),
#                                         float(cand['val_precision']) - float(max_precision_drop))
#         pass_gate = bool(enough and not_collapsed)
#         reason = ""
#         if not pass_gate:
#             if not enough and not not_collapsed:
#                 reason = "insufficient_test_preds_and_precision_collapse"
#             elif not enough:
#                 reason = "insufficient_test_preds"
#             else:
#                 reason = "precision_collapse"
#
#         records_all.append({
#             **cand['params'],
#             'threshold': thr,
#             'val_precision_lcb': cand['val_precision_lcb'],
#             'val_precision': cand['val_precision'],
#             'val_accuracy': cand['val_accuracy'],
#             'n_preds_val': cand['n_preds_val'],
#             'n_preds_test': n_preds_test,
#             'test_precision': float(prc_test),
#             'test_accuracy': float(acc_test),
#             'pass_test_gate': pass_gate,
#             'fail_reason': reason,
#             'model_obj': final_calibrator if pass_gate else None,
#         })
#
#     survivors_df = pd.DataFrame(records_all)
#     passers = survivors_df[survivors_df['pass_test_gate']].copy()
#
#     # ------------------ Persist outputs ------------------
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     tag = "xgb" if base_model == "xgb" else "mlp"
#
#     if passers.empty:
#         fail_csv = None
#         if save_diagnostics_on_fail:
#             diag = (survivors_df
#                     .drop(columns=['model_obj'])
#                     .sort_values(by=['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                                  ascending=[False, False, False, False])
#                     .assign(market=market))
#             fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv")
#             diag.to_csv(fail_csv, index=False)
#             summary_df = diag
#         else:
#             summary_df = survivors_df.drop(columns=['model_obj'])
#         msg = (f"All Top-{len(candidates)} failed the TEST gate "
#                f"(n_preds_test ≥ {int(min_test_samples)} and precision not collapsing).")
#         if on_fail == "raise":
#             raise RuntimeError(msg)
#         if on_fail == "warn":
#             print("[WARN]", msg)
#         return {
#             'status': 'failed_test_gate',
#             'csv': fail_csv,
#             'model_pkl': None,
#             'summary_df': summary_df,
#             'validation_table': ranked,
#         }
#
#     passers_sorted = passers.sort_values(
#         by=['val_precision_lcb', 'val_precision', 'test_precision', 'n_preds_test', 'val_accuracy'],
#         ascending=[False, False, False, False, False]
#     ).reset_index(drop=True)
#
#     # Save PKL
#     pkl_path = os.path.join(model_dir, f"best_model_{market}_{tag}_calibrated_{timestamp}.pkl")
#     # Prepare CSV (include PKL for top row)
#     csv_df = passers_sorted.drop(columns=['model_obj']).copy()
#     csv_df['model_pkl'] = ""
#     csv_df.loc[0, 'model_pkl'] = pkl_path
#     csv_df['market'] = market
#     csv_path = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}.csv")
#     csv_df.to_csv(csv_path, index=False)
#
#     # Save the model PKL for the top row
#     top_row = passers_sorted.iloc[0]
#     chosen_model = top_row['model_obj']
#     chosen_params = {k: top_row[k] for k in param_keys if k in passers_sorted.columns}
#     chosen_threshold = float(top_row['threshold'])
#
#     joblib.dump(
#         {
#             'model': chosen_model,
#             'threshold': chosen_threshold,
#             'features': features,
#             'base_model': base_model,
#             'best_params': chosen_params,
#             'precision_test_threshold': float(precision_test_threshold),
#             'min_samples': int(min_samples),
#             'min_test_samples': int(min_test_samples),
#             'val_conf_level': float(val_conf_level),
#             'max_precision_drop': float(max_precision_drop),
#             'market': market,
#             'notes': (
#                 'CSV includes only candidates passing test gate; ranked by '
#                 'val_precision_lcb → val_precision → test_precision → n_preds_test → val_accuracy. '
#                 'Seeds are random each run.'
#             ),
#             'run_seed': int(RUN_SEED),
#         },
#         pkl_path
#     )
#
#     return {
#         'status': 'ok',
#         'csv': csv_path,
#         'model_pkl': pkl_path,
#         'summary_df': csv_df,           # passers only, with model_pkl on row 0
#         'validation_table': ranked,     # full validation ranking (post-gates)
#     }

# def run_models_outcome(
#     matches_filtered: pd.DataFrame,
#     features: list,
#     # ── gates ──────────────────────────────────────────────────────────────
#     min_samples: int = 200,
#     min_test_samples: int = 100,
#     precision_test_threshold: float = 0.80,
#     # ── model/search ───────────────────────────────────────────────────────
#     base_model: str = "xgb",
#     search_mode: str = "random",
#     n_random_param_sets: int = 10,
#     cpu_jobs: int = 6,
#     top_k: int = 10,
#     thresholds: np.ndarray | None = None,        # USED only for CLASSIFY markets
#     out_dir: str | None = None,
#     # ── anti-overfitting ──────────────────────────────────────────────────
#     val_conf_level: float = 0.99,
#     max_precision_drop: float = 0.05,
#     # ── failure handling ───────────────────────────────────────────────────
#     on_fail: str = "return",                     # "return" | "warn" | "raise"
#     save_diagnostics_on_fail: bool = True,
#     # ── market ─────────────────────────────────────────────────────────────
#     market: str = "LAY_AWAY",                    # LAY_* | BACK_* | OVER | UNDER (or other classify markets)
#     # ── VALUE LAY controls ────────────────────────────────────────────────
#     use_value_for_lay: bool = True,
#     liability_test: float = 1.0,                 # liability per lay bet on TEST
#     min_val_bets_val_lay: int = 150,             # min lay value bets across val folds
#     value_edge_grid_lay: np.ndarray | None = None,  # grid of edges to SWEEP (e.g. 0.00→0.20)
#     # ── VALUE BACK controls ────────────────────────────────────────────────
#     use_value_for_back: bool = True,
#     back_stake_test: float = 1.0,                # stake per back bet on TEST
#     min_val_bets_val_back: int = 150,            # min back value bets across val folds
#     value_edge_grid_back: np.ndarray | None = None, # grid of edges to SWEEP (e.g. 0.00→0.20)
#     # ── OUTPUTS: chosen model ─────────────────────────────────────────────
#     save_bets_csv: bool = False,
#     bets_csv_dir: str | None = None,
#     plot_pl: bool = False,
#     plot_dir: str | None = None,
#     plot_title_suffix: str = "",
#     # ── OUTPUTS: ALL candidates ───────────────────────────────────────────
#     save_all_bets_csv: bool = False,
#     all_bets_dir: str | None = None,
#     all_bets_include_failed: bool = True,        # include candidates that failed TEST gate
# ):
#     """
#     Rolling time-ordered CV with calibration.
#
#     VALUE LAY (market starts with LAY_):
#       fair_odds = 1 / P(selection wins). Place lay if fair ≥ market × (1+edge_param).
#       Stake to constant liability L: stake = L / (odds - 1).
#       P/L: +stake if selection loses (target==1), else −L.
#       Ranked by one-sided p-value vs break-even (then P/L).  Sweep: edge_param.
#
#     VALUE BACK (market starts with BACK_):
#       Place back if market ≥ fair × (1+edge_param). Fixed stake S.
#       P/L: +(odds−1)*S if selection wins (target==1), else −S.
#       Ranked by one-sided p-value vs break-even (then P/L).  Sweep: edge_param.
#
#     CLASSIFY (e.g., OVER/UNDER):
#       We SWEEP probability thresholds (argument `thresholds`) on validation and carry the
#       chosen threshold into test. Ranked by validation LCB → precision → etc.
#     """
#     # ---------------- setup ----------------
#     import os, secrets, hashlib, json
#     from datetime import datetime
#     import numpy as np
#     import pandas as pd
#     from itertools import product
#     from sklearn.model_selection import ParameterSampler
#     from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, log_loss, brier_score_loss
#     from sklearn.calibration import CalibratedClassifierCV
#     from sklearn.pipeline import Pipeline, make_pipeline
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.neural_network import MLPClassifier
#     from joblib import Parallel, delayed, parallel_backend
#     from tqdm import tqdm
#     from tqdm_joblib import tqdm_joblib
#     import joblib
#
#     # --- xgboost import (optional)
#     try:
#         import xgboost as xgb
#         _HAS_XGB_LOCAL = True
#     except Exception:
#         _HAS_XGB_LOCAL = False
#     _HAS_XGB = globals().get("_HAS_XGB", _HAS_XGB_LOCAL)
#     if base_model == "xgb" and not _HAS_XGB:
#         raise ImportError("XGBoost not available; set base_model='mlp' or install xgboost.")
#
#     # --- random dists
#     try:
#         _randint; _uniform; _loguniform
#     except NameError:
#         from scipy.stats import randint as _randint
#         from scipy.stats import uniform as _uniform
#         from scipy.stats import loguniform as _loguniform
#
#     # --- Wilson LCB & normal CDF
#     try:
#         from scipy.stats import norm
#         _Z = lambda conf: float(norm.ppf(1 - (1 - conf) / 2))
#         _Phi = lambda z: float(norm.cdf(z))
#     except Exception:
#         import math
#         _Z = lambda conf: 1.96 if abs(conf - 0.95) < 1e-6 else 1.64
#         _Phi = lambda z: 0.5 * (1.0 + math.erf(z / (2**0.5)))
#
#     def _wilson_lcb(tp: int, fp: int, conf: float) -> float:
#         n = tp + fp
#         if n <= 0: return 0.0
#         p = tp / n
#         z = _Z(conf)
#         denom = 1.0 + (z*z)/n
#         centre = p + (z*z)/(2*n)
#         rad = z * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
#         return max(0.0, (centre - rad) / denom)
#
#     # defaults
#     if thresholds is None:
#         thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)  # CLASSIFY only
#     if value_edge_grid_lay is None:
#         value_edge_grid_lay = np.round(np.arange(0.00, 0.201, 0.01), 2)
#     if value_edge_grid_back is None:
#         value_edge_grid_back = np.round(np.arange(0.00, 0.201, 0.01), 2)
#
#     # --- paths
#     BASE = r"C:\Users\leere\PycharmProjects\Football_ML3\FT Results"
#     PKL_DIRS = {
#         "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "model_file"),
#         "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "model_file"),
#         "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "model_file"),
#         "BACK_HOME": os.path.join(BASE, "Back_Home", "model_file"),
#         "BACK_AWAY": os.path.join(BASE, "Back_Away", "model_file"),
#         "BACK_DRAW": os.path.join(BASE, "Back_Draw", "model_file"),
#         "OVER":      os.path.join(BASE, "Over_2_5",  "model_file"),
#         "UNDER":     os.path.join(BASE, "Under_2_5", "model_file"),
#     }
#     CSV_DIRS = {
#         "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "best_model_metrics"),
#         "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "best_model_metrics"),
#         "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "best_model_metrics"),
#         "BACK_HOME": os.path.join(BASE, "Back_Home", "best_model_metrics"),
#         "BACK_AWAY": os.path.join(BASE, "Back_Away", "best_model_metrics"),
#         "BACK_DRAW": os.path.join(BASE, "Back_Draw", "best_model_metrics"),
#         "OVER":      os.path.join(BASE, "Over_2_5",  "best_model_metrics"),
#         "UNDER":     os.path.join(BASE, "Under_2_5", "best_model_metrics"),
#     }
#
#     market = str(market).upper().strip()
#     if market not in PKL_DIRS: raise ValueError(f"Unsupported market '{market}'.")
#     _IS_LAY  = market.startswith("LAY_")
#     _IS_BACK = market.startswith("BACK_")
#     _USE_VALUE_LAY  = bool(use_value_for_lay and _IS_LAY)
#     _USE_VALUE_BACK = bool(use_value_for_back and _IS_BACK)
#     _USE_VALUE = _USE_VALUE_LAY or _USE_VALUE_BACK
#     _IS_CLASSIFY = not _USE_VALUE  # thresholds swept only in CLASSIFY
#
#     csv_save_dir = out_dir if (out_dir and len(str(out_dir)) > 0) else CSV_DIRS[market]
#     os.makedirs(csv_save_dir, exist_ok=True)
#     model_dir = PKL_DIRS[market]; os.makedirs(model_dir, exist_ok=True)
#     if bets_csv_dir is None: bets_csv_dir = csv_save_dir
#     if plot_dir is None: plot_dir = csv_save_dir
#     os.makedirs(bets_csv_dir, exist_ok=True)
#     os.makedirs(plot_dir, exist_ok=True)
#     if all_bets_dir is None:
#         all_bets_dir = os.path.join(os.path.dirname(CSV_DIRS[market]), "all_bets")
#     os.makedirs(all_bets_dir, exist_ok=True)
#
#     RUN_SEED = secrets.randbits(32)
#     def _seed_from(*vals) -> int:
#         h = hashlib.blake2b(digest_size=8)
#         h.update(int(RUN_SEED).to_bytes(8,'little',signed=False))
#         for v in vals: h.update(str(v).encode('utf-8'))
#         return int.from_bytes(h.digest(),'little') & 0x7FFFFFFF
#
#     def _as_float(x):
#         try: return float(x)
#         except Exception: return float(str(x))
#     def _as_int(x): return int(float(x))
#
#     # ---------------- data ----------------
#     req_cols = {'date','target'}
#     if _USE_VALUE: req_cols |= {'home_odds','draw_odds','away_odds'}
#     missing = req_cols - set(matches_filtered.columns)
#     if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
#
#     df = matches_filtered.copy()
#     df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     df = df.sort_values('date').reset_index(drop=True)
#
#     cols_needed = list(set(features) | {'target'} | ({'home_odds','draw_odds','away_odds'} if _USE_VALUE else set()))
#     df = df.dropna(subset=cols_needed).reset_index(drop=True)
#
#     X = df[features].copy()
#     y = df['target'].astype(int).reset_index(drop=True)
#
#     n = len(X)
#     if n < max(min_samples * 3, 500): raise RuntimeError(f"Not enough rows: {n}")
#
#     # temporal split
#     test_start = int(0.85 * n)
#     pretest_end = test_start
#     X_test = X.iloc[test_start:].reset_index(drop=True)
#     y_test = y.iloc[test_start:].reset_index(drop=True)
#     df_test = df.iloc[test_start:].reset_index(drop=True)
#
#     # rolling validation folds
#     N_FOLDS = 5
#     total_val_len = max(1, int(0.15 * n))
#     val_len = max(1, total_val_len // N_FOLDS)
#     fold_val_ends = [pretest_end - total_val_len + (i + 1) * val_len for i in range(N_FOLDS)]
#     fold_val_starts = [end - val_len for end in fold_val_ends]
#     if fold_val_ends:
#         fold_val_ends[-1] = min(fold_val_ends[-1], pretest_end)
#         fold_val_starts[-1] = max(0, fold_val_ends[-1] - val_len)
#
#     # final small validation slice (for calibration before test)
#     final_val_len = max(1, val_len)
#     final_val_start = max(0, test_start - final_val_len)
#     X_train_final = X.iloc[:final_val_start]
#     y_train_final = y.iloc[:final_val_start]
#     X_val_final   = X.iloc[final_val_start:test_start]
#     y_val_final   = y.iloc[final_val_start:test_start]
#
#     # ---------------- param spaces ----------------
#     xgb_param_grid = {'n_estimators':[200],'max_depth':[5],'learning_rate':[0.1],'subsample':[0.7],
#                       'colsample_bytree':[1.0],'min_child_weight':[5],'reg_lambda':[1.0]}
#     xgb_param_distributions = {'n_estimators':_randint(100,1001),'max_depth':_randint(3,8),
#                                'learning_rate':_loguniform(0.01,0.2),'min_child_weight':_randint(3,13),
#                                'subsample':_uniform(0.7,0.3),'colsample_bytree':_uniform(0.6,0.4),
#                                'reg_lambda':_loguniform(0.1,10.0)}
#     mlp_param_grid = {'hidden_layer_sizes':[(128,),(256,),(128,64)],'alpha':[1e-4],
#                       'learning_rate_init':[1e-3],'batch_size':['auto'],'max_iter':[200]}
#     mlp_param_distributions = {'hidden_layer_sizes':[(64,),(128,),(256,),(128,64),(256,128)],
#                                'alpha':_loguniform(1e-5,1e-2),'learning_rate_init':_loguniform(5e-4,5e-2),
#                                'batch_size':_randint(32,257),'max_iter':_randint(150,401)}
#
#     def cast_params(p: dict) -> dict:
#         q = dict(p)
#         if base_model == "xgb":
#             for k in ['n_estimators','max_depth','min_child_weight']:
#                 if k in q: q[k] = _as_int(q[k])
#             for k in ['learning_rate','subsample','colsample_bytree','reg_lambda']:
#                 if k in q: q[k] = _as_float(q[k])
#         else:
#             if 'max_iter' in q: q['max_iter'] = _as_int(q['max_iter'])
#             if 'batch_size' in q and q['batch_size'] != 'auto': q['batch_size'] = _as_int(q['batch_size'])
#             if 'alpha' in q: q['alpha'] = _as_float(q['alpha'])
#             if 'learning_rate_init' in q: q['learning_rate_init'] = _as_float(q['learning_rate_init'])
#             if 'hidden_layer_sizes' in q:
#                 h = q['hidden_layer_sizes']
#                 if isinstance(h, str):
#                     parts = [pp.strip() for pp in h.strip("()").split(",") if pp.strip()!='']
#                     q['hidden_layer_sizes'] = tuple(_as_int(pp) for pp in parts) if parts else (128,)
#                 elif isinstance(h, (list, tuple, np.ndarray)):
#                     q['hidden_layer_sizes'] = tuple(int(v) for v in h)
#                 else:
#                     q['hidden_layer_sizes'] = (int(h),)
#         return q
#
#     def _final_step_name(estimator):
#         try:
#             if isinstance(estimator, Pipeline): return estimator.steps[-1][0]
#         except Exception:
#             pass
#         return None
#
#     def build_model(params: dict, spw: float):
#         model_seed = _seed_from("model", base_model, tuple(sorted(params.items())))
#         if base_model == "xgb":
#             return xgb.XGBClassifier(
#                 objective='binary:logistic',
#                 eval_metric='auc',
#                 random_state=model_seed,
#                 scale_pos_weight=spw,
#                 n_jobs=1,
#                 tree_method="hist",
#                 verbosity=0,
#                 **params
#             )
#         else:
#             mlp = MLPClassifier(
#                 random_state=model_seed,
#                 early_stopping=True,
#                 n_iter_no_change=20,
#                 validation_fraction=0.1,
#                 solver="adam",
#                 **params
#             )
#             return make_pipeline(StandardScaler(with_mean=True, with_std=True), mlp)
#
#     def fit_model(model, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None):
#         if base_model == "xgb":
#             try:
#                 model.set_params(verbosity=0, early_stopping_rounds=50)
#                 if X_va is not None and y_va is not None:
#                     model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
#                 else:
#                     model.fit(X_tr, y_tr, verbose=False)
#             except Exception:
#                 model.fit(X_tr, y_tr, verbose=False)
#         else:
#             fit_kwargs = {}
#             if sample_weight is not None:
#                 stepname = _final_step_name(model)
#                 if stepname is not None:
#                     fit_kwargs[f"{stepname}__sample_weight"] = sample_weight
#             try:
#                 model.fit(X_tr, y_tr, **fit_kwargs)
#             except TypeError:
#                 model.fit(X_tr, y_tr)
#
#     def fit_calibrator(fitted, X_va, y_va):
#         try:
#             from sklearn.calibration import FrozenEstimator
#             frozen = FrozenEstimator(fitted)
#             cal = CalibratedClassifierCV(frozen, method='sigmoid', cv=None)
#             cal.fit(X_va, y_va)
#             return cal
#         except Exception:
#             try:
#                 cal = CalibratedClassifierCV(fitted, method='sigmoid', cv='prefit')
#                 cal.fit(X_va, y_va); return cal
#             except Exception:
#                 return fitted
#
#     def _unwrap_estimator(est):
#         if isinstance(est, Pipeline): return est.steps[-1][1]
#         return est
#
#     def predict_proba_pos(model_or_cal, X_):
#         proba = model_or_cal.predict_proba(X_)
#         if proba.ndim == 2:
#             classes = getattr(model_or_cal, "classes_", None)
#             if classes is None:
#                 base = _unwrap_estimator(model_or_cal); classes = getattr(base, "classes_", None)
#             if classes is not None and len(classes) == proba.shape[1]:
#                 try:
#                     idx = int(np.where(np.asarray(classes) == 1)[0][0])
#                     return proba[:, idx].astype(np.float32)
#                 except Exception:
#                     pass
#             if proba.shape[1] == 2: return proba[:, 1].astype(np.float32)
#             if proba.shape[1] == 1:
#                 only = getattr(model_or_cal, "classes_", [0])[0]
#                 return (np.ones_like(proba[:,0]) if only==1 else np.zeros_like(proba[:,0])).astype(np.float32)
#         return np.asarray(proba, dtype=np.float32)
#
#     # --- p-value helper (VALUE modes)
#     def _pvalue_break_even(bets_df: pd.DataFrame, mode: str) -> dict:
#         if not isinstance(bets_df, pd.DataFrame) or bets_df.empty:
#             return {'z': 0.0, 'p_value': 1.0, 'var_sum': 0.0, 'n': 0, 'total_pl': 0.0}
#         o = np.asarray(bets_df['market_odds'].values, dtype=float)
#         o = np.where(o <= 1.0, np.nan, o)
#         p = 1.0 / o  # null probability selection wins
#         pl = np.asarray(bets_df['pl'].values, dtype=float)
#         if mode == 'VALUE_BACK':
#             S = np.asarray(bets_df['stake'].values, dtype=float)
#             var_i = (p * ((o - 1.0) * S) ** 2) + ((1.0 - p) * (S ** 2))
#         else:  # VALUE_LAY
#             L = np.asarray(bets_df['liability'].values, dtype=float)
#             stake = np.asarray(bets_df['stake'].values, dtype=float)  # L/(o-1)
#             var_i = (p * (L ** 2)) + ((1.0 - p) * (stake ** 2))
#         var_i = np.where(np.isfinite(var_i), var_i, 0.0)
#         var_sum = float(np.nansum(var_i))
#         total_pl = float(np.nansum(np.where(np.isfinite(pl), pl, 0.0)))
#         z = total_pl / (np.sqrt(var_sum) + 1e-12)
#         p_val = max(0.0, 1.0 - _Phi(z))  # one-sided
#         return {'z': float(z), 'p_value': float(p_val), 'var_sum': var_sum, 'n': int(len(pl)), 'total_pl': total_pl}
#
#     # ---------------- search space ----------------
#     if search_mode.lower() == "grid":
#         grid = xgb_param_grid if base_model == "xgb" else mlp_param_grid
#         all_param_dicts = [dict(zip(grid.keys(), combo)) for combo in product(*grid.values())]
#     else:
#         dists = xgb_param_distributions if base_model == "xgb" else mlp_param_distributions
#         sampler_seed = _seed_from("sampler")
#         all_param_dicts = list(ParameterSampler(dists, n_iter=n_random_param_sets, random_state=sampler_seed))
#
#     # ---------------- validation eval ----------------
#     def evaluate_param_set(param_dict, *_):
#         safe = cast_params(param_dict)
#         rows = []; val_prob_all=[]; val_true_all=[]
#
#         for vstart, vend in zip(fold_val_starts, fold_val_ends):
#             if vstart is None or vend is None or vstart <= 0 or vend <= vstart: continue
#
#             X_tr, y_tr = X.iloc[:vstart], y.iloc[:vstart]
#             X_va, y_va = X.iloc[vstart:vend], y.iloc[vstart:vend]
#             df_va = df.iloc[vstart:vend]
#             if y_tr.nunique() < 2: continue
#
#             pos = int(y_tr.sum()); neg = len(y_tr) - pos
#             spw = (neg/pos) if pos > 0 else 1.0
#
#             sample_weight = None
#             if base_model == "mlp":
#                 w_pos = spw
#                 sample_weight = np.where(y_tr.values==1, w_pos, 1.0).astype(np.float32)
#
#             model = build_model(safe, spw)
#             fit_model(model, X_tr, y_tr, X_va, y_va, sample_weight=sample_weight)
#             cal = fit_calibrator(model, X_va, y_va)
#
#             p_pos = predict_proba_pos(cal, X_va)
#             val_prob_all.append(p_pos)
#             y_true = y_va.values.astype(np.uint8); val_true_all.append(y_true)
#
#             if _IS_CLASSIFY:
#                 # Sweep probability thresholds for classification markets
#                 for thr in thresholds:
#                     thr = float(thr)
#                     y_pred = (p_pos >= thr).astype(np.uint8)
#                     n_preds = int(y_pred.sum())
#                     tp = int(((y_true == 1) & (y_pred == 1)).sum())
#                     fp = int(((y_true == 0) & (y_pred == 1)).sum())
#                     prc = precision_score(y_va, y_pred, zero_division=0)
#                     acc = accuracy_score(y_va, y_pred)
#                     rows.append({
#                         **safe,
#                         'threshold': thr,
#                         'edge_param': np.nan,
#                         'fold_vstart': int(vstart),
#                         'fold_vend': int(vend),
#                         'n_preds_val': n_preds,
#                         'tp_val': tp,
#                         'fp_val': fp,
#                         'val_precision': float(prc),
#                         'val_accuracy': float(acc),
#                         'val_edge_ratio_mean': np.nan,
#                         'val_edge_ratio_mean_back': np.nan,
#                         'n_value_bets_val': 0,
#                     })
#             else:
#                 # VALUE modes: sweep value edges (thresholds not used at all)
#                 edge_grid = value_edge_grid_lay if _USE_VALUE_LAY else value_edge_grid_back
#                 for edge_param in edge_grid:
#                     # diagnostics for precision/accuracy reporting (not used for selection)
#                     y_pred = (p_pos >= 0.5).astype(np.uint8)
#                     n_preds = int(y_pred.sum())
#                     tp = int(((y_true==1) & (y_pred==1)).sum())
#                     fp = int(((y_true==0) & (y_pred==1)).sum())
#                     prc = precision_score(y_va, y_pred, zero_division=0)
#                     acc = accuracy_score(y_va, y_pred)
#
#                     r = {
#                         **safe,
#                         'threshold': np.nan,                  # no threshold in VALUE mode
#                         'edge_param': float(edge_param),      # the swept knob
#                         'fold_vstart': int(vstart),
#                         'fold_vend': int(vend),
#                         'n_preds_val': n_preds,
#                         'tp_val': tp,
#                         'fp_val': fp,
#                         'val_precision': float(prc),
#                         'val_accuracy': float(acc),
#                         'val_edge_ratio_mean': np.nan,
#                         'val_edge_ratio_mean_back': np.nan,
#                         'n_value_bets_val': 0,
#                     }
#
#                     # count value opportunities under this edge_param
#                     if _USE_VALUE_LAY:
#                         if market == "LAY_AWAY":
#                             p_sel_win = 1.0 - p_pos; mkt = df_va['away_odds'].values
#                         elif market == "LAY_HOME":
#                             p_sel_win = 1.0 - p_pos; mkt = df_va['home_odds'].values
#                         else:  # LAY_DRAW
#                             p_sel_win = 1.0 - p_pos; mkt = df_va['draw_odds'].values
#                         fair = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
#                         edge_mask = (fair >= (1.0 + float(edge_param)) * mkt) & np.isfinite(mkt)
#                         with np.errstate(divide='ignore', invalid='ignore'):
#                             edge_ratio = fair / mkt
#                             r['val_edge_ratio_mean'] = float(np.nanmean(np.where(np.isfinite(edge_ratio), edge_ratio, np.nan)))
#                         r['n_value_bets_val'] = int(edge_mask.sum())
#                     else:
#                         if market == "BACK_AWAY":
#                             p_sel_win = p_pos; mkt = df_va['away_odds'].values
#                         elif market == "BACK_HOME":
#                             p_sel_win = p_pos; mkt = df_va['home_odds'].values
#                         else:  # BACK_DRAW
#                             p_sel_win = p_pos; mkt = df_va['draw_odds'].values
#                         fair = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
#                         edge_mask = (mkt >= (1.0 + float(edge_param)) * fair) & np.isfinite(mkt)
#                         with np.errstate(divide='ignore', invalid='ignore'):
#                             edge_ratio = mkt / fair
#                             r['val_edge_ratio_mean_back'] = float(np.nanmean(np.where(np.isfinite(edge_ratio), edge_ratio, np.nan)))
#                         r['n_value_bets_val'] = int(edge_mask.sum())
#
#                     rows.append(r)
#
#         # pooled diagnostics
#         if val_prob_all:
#             vp = np.concatenate(val_prob_all, axis=0)
#             vt = np.concatenate(val_true_all, axis=0)
#             try: val_auc = float(roc_auc_score(vt, vp))
#             except Exception: val_auc = np.nan
#             try: val_ll  = float(log_loss(vt, vp, labels=[0, 1]))
#             except Exception: val_ll = np.nan
#             try: val_bri = float(brier_score_loss(vt, vp))
#             except Exception: val_bri = np.nan
#         else:
#             val_auc = val_ll = val_bri = np.nan
#
#         for r in rows:
#             r['val_auc'] = val_auc
#             r['val_logloss'] = val_ll
#             r['val_brier'] = val_bri
#
#         return rows
#
#     # ---------------- search ----------------
#     if base_model == "mlp":
#         eff_jobs = min(max(1, cpu_jobs), 4); prefer = "threads"; backend = "threading"; pre_dispatch = eff_jobs
#         ctx = parallel_backend(backend, n_jobs=eff_jobs)
#     else:
#         eff_jobs = max(1, min(cpu_jobs, 4)) if cpu_jobs != -1 else 4
#         prefer = "processes"; backend = "loky"; pre_dispatch = f"{2*eff_jobs}"
#         ctx = parallel_backend(backend, n_jobs=eff_jobs, inner_max_num_threads=1)
#
#     with ctx:
#         try:
#             with tqdm_joblib(tqdm(total=len(all_param_dicts), desc=f"Param search ({search_mode}, {base_model})")) as _:
#                 out = Parallel(n_jobs=eff_jobs, batch_size=1, prefer=prefer, pre_dispatch=pre_dispatch)(
#                     delayed(evaluate_param_set)(pd_) for pd_ in all_param_dicts
#                 )
#         except OSError as e:
#             print(f"[WARN] Parallel failed with {e}. Falling back to serial search...")
#             out = []
#             for pd_ in tqdm(all_param_dicts, desc=f"Param search (serial, {base_model})"):
#                 out.append(evaluate_param_set(pd_))
#
#     val_rows = [r for sub in out for r in sub]
#     if not val_rows: raise RuntimeError("No validation rows produced (check folds and input data).")
#     val_df = pd.DataFrame(val_rows)
#
#     # ---------------- validation aggregate ----------------
#     if base_model == "xgb":
#         param_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']
#     else:
#         param_keys = ['hidden_layer_sizes','alpha','learning_rate_init','batch_size','max_iter']
#
#     # group by params + threshold (CLASSIFY) OR params + edge_param (VALUE)
#     group_cols = param_keys + (['threshold'] if _IS_CLASSIFY else ['edge_param'])
#
#     agg_dict = {
#         'n_preds_val': 'sum',
#         'tp_val': 'sum',
#         'fp_val': 'sum',
#         'val_precision': 'mean',
#         'val_accuracy': 'mean',
#         'val_auc': 'mean',
#         'val_logloss': 'mean',
#         'val_brier': 'mean',
#         'n_value_bets_val': 'sum',
#     }
#     if 'val_edge_ratio_mean' in val_df.columns: agg_dict['val_edge_ratio_mean'] = 'mean'
#     if 'val_edge_ratio_mean_back' in val_df.columns: agg_dict['val_edge_ratio_mean_back'] = 'mean'
#
#     agg = val_df.groupby(group_cols, as_index=False).agg(agg_dict)
#     agg['val_precision_pooled'] = agg.apply(lambda r: (r['tp_val'] / max(1, (r['tp_val'] + r['fp_val']))), axis=1)
#     agg['val_precision_lcb'] = agg.apply(lambda r: _wilson_lcb(int(r['tp_val']), int(r['fp_val']), conf=val_conf_level), axis=1)
#
#     qual_mask = (
#         (agg['val_precision'] >= float(precision_test_threshold)) &
#         (agg['n_preds_val'] >= int(min_samples))
#     )
#     if _USE_VALUE_LAY:  qual_mask &= (agg['n_value_bets_val'] >= int(min_val_bets_val_lay))
#     if _USE_VALUE_BACK: qual_mask &= (agg['n_value_bets_val'] >= int(min_val_bets_val_back))
#     qual = agg[qual_mask].copy()
#
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     if qual.empty:
#         fail_csv = None
#         if save_diagnostics_on_fail:
#             diag = (agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                                     ascending=[False, False, False, False])
#                     .assign(fail_reason="failed_validation_gate", market=market))
#             fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv")
#             os.makedirs(csv_save_dir, exist_ok=True); diag.to_csv(fail_csv, index=False)
#         msg = "No strategy met validation gates."
#         if on_fail == "raise": raise RuntimeError(msg)
#         if on_fail == "warn": print("[WARN]", msg)
#         return {'status':'failed_validation_gate','csv':fail_csv,'model_pkl':None,
#                 'summary_df':None,'validation_table':agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                 ascending=[False,False,False,False]).reset_index(drop=True)}
#
#     ranked = qual.sort_values(by=['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                               ascending=[False, False, False, False]).reset_index(drop=True)
#     topk_val = ranked.head(top_k).reset_index(drop=True)
#
#     def _extract_params_from_row(row):
#         return cast_params({k: row[k] for k in param_keys if k in row.index})
#
#     candidates = []
#     for _, row in topk_val.iterrows():
#         c = {
#             'params': _extract_params_from_row(row),
#             'val_precision': float(row['val_precision']),
#             'val_precision_lcb': float(row['val_precision_lcb']),
#             'val_accuracy': float(row['val_accuracy']),
#             'n_preds_val': int(row['n_preds_val']),
#         }
#         if _IS_CLASSIFY:
#             c['threshold'] = float(row['threshold'])
#         else:
#             c['edge_param'] = float(row['edge_param'])
#         candidates.append(c)
#
#     # ---------------- test eval ----------------
#     records_all = []
#     all_bets_collector = []  # across all candidates
#
#     def _name_cols(subdf):
#         cols = {}
#         for c in ['date','league','country','home_team','away_team','match_id']:
#             if c in subdf.columns: cols[c] = subdf[c].values
#         if {'home_team','away_team'}.issubset(subdf.columns):
#             cols['event_name'] = (subdf['home_team'] + ' v ' + subdf['away_team']).values
#         return cols
#
#     for cand_id, cand in enumerate(candidates):
#         best_params = cast_params(cand['params'])
#         pos = int(y_train_final.sum()); neg = len(y_train_final) - pos
#         spw_final = (neg/pos) if pos > 0 else 1.0
#
#         final_model = build_model(best_params, spw_final)
#         final_sample_weight = None
#         if base_model == "mlp":
#             w_pos = spw_final
#             final_sample_weight = np.where(y_train_final.values==1, w_pos, 1.0).astype(np.float32)
#
#         fit_model(final_model, X_train_final, y_train_final, X_val_final, y_val_final, sample_weight=final_sample_weight)
#         final_calibrator = fit_calibrator(final_model, X_val_final, y_val_final)
#         p_pos_test = predict_proba_pos(final_calibrator, X_test)
#
#         if _USE_VALUE_LAY:
#             if market == "LAY_AWAY":
#                 p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['away_odds'].values; sel_name = 'AWAY'
#             elif market == "LAY_HOME":
#                 p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['home_odds'].values; sel_name = 'HOME'
#             elif market == "LAY_DRAW":
#                 p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['draw_odds'].values; sel_name = 'DRAW'
#             else:
#                 raise RuntimeError("Internal: LAY mode only for LAY_*")
#             fair_odds = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
#             valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01)
#             edge = float(cand.get('edge_param', 0.0))                  # edge chosen in validation
#             edge_mask = valid & (fair_odds >= (1.0 + edge) * mkt_odds)
#             n_bets = int(edge_mask.sum())
#             L = float(liability_test)
#             stake = np.zeros_like(mkt_odds, dtype=np.float64)
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 stake[edge_mask] = L / (mkt_odds[edge_mask] - 1.0)
#             sel_wins = (y_test.values == 0)  # selection wins → away/home/draw actually wins → lay loses
#             pl = np.zeros_like(stake)
#             pl[edge_mask & (~sel_wins)] = stake[edge_mask & (~sel_wins)]
#             pl[edge_mask & (sel_wins)]  = -L
#             total_pl = float(pl.sum()); avg_pl = float(total_pl / max(1, n_bets))
#             lays_as_preds = np.zeros_like(edge_mask, dtype=np.uint8); lays_as_preds[edge_mask] = 1
#             prc_test = precision_score(y_test, lays_as_preds, zero_division=0)
#             acc_test = accuracy_score(y_test, lays_as_preds)
#
#             bet_idx = np.where(edge_mask)[0]
#             name_cols = _name_cols(df_test.iloc[bet_idx])
#             bets_df = pd.DataFrame({
#                 **name_cols,
#                 'selection': sel_name,
#                 'market_odds': mkt_odds[bet_idx],
#                 'fair_odds': fair_odds[bet_idx],
#                 'edge_ratio': np.where(fair_odds[bet_idx] > 0, fair_odds[bet_idx] / mkt_odds[bet_idx], np.nan),
#                 'liability': L,
#                 'stake': stake[bet_idx],
#                 'selection_won': sel_wins[bet_idx].astype(int),
#                 'target': y_test.values[bet_idx],
#                 'pl': pl[bet_idx],
#             })
#             if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
#             bets_df['cum_pl'] = bets_df['pl'].cumsum()
#
#             pv = _pvalue_break_even(bets_df, mode='VALUE_LAY')
#
#             enough = n_bets >= int(min_test_samples)
#             not_collapsed = prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop))
#             pass_gate = bool(enough and not_collapsed)
#             reason = "" if pass_gate else ("insufficient_test_bets" if not enough else "precision_collapse")
#
#             if len(bets_df):
#                 meta = {
#                     'candidate_id': cand_id,
#                     'passed_test_gate': bool(pass_gate),
#                     'mode': 'VALUE_LAY',
#                     'market': market,
#                     'threshold': np.nan,                     # irrelevant in VALUE modes
#                     'edge_param': edge,                      # carry to outputs
#                     'val_precision': float(cand['val_precision']),
#                     'val_precision_lcb': float(cand['val_precision_lcb']),
#                     'n_value_bets_test': int(n_bets),
#                     'total_pl': float(total_pl),
#                     'avg_pl': float(avg_pl),
#                     'liability': float(L),
#                     'p_value': pv['p_value'],
#                     'zscore': pv['z'],
#                     'params_json': json.dumps(best_params, default=float),
#                 }
#                 bdf = bets_df.copy()
#                 for k, v in meta.items(): bdf[k] = v
#                 all_bets_collector.append(bdf)
#
#             rec = {
#                 **best_params, 'threshold': np.nan, 'edge_param': edge,
#                 'val_precision_lcb': cand['val_precision_lcb'], 'val_precision': cand['val_precision'],
#                 'val_accuracy': cand['val_accuracy'],
#                 'n_value_bets_test': n_bets, 'test_precision_bets': float(prc_test),
#                 'test_accuracy_bets': float(acc_test), 'total_pl': total_pl, 'avg_pl': avg_pl,
#                 'p_value': pv['p_value'], 'zscore': pv['z'],
#                 'pass_test_gate': pass_gate, 'fail_reason': reason, 'model_obj': final_calibrator if pass_gate else None,
#                 'mode': 'VALUE_LAY', 'liability': L, 'bets': bets_df if pass_gate else None,
#             }
#
#         elif _USE_VALUE_BACK:
#             if market == "BACK_AWAY":
#                 p_sel_win = p_pos_test; mkt_odds = df_test['away_odds'].values; sel_name = 'AWAY'
#             elif market == "BACK_HOME":
#                 p_sel_win = p_pos_test; mkt_odds = df_test['home_odds'].values; sel_name = 'HOME'
#             elif market == "BACK_DRAW":
#                 p_sel_win = p_pos_test; mkt_odds = df_test['draw_odds'].values; sel_name = 'DRAW'
#             else:
#                 raise RuntimeError("Internal: BACK mode only for BACK_*")
#             fair_odds = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
#             valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01)
#             edge = float(cand.get('edge_param', 0.0))                  # edge chosen in validation
#             edge_mask = valid & (mkt_odds >= (1.0 + edge) * fair_odds)
#             n_bets = int(edge_mask.sum())
#             S = float(back_stake_test)
#             stake = np.zeros_like(mkt_odds, dtype=np.float64); stake[edge_mask] = S
#             sel_wins = (y_test.values == 1)
#             pl = np.zeros_like(stake)
#             win_idx = edge_mask & sel_wins
#             pl[win_idx] = (mkt_odds[win_idx] - 1.0) * S
#             lose_idx = edge_mask & (~sel_wins)
#             pl[lose_idx] = -S
#             total_pl = float(pl.sum()); avg_pl = float(total_pl / max(1, n_bets))
#             backs_as_preds = np.zeros_like(edge_mask, dtype=np.uint8); backs_as_preds[edge_mask] = 1
#             prc_test = precision_score(y_test, backs_as_preds, zero_division=0)
#             acc_test = accuracy_score(y_test, backs_as_preds)
#
#             bet_idx = np.where(edge_mask)[0]
#             name_cols = _name_cols(df_test.iloc[bet_idx])
#             bets_df = pd.DataFrame({
#                 **name_cols,
#                 'selection': sel_name,
#                 'market_odds': mkt_odds[bet_idx],
#                 'fair_odds': fair_odds[bet_idx],
#                 'edge_ratio': np.where(fair_odds[bet_idx] > 0, mkt_odds[bet_idx] / fair_odds[bet_idx], np.nan),
#                 'stake': S,
#                 'selection_won': sel_wins[bet_idx].astype(int),
#                 'target': y_test.values[bet_idx],
#                 'pl': pl[bet_idx],
#             })
#             if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
#             bets_df['cum_pl'] = bets_df['pl'].cumsum()
#
#             pv = _pvalue_break_even(bets_df, mode='VALUE_BACK')
#
#             enough = n_bets >= int(min_test_samples)
#             not_collapsed = prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop))
#             pass_gate = bool(enough and not_collapsed)
#             reason = "" if pass_gate else ("insufficient_test_bets" if not enough else "precision_collapse")
#
#             if len(bets_df):
#                 meta = {
#                     'candidate_id': cand_id,
#                     'passed_test_gate': bool(pass_gate),
#                     'mode': 'VALUE_BACK',
#                     'market': market,
#                     'threshold': np.nan,
#                     'edge_param': edge,
#                     'val_precision': float(cand['val_precision']),
#                     'val_precision_lcb': float(cand['val_precision_lcb']),
#                     'n_value_bets_test': int(n_bets),
#                     'total_pl': float(total_pl),
#                     'avg_pl': float(avg_pl),
#                     'stake': float(S),
#                     'p_value': pv['p_value'],
#                     'zscore': pv['z'],
#                     'params_json': json.dumps(best_params, default=float),
#                 }
#                 bdf = bets_df.copy()
#                 for k, v in meta.items(): bdf[k] = v
#                 all_bets_collector.append(bdf)
#
#             rec = {
#                 **best_params, 'threshold': np.nan, 'edge_param': edge,
#                 'val_precision_lcb': cand['val_precision_lcb'], 'val_precision': cand['val_precision'],
#                 'val_accuracy': cand['val_accuracy'],
#                 'n_value_bets_test': n_bets, 'test_precision_bets': float(prc_test),
#                 'test_accuracy_bets': float(acc_test), 'total_pl': total_pl, 'avg_pl': avg_pl,
#                 'p_value': pv['p_value'], 'zscore': pv['z'],
#                 'pass_test_gate': pass_gate, 'fail_reason': reason, 'model_obj': final_calibrator if pass_gate else None,
#                 'mode': 'VALUE_BACK', 'bets': bets_df if pass_gate else None,
#             }
#
#         else:
#             # CLASSIFY markets: use candidate threshold
#             thr = float(cand['threshold'])
#             y_pred = (p_pos_test >= thr).astype(np.uint8)
#             n_preds_test = int(y_pred.sum())
#             prc_test = precision_score(y_test, y_pred, zero_division=0)
#             acc_test = accuracy_score(y_test, y_pred)
#             enough = n_preds_test >= int(min_test_samples)
#             not_collapsed = prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop))
#             pass_gate = bool(enough and not_collapsed)
#             reason = "" if pass_gate else ("insufficient_test_preds" if not enough else "precision_collapse")
#             rec = {
#                 **best_params, 'threshold': thr, 'edge_param': np.nan,
#                 'val_precision_lcb': cand['val_precision_lcb'], 'val_precision': cand['val_precision'],
#                 'val_accuracy': cand['val_accuracy'],
#                 'n_preds_test': n_preds_test, 'test_precision': float(prc_test), 'test_accuracy': float(acc_test),
#                 'pass_test_gate': pass_gate, 'fail_reason': reason, 'model_obj': final_calibrator if pass_gate else None,
#                 'mode': 'CLASSIFY', 'bets': None,
#             }
#
#         records_all.append(rec)
#
#     survivors_df = pd.DataFrame(records_all)
#     passers = survivors_df[survivors_df['pass_test_gate']].copy()
#
#     # ---------------- save / rank ----------------
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     tag = "xgb" if base_model == "xgb" else "mlp"
#
#     if passers.empty:
#         fail_csv = None
#         if save_diagnostics_on_fail:
#             sort_cols = (['p_value','total_pl','val_precision_lcb'] if _USE_VALUE
#                          else ['val_precision_lcb','val_precision','n_preds_test','val_accuracy'])
#             asc = ([True, False, False] if _USE_VALUE else [False, False, False, False])
#             diag = (survivors_df
#                     .drop(columns=['model_obj','bets'], errors='ignore')
#                     .sort_values(by=sort_cols, ascending=asc)
#                     .assign(market=market))
#             fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv")
#             diag.to_csv(fail_csv, index=False); summary_df = diag
#         else:
#             summary_df = survivors_df.drop(columns=['model_obj','bets'], errors='ignore')
#
#         all_bets_csv_path = None
#         if save_all_bets_csv and _USE_VALUE and all_bets_collector:
#             all_bets_df = pd.concat(all_bets_collector, ignore_index=True)
#             if not all_bets_include_failed:
#                 all_bets_df = all_bets_df[all_bets_df['passed_test_gate'] == True]
#             all_bets_csv_path = os.path.join(all_bets_dir, f"all_bets_{market}_{timestamp}.csv")
#             all_bets_df.to_csv(all_bets_csv_path, index=False)
#
#         msg = "All Top-K failed the TEST gate."
#         if on_fail == "raise": raise RuntimeError(msg)
#         if on_fail == "warn": print("[WARN]", msg)
#         return {'status':'failed_test_gate','csv':fail_csv,'model_pkl':None,
#                 'summary_df':summary_df,'validation_table':ranked,
#                 'bets_csv':None,'pl_plot':None,'all_bets_csv':all_bets_csv_path}
#
#     # Final ranking
#     if _USE_VALUE:
#         passers_sorted = passers.sort_values(
#             by=['p_value','total_pl','avg_pl','val_precision_lcb','val_precision'],
#             ascending=[True, False, False, False, False]
#         ).reset_index(drop=True)
#     else:
#         passers_sorted = passers.sort_values(
#             by=['val_precision_lcb','val_precision','test_precision','n_preds_test','val_accuracy'],
#             ascending=[False, False, False, False, False]
#         ).reset_index(drop=True)
#
#     # Save PKL + CSV
#     pkl_path = os.path.join(model_dir, f"best_model_{market}_{tag}_calibrated_{timestamp}.pkl")
#     csv_df = passers_sorted.drop(columns=['model_obj','bets'], errors='ignore').copy()
#     csv_df['model_pkl'] = ""; csv_df.loc[0, 'model_pkl'] = pkl_path
#     csv_df['market'] = market
#     csv_path = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}.csv")
#     csv_df.to_csv(csv_path, index=False)
#
#     # Save top model
#     top_row = passers_sorted.iloc[0]
#     chosen_model = top_row['model_obj']
#     if base_model == "xgb":
#         param_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']
#     else:
#         param_keys = ['hidden_layer_sizes','alpha','learning_rate_init','batch_size','max_iter']
#     chosen_params = {k: top_row[k] for k in param_keys if k in passers_sorted.columns}
#     chosen_threshold = float(top_row.get('threshold', np.nan))
#     chosen_edge = float(top_row.get('edge_param', np.nan))
#
#     joblib.dump(
#         {
#             'model': chosen_model,
#             'threshold': chosen_threshold,   # NaN in VALUE modes; meaningful in CLASSIFY
#             'edge_param': chosen_edge,       # meaningful in VALUE modes
#             'features': features,
#             'base_model': base_model,
#             'best_params': chosen_params,
#             'precision_test_threshold': float(precision_test_threshold),
#             'min_samples': int(min_samples),
#             'min_test_samples': int(min_test_samples),
#             'val_conf_level': float(val_conf_level),
#             'max_precision_drop': float(max_precision_drop),
#             'market': market,
#             'mode': top_row['mode'],
#             'liability': float(top_row.get('liability', np.nan)) if _USE_VALUE_LAY else None,
#             'stake': float(top_row.get('stake', np.nan)) if _USE_VALUE_BACK else None,
#             'p_value': float(top_row.get('p_value', np.nan)) if _USE_VALUE else None,
#             'zscore': float(top_row.get('zscore', np.nan)) if _USE_VALUE else None,
#             'notes': ('VALUE modes ranked by p-value vs break-even and P/L; '
#                       'VALUE modes sweep edge_param; CLASSIFY markets sweep thresholds.'),
#             'run_seed': int(RUN_SEED),
#         },
#         pkl_path
#     )
#
#     # chosen bets CSV / plot
#     bets_path = None
#     plot_path = None
#     if _USE_VALUE and (save_bets_csv or plot_pl):
#         bets_df = top_row.get('bets', None)
#         if isinstance(bets_df, pd.DataFrame) and len(bets_df):
#             if save_bets_csv:
#                 bets_name = f"bets_{market}_{timestamp}.csv"
#                 bets_path = os.path.join(bets_csv_dir, bets_name)
#                 bets_df.to_csv(bets_path, index=False)
#             if plot_pl:
#                 try:
#                     import matplotlib.pyplot as plt
#                     fig = plt.figure()
#                     x = bets_df['date'] if 'date' in bets_df.columns else np.arange(len(bets_df))
#                     plt.plot(x, bets_df['cum_pl'])
#                     title = f"{market} cumulative P/L ({'VALUE_LAY' if _USE_VALUE_LAY else 'VALUE_BACK'})"
#                     if plot_title_suffix: title += f" — {plot_title_suffix}"
#                     plt.title(title); plt.xlabel('Date' if 'date' in bets_df.columns else 'Bet #'); plt.ylabel('Cumulative P/L')
#                     plt.tight_layout()
#                     plot_name = f"cum_pl_{market}_{timestamp}.png"
#                     plot_path = os.path.join(plot_dir, plot_name)
#                     plt.savefig(plot_path, dpi=160); plt.close(fig)
#                 except Exception as e:
#                     print(f"[WARN] Failed to create plot: {e}")
#
#     # ALL bets export
#     all_bets_csv_path = None
#     if save_all_bets_csv and _USE_VALUE and all_bets_collector:
#         all_bets_df = pd.concat(all_bets_collector, ignore_index=True)
#         if not all_bets_include_failed:
#             all_bets_df = all_bets_df[all_bets_df['passed_test_gate'] == True]
#         preferred = [c for c in [
#             'date','league','country','home_team','away_team','match_id','event_name','selection',
#             'market_odds','fair_odds','edge_ratio','stake','liability','selection_won','target','pl','cum_pl',
#             'candidate_id','passed_test_gate','mode','market','threshold','edge_param',
#             'val_precision','val_precision_lcb','n_value_bets_test','total_pl','avg_pl','p_value','zscore','params_json'
#         ] if c in all_bets_df.columns]
#         all_bets_df = all_bets_df[preferred + [c for c in all_bets_df.columns if c not in preferred]]
#         all_bets_csv_path = os.path.join(all_bets_dir, f"all_bets_{market}_{timestamp}.csv")
#         all_bets_df.to_csv(all_bets_csv_path, index=False)
#
#     return {
#         'status': 'ok',
#         'csv': csv_path,
#         'model_pkl': pkl_path,
#         'summary_df': csv_df,
#         'validation_table': ranked,
#         'bets_csv': bets_path,
#         'pl_plot': plot_path,
#         'all_bets_csv': all_bets_csv_path,
#     }


# def run_models_outcome(
#     matches_filtered: pd.DataFrame,
#     features: list,
#     # ── gates ──────────────────────────────────────────────────────────────
#     min_samples: int = 200,
#     min_test_samples: int = 100,
#     precision_test_threshold: float = 0.80,
#     # ── model/search ───────────────────────────────────────────────────────
#     base_model: str = "xgb",               # "xgb" or "mlp"
#     search_mode: str = "random",           # "random" or "grid"
#     n_random_param_sets: int = 10,
#     cpu_jobs: int = 6,
#     top_k: int = 10,
#     thresholds: np.ndarray | None = None,  # USED only for CLASSIFY markets
#     out_dir: str | None = None,
#     # ── anti-overfitting ──────────────────────────────────────────────────
#     val_conf_level: float = 0.99,
#     max_precision_drop: float = 0.05,
#     # ── failure handling ───────────────────────────────────────────────────
#     on_fail: str = "return",               # "return" | "warn" | "raise"
#     save_diagnostics_on_fail: bool = True,
#     # ── market ─────────────────────────────────────────────────────────────
#     market: str = "LAY_AWAY",              # LAY_* | BACK_* | OVER | UNDER (or other classify markets)
#
#     # ── VALUE LAY controls ────────────────────────────────────────────────
#     use_value_for_lay: bool = True,
#     value_edge_grid_lay: np.ndarray | None = None,   # e.g. np.round(np.arange(0.00,0.201,0.01),2)
#
#     # ── VALUE BACK controls ────────────────────────────────────────────────
#     use_value_for_back: bool = False,
#     value_edge_grid_back: np.ndarray | None = None,
#
#     # ── staking plan search (toggle) ───────────────────────────────────────
#     enable_staking_plan_search: bool = True,
#     staking_plan_lay_options: list[str] | None = None,   # ["liability","flat_stake","edge_prop","kelly_approx"]
#     staking_plan_back_options: list[str] | None = None,  # ["flat","edge_prop","kelly"]
#
#     # ── single-plan (used when enable_staking_plan_search=False) ──────────
#     staking_plan_lay: str = "liability",
#     staking_plan_back: str = "flat",
#
#     # ── LAY staking parameters (balanced defaults) ────────────────────────
#     liability_test: float = 1.0,           # base liability for "liability" plan
#     lay_flat_stake: float = 0.50,          # stake for "flat_stake"
#     lay_edge_scale: float = 0.05,          # liability ∝ edge/lay_edge_scale for "edge_prop"
#     kelly_fraction_lay: float = 1.0,       # scale for "kelly_approx" (0..1)
#     min_lay_stake: float = 0.0,
#     max_lay_stake: float = 1.0,
#     min_lay_liability: float = 0.0,
#     max_lay_liability: float = 2.0,
#
#     # ── BACK staking parameters ────────────────────────────────────────────
#     back_stake_test: float = 1.0,          # base stake for "flat"
#     back_edge_scale: float = 0.10,         # stake ∝ edge/back_edge_scale for "edge_prop"
#     kelly_fraction_back: float = 0.25,     # Kelly fraction (0..1)
#     bankroll_back: float = 100.0,          # bankroll for kelly stake
#     min_back_stake: float = 0.0,
#     max_back_stake: float = 10.0,
#
#     # ── COMMISSION (applied to net winning returns) ───────────────────────
#     commission_rate: float = 0.02,         # 2% commission on winnings
#
#     # ── OUTPUTS: chosen model ─────────────────────────────────────────────
#     save_bets_csv: bool = False,
#     bets_csv_dir: str | None = None,
#     plot_pl: bool = False,
#     plot_dir: str | None = None,
#     plot_title_suffix: str = "",
#     # ── OUTPUTS: ALL candidates (debug/exploration) ───────────────────────
#     save_all_bets_csv: bool = False,
#     all_bets_dir: str | None = None,
#     all_bets_include_failed: bool = True,
# ):
#     """
#     Rolling time-ordered CV with calibration.
#
#     VALUE LAY (market starts with LAY_):
#       fair_odds = 1 / P(selection wins). Place a lay if fair ≥ market × (1+edge).
#       Commission on winning lay: profit = stake * (1 - commission_rate).
#       Loss if selection wins: −liability.
#
#     VALUE BACK (market starts with BACK_):
#       Place a back if market ≥ fair × (1+edge).
#       Commission on winning back: profit = (odds−1)*stake * (1 - commission_rate).
#       Loss if selection loses: −stake.
#
#     CLASSIFY (e.g., OVER/UNDER):
#       Sweep probability thresholds (argument `thresholds`) during validation and
#       carry the best threshold into test. (Value staking plans are not used here.)
#     """
#     # ---------------- setup ----------------
#     import os, secrets, hashlib, json
#     from datetime import datetime
#     import numpy as np
#     import pandas as pd
#     from itertools import product
#     from sklearn.model_selection import ParameterSampler
#     from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, log_loss, brier_score_loss
#     from sklearn.calibration import CalibratedClassifierCV
#     from sklearn.pipeline import Pipeline, make_pipeline
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.neural_network import MLPClassifier
#     from joblib import Parallel, delayed, parallel_backend
#     from tqdm import tqdm
#     from tqdm_joblib import tqdm_joblib
#     import joblib
#
#     # --- xgboost import (optional)
#     try:
#         import xgboost as xgb
#         _HAS_XGB_LOCAL = True
#     except Exception:
#         _HAS_XGB_LOCAL = False
#     _HAS_XGB = globals().get("_HAS_XGB", _HAS_XGB_LOCAL)
#     if base_model == "xgb" and not _HAS_XGB:
#         raise ImportError("XGBoost not available; set base_model='mlp' or install xgboost.")
#
#     # --- random dists
#     try:
#         _randint; _uniform; _loguniform
#     except NameError:
#         from scipy.stats import randint as _randint
#         from scipy.stats import uniform as _uniform
#         from scipy.stats import loguniform as _loguniform
#
#     # --- Wilson LCB & normal CDF
#     try:
#         from scipy.stats import norm
#         _Z = lambda conf: float(norm.ppf(1 - (1 - conf) / 2))
#         _Phi = lambda z: float(norm.cdf(z))
#     except Exception:
#         import math
#         _Z = lambda conf: 1.96 if abs(conf - 0.95) < 1e-6 else 1.64
#         _Phi = lambda z: 0.5 * (1.0 + math.erf(z / (2**0.5)))
#
#     def _wilson_lcb(tp: int, fp: int, conf: float) -> float:
#         n = tp + fp
#         if n <= 0: return 0.0
#         p = tp / n
#         z = _Z(conf)
#         denom = 1.0 + (z*z)/n
#         centre = p + (z*z)/(2*n)
#         rad = z * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
#         return max(0.0, (centre - rad) / denom)
#
#     # defaults
#     if thresholds is None:
#         thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)  # CLASSIFY only
#     if value_edge_grid_lay is None:
#         value_edge_grid_lay = np.round(np.arange(0.00, 0.201, 0.01), 2)
#     if value_edge_grid_back is None:
#         value_edge_grid_back = np.round(np.arange(0.00, 0.201, 0.01), 2)
#
#     # normalise staking-plan options
#     if staking_plan_lay_options is None:
#         staking_plan_lay_options = ["liability", "flat_stake", "edge_prop", "kelly_approx"]
#     if staking_plan_back_options is None:
#         staking_plan_back_options = ["flat", "edge_prop", "kelly"]
#
#     if not enable_staking_plan_search:
#         staking_plan_lay_options = [staking_plan_lay]
#         staking_plan_back_options = [staking_plan_back]
#
#     # --- paths (metrics vs bets kept separate)
#     BASE = r"C:\Users\leere\PycharmProjects\Football_ML3\FT Results"
#
#     PKL_DIRS = {
#         "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "model_file"),
#         "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "model_file"),
#         "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "model_file"),
#         "BACK_HOME": os.path.join(BASE, "Back_Home", "model_file"),
#         "BACK_AWAY": os.path.join(BASE, "Back_Away", "model_file"),
#         "BACK_DRAW": os.path.join(BASE, "Back_Draw", "model_file"),
#         "OVER":      os.path.join(BASE, "Over_2_5",  "model_file"),
#         "UNDER":     os.path.join(BASE, "Under_2_5", "model_file"),
#     }
#
#     # Metrics (model selection summaries etc.)
#     CSV_DIRS = {
#         "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "best_model_metrics"),
#         "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "best_model_metrics"),
#         "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "best_model_metrics"),
#         "BACK_HOME": os.path.join(BASE, "Back_Home", "best_model_metrics"),
#         "BACK_AWAY": os.path.join(BASE, "Back_Away", "best_model_metrics"),
#         "BACK_DRAW": os.path.join(BASE, "Back_Draw", "best_model_metrics"),
#         "OVER":      os.path.join(BASE, "Over_2_5",  "best_model_metrics"),
#         "UNDER":     os.path.join(BASE, "Under_2_5", "best_model_metrics"),
#     }
#
#     # Dedicated bet-level folders (kept OUT of best_model_metrics)
#     BETS_DIRS = {
#         "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "bets"),
#         "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "bets"),
#         "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "bets"),
#         "BACK_HOME": os.path.join(BASE, "Back_Home", "bets"),
#         "BACK_AWAY": os.path.join(BASE, "Back_Away", "bets"),
#         "BACK_DRAW": os.path.join(BASE, "Back_Draw", "bets"),
#         "OVER":      os.path.join(BASE, "Over_2_5",  "bets"),
#         "UNDER":     os.path.join(BASE, "Under_2_5", "bets"),
#     }
#
#     ALL_BETS_DIRS = {k: os.path.join(os.path.dirname(CSV_DIRS[k]), "all_bets") for k in CSV_DIRS}
#
#     market = str(market).upper().strip()
#     if market not in PKL_DIRS or market not in CSV_DIRS:
#         raise ValueError(f"Unsupported market '{market}'.")
#
#     _IS_LAY  = market.startswith("LAY_")
#     _IS_BACK = market.startswith("BACK_")
#     _USE_VALUE_LAY  = bool(use_value_for_lay and _IS_LAY)
#     _USE_VALUE_BACK = bool(use_value_for_back and _IS_BACK)
#     _USE_VALUE = _USE_VALUE_LAY or _USE_VALUE_BACK
#     _IS_CLASSIFY = not _USE_VALUE
#
#     # Metrics/model destinations
#     csv_save_dir = out_dir if (out_dir and len(str(out_dir)) > 0) else CSV_DIRS[market]
#     os.makedirs(csv_save_dir, exist_ok=True)
#     model_dir = PKL_DIRS[market]
#     os.makedirs(model_dir, exist_ok=True)
#
#     # Bets destinations (defaults away from metrics dir)
#     if bets_csv_dir is None:
#         bets_csv_dir = BETS_DIRS[market]
#     if all_bets_dir is None:
#         all_bets_dir = ALL_BETS_DIRS[market]
#     if plot_dir is None:
#         plot_dir = BETS_DIRS[market]
#     os.makedirs(bets_csv_dir, exist_ok=True)
#     os.makedirs(all_bets_dir, exist_ok=True)
#     os.makedirs(plot_dir, exist_ok=True)
#
#     RUN_SEED = secrets.randbits(32)
#     def _seed_from(*vals) -> int:
#         h = hashlib.blake2b(digest_size=8)
#         h.update(int(RUN_SEED).to_bytes(8,'little',signed=False))
#         for v in vals: h.update(str(v).encode('utf-8'))
#         return int.from_bytes(h.digest(),'little') & 0x7FFFFFFF
#
#     def _as_float(x):
#         try: return float(x)
#         except Exception: return float(str(x))
#     def _as_int(x): return int(float(x))
#
#     # ---------------- data ----------------
#     req_cols = {'date','target'}
#     if _USE_VALUE: req_cols |= {'home_odds','draw_odds','away_odds'}
#     missing = req_cols - set(matches_filtered.columns)
#     if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
#
#     df = matches_filtered.copy()
#     df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     df = df.sort_values('date').reset_index(drop=True)
#
#     cols_needed = list(set(features) | {'target'} | ({'home_odds','draw_odds','away_odds'} if _USE_VALUE else set()))
#     df = df.dropna(subset=cols_needed).reset_index(drop=True)
#
#     X = df[features].copy()
#     y = df['target'].astype(int).reset_index(drop=True)
#
#     n = len(X)
#     if n < max(min_samples * 3, 500): raise RuntimeError(f"Not enough rows: {n}")
#
#     # temporal split
#     test_start = int(0.85 * n)
#     pretest_end = test_start
#     X_test = X.iloc[test_start:].reset_index(drop=True)
#     y_test = y.iloc[test_start:].reset_index(drop=True)
#     df_test = df.iloc[test_start:].reset_index(drop=True)
#
#     # rolling validation folds
#     N_FOLDS = 5
#     total_val_len = max(1, int(0.15 * n))
#     val_len = max(1, total_val_len // N_FOLDS)
#     fold_val_ends = [pretest_end - total_val_len + (i + 1) * val_len for i in range(N_FOLDS)]
#     fold_val_starts = [end - val_len for end in fold_val_ends]
#     if fold_val_ends:
#         fold_val_ends[-1] = min(fold_val_ends[-1], pretest_end)
#         fold_val_starts[-1] = max(0, fold_val_ends[-1] - val_len)
#
#     # final small validation slice (for calibration before test)
#     final_val_len = max(1, val_len)
#     final_val_start = max(0, test_start - final_val_len)
#     X_train_final = X.iloc[:final_val_start]
#     y_train_final = y.iloc[:final_val_start]
#     X_val_final   = X.iloc[final_val_start:test_start]
#     y_val_final   = y.iloc[final_val_start:test_start]
#
#     # ---------------- param spaces ----------------
#     xgb_param_grid = {'n_estimators':[200],'max_depth':[5],'learning_rate':[0.1],'subsample':[0.7],
#                       'colsample_bytree':[1.0],'min_child_weight':[5],'reg_lambda':[1.0]}
#     xgb_param_distributions = {'n_estimators':_randint(100,1001),'max_depth':_randint(3,8),
#                                'learning_rate':_loguniform(0.01,0.2),'min_child_weight':_randint(3,13),
#                                'subsample':_uniform(0.7,0.3),'colsample_bytree':_uniform(0.6,0.4),
#                                'reg_lambda':_loguniform(0.1,10.0)}
#     mlp_param_grid = {'hidden_layer_sizes':[(128,),(256,),(128,64)],'alpha':[1e-4],
#                       'learning_rate_init':[1e-3],'batch_size':['auto'],'max_iter':[200]}
#     mlp_param_distributions = {'hidden_layer_sizes':[(64,),(128,),(256,),(128,64),(256,128)],
#                                'alpha':_loguniform(1e-5,1e-2),'learning_rate_init':_loguniform(5e-4,5e-2),
#                                'batch_size':_randint(32,257),'max_iter':_randint(150,401)}
#
#     def cast_params(p: dict) -> dict:
#         q = dict(p)
#         if base_model == "xgb":
#             for k in ['n_estimators','max_depth','min_child_weight']:
#                 if k in q: q[k] = _as_int(q[k])
#             for k in ['learning_rate','subsample','colsample_bytree','reg_lambda']:
#                 if k in q: q[k] = _as_float(q[k])
#         else:
#             if 'max_iter' in q: q['max_iter'] = _as_int(q['max_iter'])
#             if 'batch_size' in q and q['batch_size'] != 'auto': q['batch_size'] = _as_int(q['batch_size'])
#             if 'alpha' in q: q['alpha'] = _as_float(q['alpha'])
#             if 'learning_rate_init' in q: q['learning_rate_init'] = _as_float(q['learning_rate_init'])
#             if 'hidden_layer_sizes' in q:
#                 h = q['hidden_layer_sizes']
#                 if isinstance(h, str):
#                     parts = [pp.strip() for pp in h.strip("()").split(",") if pp.strip()!='']
#                     q['hidden_layer_sizes'] = tuple(_as_int(pp) for pp in parts) if parts else (128,)
#                 elif isinstance(h, (list, tuple, np.ndarray)):
#                     q['hidden_layer_sizes'] = tuple(int(v) for v in h)
#                 else:
#                     q['hidden_layer_sizes'] = (int(h),)
#         return q
#
#     def _final_step_name(estimator):
#         try:
#             if isinstance(estimator, Pipeline): return estimator.steps[-1][0]
#         except Exception:
#             pass
#         return None
#
#     def build_model(params: dict, spw: float):
#         model_seed = _seed_from("model", base_model, tuple(sorted(params.items())))
#         if base_model == "xgb":
#             return xgb.XGBClassifier(
#                 objective='binary:logistic',
#                 eval_metric='auc',
#                 random_state=model_seed,
#                 scale_pos_weight=spw,
#                 n_jobs=1,
#                 tree_method="hist",
#                 verbosity=0,
#                 **params
#             )
#         else:
#             mlp = MLPClassifier(
#                 random_state=model_seed,
#                 early_stopping=True,
#                 n_iter_no_change=20,
#                 validation_fraction=0.1,
#                 solver="adam",
#                 **params
#             )
#             return make_pipeline(StandardScaler(with_mean=True, with_std=True), mlp)
#
#     def fit_model(model, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None):
#         if base_model == "xgb":
#             try:
#                 model.set_params(verbosity=0, early_stopping_rounds=50)
#                 if X_va is not None and y_va is not None:
#                     model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
#                 else:
#                     model.fit(X_tr, y_tr, verbose=False)
#             except Exception:
#                 model.fit(X_tr, y_tr, verbose=False)
#         else:
#             fit_kwargs = {}
#             if sample_weight is not None:
#                 stepname = _final_step_name(model)
#                 if stepname is not None:
#                     fit_kwargs[f"{stepname}__sample_weight"] = sample_weight
#             try:
#                 model.fit(X_tr, y_tr, **fit_kwargs)
#             except TypeError:
#                 model.fit(X_tr, y_tr)
#
#     def fit_calibrator(fitted, X_va, y_va):
#         try:
#             from sklearn.calibration import FrozenEstimator
#             frozen = FrozenEstimator(fitted)
#             cal = CalibratedClassifierCV(frozen, method='sigmoid', cv=None)
#             cal.fit(X_va, y_va)
#             return cal
#         except Exception:
#             try:
#                 cal = CalibratedClassifierCV(fitted, method='sigmoid', cv='prefit')
#                 cal.fit(X_va, y_va); return cal
#             except Exception:
#                 return fitted
#
#     def _unwrap_estimator(est):
#         if isinstance(est, Pipeline): return est.steps[-1][1]
#         return est
#
#     def predict_proba_pos(model_or_cal, X_):
#         proba = model_or_cal.predict_proba(X_)
#         if proba.ndim == 2:
#             classes = getattr(model_or_cal, "classes_", None)
#             if classes is None:
#                 base = _unwrap_estimator(model_or_cal); classes = getattr(base, "classes_", None)
#             if classes is not None and len(classes) == proba.shape[1]:
#                 try:
#                     idx = int(np.where(np.asarray(classes) == 1)[0][0])
#                     return proba[:, idx].astype(np.float32)
#                 except Exception:
#                     pass
#             if proba.shape[1] == 2: return proba[:, 1].astype(np.float32)
#             if proba.shape[1] == 1:
#                 only = getattr(model_or_cal, "classes_", [0])[0]
#                 return (np.ones_like(proba[:,0]) if only==1 else np.zeros_like(proba[:,0])).astype(np.float32)
#         return np.asarray(proba, dtype=np.float32)
#
#     # --- p-value helper (VALUE modes) with commission-adjusted pay-offs ----
#     def _pvalue_break_even(bdf: pd.DataFrame, mode: str) -> dict:
#         if not isinstance(bdf, pd.DataFrame) or bdf.empty:
#             return {'z': 0.0, 'p_value': 1.0, 'var_sum': 0.0, 'n': 0, 'total_pl': 0.0}
#         o = np.asarray(bdf['market_odds'].values, dtype=float)
#         o = np.where(o <= 1.0, np.nan, o)
#         p = 1.0 / o  # null probability (selection wins)
#         if mode == 'VALUE_BACK':
#             S = np.asarray(bdf['stake'].values, dtype=float)
#             win = (o - 1.0) * S * (1.0 - commission_rate)
#             lose = -S
#         else:  # VALUE_LAY
#             L = np.asarray(bdf['liability'].values, dtype=float)  # loss if selection wins
#             stake = np.asarray(bdf['stake'].values, dtype=float)  # gross win if selection loses
#             win  = stake * (1.0 - commission_rate)
#             lose = -L
#         var_i = p * (win ** 2) + (1.0 - p) * (lose ** 2)
#         var_i = np.where(np.isfinite(var_i), var_i, 0.0)
#         pl = np.asarray(bdf['pl'].values, dtype=float)
#         total_pl = float(np.nansum(np.where(np.isfinite(pl), pl, 0.0)))
#         var_sum = float(np.nansum(var_i))
#         z = total_pl / (np.sqrt(var_sum) + 1e-12)
#         p_val = max(0.0, 1.0 - _Phi(z))  # one-sided
#         return {'z': float(z), 'p_value': float(p_val), 'var_sum': var_sum, 'n': int(len(pl)), 'total_pl': total_pl}
#
#     # --- staking calculators ------------------------------------------------
#     def _lay_stakes(odds: np.ndarray, fair_over_market_minus1: np.ndarray, plan: str):
#         o = np.asarray(odds, dtype=float)
#         edge_plus = np.maximum(fair_over_market_minus1, 0.0)
#
#         if plan == "liability":
#             L = np.full_like(o, float(liability_test), dtype=float)
#             stake = np.divide(L, np.maximum(o - 1.0, 1e-9))
#         elif plan == "flat_stake":
#             stake = np.full_like(o, float(lay_flat_stake), dtype=float)
#             L = stake * (o - 1.0)
#         elif plan == "edge_prop":
#             L = float(liability_test) * np.divide(edge_plus, max(1e-9, float(lay_edge_scale)))
#             L = np.clip(L, float(min_lay_liability), float(max_lay_liability))
#             stake = np.divide(L, np.maximum(o - 1.0, 1e-9))
#         elif plan == "kelly_approx":
#             L = float(liability_test) * float(kelly_fraction_lay) * edge_plus
#             L = np.clip(L, float(min_lay_liability), float(max_lay_liability))
#             stake = np.divide(L, np.maximum(o - 1.0, 1e-9))
#         else:
#             raise ValueError(f"Unknown staking_plan_lay: {plan}")
#
#         stake = np.clip(stake, float(min_lay_stake), float(max_lay_stake))
#         L = np.clip(L, float(min_lay_liability), float(max_lay_liability))
#         return stake, L
#
#     def _back_stakes(odds: np.ndarray, fair_over_market_minus1: np.ndarray, plan: str, p_win: np.ndarray):
#         o = np.asarray(odds, dtype=float)
#         p = np.clip(np.asarray(p_win, dtype=float), 0.0, 1.0)
#         edge_plus = np.maximum(fair_over_market_minus1, 0.0)
#
#         if plan == "flat":
#             stake = np.full_like(o, float(back_stake_test), dtype=float)
#         elif plan == "edge_prop":
#             stake = float(back_stake_test) * np.divide(edge_plus, max(1e-9, float(back_edge_scale)))
#         elif plan == "kelly":
#             b = np.maximum(o - 1.0, 1e-9)
#             f = (b * p - (1.0 - p)) / b
#             f = np.maximum(f, 0.0)
#             stake = float(bankroll_back) * float(kelly_fraction_back) * f
#         else:
#             raise ValueError(f"Unknown staking_plan_back: {plan}")
#
#         stake = np.clip(stake, float(min_back_stake), float(max_back_stake))
#         return stake
#
#     # ---------------- search space ----------------
#     if search_mode.lower() == "grid":
#         grid = xgb_param_grid if base_model == "xgb" else mlp_param_grid
#         all_param_dicts = [dict(zip(grid.keys(), combo)) for combo in product(*grid.values())]
#     else:
#         dists = xgb_param_distributions if base_model == "xgb" else mlp_param_distributions
#         sampler_seed = _seed_from("sampler")
#         all_param_dicts = list(ParameterSampler(dists, n_iter=n_random_param_sets, random_state=sampler_seed))
#
#     # ---------------- validation eval ----------------
#     def evaluate_param_set(param_dict, *_):
#         safe = cast_params(param_dict)
#         rows = []; val_prob_all=[]; val_true_all=[]
#
#         for vstart, vend in zip(fold_val_starts, fold_val_ends):
#             if vstart is None or vend is None or vstart <= 0 or vend <= vstart: continue
#
#             X_tr, y_tr = X.iloc[:vstart], y.iloc[:vstart]
#             X_va, y_va = X.iloc[vstart:vend], y.iloc[vstart:vend]
#             df_va = df.iloc[vstart:vend]
#             if y_tr.nunique() < 2: continue
#
#             pos = int(y_tr.sum()); neg = len(y_tr) - pos
#             spw = (neg/pos) if pos > 0 else 1.0
#
#             sample_weight = None
#             if base_model == "mlp":
#                 w_pos = spw
#                 sample_weight = np.where(y_tr.values==1, w_pos, 1.0).astype(np.float32)
#
#             model = build_model(safe, spw)
#             fit_model(model, X_tr, y_tr, X_va, y_va, sample_weight=sample_weight)
#             cal = fit_calibrator(model, X_va, y_va)
#
#             p_pos = predict_proba_pos(cal, X_va)
#             val_prob_all.append(p_pos)
#             y_true = y_va.values.astype(np.uint8); val_true_all.append(y_true)
#
#             if _IS_CLASSIFY:
#                 for thr in thresholds:
#                     thr = float(thr)
#                     y_pred = (p_pos >= thr).astype(np.uint8)
#                     n_preds = int(y_pred.sum())
#                     tp = int(((y_true == 1) & (y_pred == 1)).sum())
#                     fp = int(((y_true == 0) & (y_pred == 1)).sum())
#                     prc = precision_score(y_va, y_pred, zero_division=0)
#                     acc = accuracy_score(y_va, y_pred)
#                     rows.append({
#                         **safe,
#                         'threshold': thr,
#                         'edge_param': np.nan,
#                         'fold_vstart': int(vstart),
#                         'fold_vend': int(vend),
#                         'n_preds_val': n_preds,
#                         'tp_val': tp,
#                         'fp_val': fp,
#                         'val_precision': float(prc),
#                         'val_accuracy': float(acc),
#                         'n_value_bets_val': 0,
#                         'val_edge_ratio_mean': np.nan,
#                         'val_edge_ratio_mean_back': np.nan,
#                     })
#             else:
#                 edge_grid = value_edge_grid_lay if _USE_VALUE_LAY else value_edge_grid_back
#                 for edge_param in edge_grid:
#                     if _USE_VALUE_LAY:
#                         if market == "LAY_AWAY":
#                             p_sel_win = 1.0 - p_pos; mkt = df_va['away_odds'].values
#                         elif market == "LAY_HOME":
#                             p_sel_win = 1.0 - p_pos; mkt = df_va['home_odds'].values
#                         else:
#                             p_sel_win = 1.0 - p_pos; mkt = df_va['draw_odds'].values
#                         fair = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
#                         edge_mask = (fair >= (1.0 + float(edge_param)) * mkt) & np.isfinite(mkt)
#                         with np.errstate(divide='ignore', invalid='ignore'):
#                             edge_ratio = fair / mkt
#                         val_edge_mean = float(np.nanmean(np.where(np.isfinite(edge_ratio), edge_ratio, np.nan)))
#                         n_bets = int(np.nansum(edge_mask))
#                     else:
#                         if market == "BACK_AWAY":
#                             p_sel_win = p_pos; mkt = df_va['away_odds'].values
#                         elif market == "BACK_HOME":
#                             p_sel_win = p_pos; mkt = df_va['home_odds'].values
#                         else:
#                             p_sel_win = p_pos; mkt = df_va['draw_odds'].values
#                         fair = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
#                         edge_mask = (mkt >= (1.0 + float(edge_param)) * fair) & np.isfinite(mkt)
#                         with np.errstate(divide='ignore', invalid='ignore'):
#                             edge_ratio = mkt / fair
#                         val_edge_mean = float(np.nanmean(np.where(np.isfinite(edge_ratio), edge_ratio, np.nan)))
#                         n_bets = int(np.nansum(edge_mask))
#
#                     y_pred = np.zeros_like(y_true, dtype=np.uint8)
#                     y_pred[np.where(edge_mask)[0]] = 1
#                     tp = int(((y_true == 1) & (y_pred == 1)).sum())
#                     fp = int(((y_true == 0) & (y_pred == 1)).sum())
#                     prc = precision_score(y_true, y_pred, zero_division=0)
#                     acc = accuracy_score(y_true, y_pred)
#
#                     rows.append({
#                         **safe,
#                         'threshold': np.nan,
#                         'edge_param': float(edge_param),
#                         'fold_vstart': int(vstart),
#                         'fold_vend': int(vend),
#                         'n_preds_val': int(y_pred.sum()),
#                         'tp_val': tp,
#                         'fp_val': fp,
#                         'val_precision': float(prc),
#                         'val_accuracy': float(acc),
#                         'n_value_bets_val': int(n_bets),
#                         'val_edge_ratio_mean': val_edge_mean if _USE_VALUE_LAY else np.nan,
#                         'val_edge_ratio_mean_back': val_edge_mean if _USE_VALUE_BACK else np.nan,
#                     })
#
#         # pooled diagnostics
#         if val_prob_all:
#             vp = np.concatenate(val_prob_all, axis=0)
#             vt = np.concatenate(val_true_all, axis=0)
#             try: val_auc = float(roc_auc_score(vt, vp))
#             except Exception: val_auc = np.nan
#             try: val_ll  = float(log_loss(vt, vp, labels=[0, 1]))
#             except Exception: val_ll = np.nan
#             try: val_bri = float(brier_score_loss(vt, vp))
#             except Exception: val_bri = np.nan
#         else:
#             val_auc = val_ll = val_bri = np.nan
#
#         for r in rows:
#             r['val_auc'] = val_auc
#             r['val_logloss'] = val_ll
#             r['val_brier'] = val_bri
#
#         return rows
#
#     # ---------------- search ----------------
#     if base_model == "mlp":
#         eff_jobs = min(max(1, cpu_jobs), 4); prefer = "threads"; backend = "threading"; pre_dispatch = eff_jobs
#         ctx = parallel_backend(backend, n_jobs=eff_jobs)
#     else:
#         eff_jobs = max(1, min(cpu_jobs, 4)) if cpu_jobs != -1 else 4
#         prefer = "processes"; backend = "loky"; pre_dispatch = f"{2*eff_jobs}"
#         ctx = parallel_backend(backend, n_jobs=eff_jobs, inner_max_num_threads=1)
#
#     with ctx:
#         try:
#             with tqdm_joblib(tqdm(total=len(all_param_dicts), desc=f"Param search ({search_mode}, {base_model})")) as _:
#                 out = Parallel(n_jobs=eff_jobs, batch_size=1, prefer=prefer, pre_dispatch=pre_dispatch)(
#                     delayed(evaluate_param_set)(pd_) for pd_ in all_param_dicts
#                 )
#         except OSError as e:
#             print(f"[WARN] Parallel failed with {e}. Falling back to serial search...")
#             out = []
#             for pd_ in tqdm(all_param_dicts, desc=f"Param search (serial, {base_model})"):
#                 out.append(evaluate_param_set(pd_))
#
#     val_rows = [r for sub in out for r in sub]
#     if not val_rows: raise RuntimeError("No validation rows produced (check folds and input data).")
#     val_df = pd.DataFrame(val_rows)
#
#     # ---------------- validation aggregate ----------------
#     if base_model == "xgb":
#         param_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']
#     else:
#         param_keys = ['hidden_layer_sizes','alpha','learning_rate_init','batch_size','max_iter']
#
#     group_cols = param_keys + (['threshold'] if _IS_CLASSIFY else ['edge_param'])
#
#     agg_dict = {
#         'n_preds_val': 'sum',
#         'tp_val': 'sum',
#         'fp_val': 'sum',
#         'val_precision': 'mean',
#         'val_accuracy': 'mean',
#         'val_auc': 'mean',
#         'val_logloss': 'mean',
#         'val_brier': 'mean',
#         'n_value_bets_val': 'sum',
#     }
#     if 'val_edge_ratio_mean' in val_df.columns: agg_dict['val_edge_ratio_mean'] = 'mean'
#     if 'val_edge_ratio_mean_back' in val_df.columns: agg_dict['val_edge_ratio_mean_back'] = 'mean'
#
#     agg = val_df.groupby(group_cols, as_index=False).agg(agg_dict)
#     agg['val_precision_pooled'] = agg.apply(lambda r: (r['tp_val'] / max(1, (r['tp_val'] + r['fp_val']))), axis=1)
#     agg['val_precision_lcb'] = agg.apply(lambda r: _wilson_lcb(int(r['tp_val']), int(r['fp_val']), conf=val_conf_level), axis=1)
#
#     qual_mask = (
#         (agg['val_precision'] >= float(precision_test_threshold)) &
#         (agg['n_preds_val'] >= int(min_samples))
#     )
#     if _USE_VALUE_LAY:  qual_mask &= (agg['n_value_bets_val'] >= int(min_samples))
#     if _USE_VALUE_BACK: qual_mask &= (agg['n_value_bets_val'] >= int(min_samples))
#     qual = agg[qual_mask].copy()
#
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     if qual.empty:
#         fail_csv = None
#         if save_diagnostics_on_fail:
#             diag = (agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                                     ascending=[False, False, False, False])
#                     .assign(fail_reason="failed_validation_gate", market=market))
#             fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv")
#             os.makedirs(csv_save_dir, exist_ok=True); diag.to_csv(fail_csv, index=False)
#         msg = "No strategy met validation gates."
#         if on_fail == "raise": raise RuntimeError(msg)
#         if on_fail == "warn": print("[WARN]", msg)
#         return {'status':'failed_validation_gate','csv':fail_csv,'model_pkl':None,
#                 'summary_df':None,'validation_table':agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                 ascending=[False,False,False,False]).reset_index(drop=True)}
#
#     ranked = qual.sort_values(by=['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                               ascending=[False, False, False, False]).reset_index(drop=True)
#     topk_val = ranked.head(top_k).reset_index(drop=True)
#
#     def _extract_params_from_row(row):
#         return cast_params({k: row[k] for k in param_keys if k in row.index})
#
#     candidates = []
#     for _, row in topk_val.iterrows():
#         c = {
#             'params': _extract_params_from_row(row),
#             'val_precision': float(row['val_precision']),
#             'val_precision_lcb': float(row['val_precision_lcb']),
#             'val_accuracy': float(row['val_accuracy']),
#             'n_preds_val': int(row['n_preds_val']),
#         }
#         if _IS_CLASSIFY:
#             c['threshold'] = float(row['threshold'])
#         else:
#             c['edge_param'] = float(row['edge_param'])
#         candidates.append(c)
#
#     # ---------------- test eval ----------------
#     records_all = []
#     all_bets_collector = []  # across all candidates
#
#     def _name_cols(subdf):
#         cols = {}
#         for c in ['date','league','country','home_team','away_team','match_id']:
#             if c in subdf.columns: cols[c] = subdf[c].values
#         if {'home_team','away_team'}.issubset(subdf.columns):
#             cols['event_name'] = (subdf['home_team'] + ' v ' + subdf['away_team']).values
#         return cols
#
#     for cand_id, cand in enumerate(candidates):
#         best_params = cast_params(cand['params'])
#         pos = int(y_train_final.sum()); neg = len(y_train_final) - pos
#         spw_final = (neg/pos) if pos > 0 else 1.0
#
#         final_model = build_model(best_params, spw_final)
#         final_sample_weight = None
#         if base_model == "mlp":
#             w_pos = spw_final
#             final_sample_weight = np.where(y_train_final.values==1, w_pos, 1.0).astype(np.float32)
#
#         fit_model(final_model, X_train_final, y_train_final, X_val_final, y_val_final, sample_weight=final_sample_weight)
#         final_calibrator = fit_calibrator(final_model, X_val_final, y_val_final)
#         p_pos_test = predict_proba_pos(final_calibrator, X_test)
#
#         if _USE_VALUE_LAY:
#             if market == "LAY_AWAY":
#                 p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['away_odds'].values; sel_name = 'AWAY'
#             elif market == "LAY_HOME":
#                 p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['home_odds'].values; sel_name = 'HOME'
#             elif market == "LAY_DRAW":
#                 p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['draw_odds'].values; sel_name = 'DRAW'
#             else:
#                 raise RuntimeError("Internal: LAY mode only for LAY_*")
#
#             fair_odds = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
#             valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01)
#             edge = float(cand.get('edge_param', 0.0))
#             edge_mask = valid & (fair_odds >= (1.0 + edge) * mkt_odds)
#
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 edge_ratio_minus1 = np.where(mkt_odds > 0, fair_odds / mkt_odds - 1.0, 0.0)
#
#             for plan in staking_plan_lay_options:
#                 stake, liability = _lay_stakes(mkt_odds, edge_ratio_minus1, plan)
#                 stake = np.where(edge_mask, stake, 0.0)
#                 liability = np.where(edge_mask, liability, 0.0)
#
#                 sel_wins = (y_test.values == 0)
#                 pl = np.zeros_like(stake)
#                 # winning lay (selection loses): profit after commission
#                 idx_win = (stake > 0) & (~sel_wins)
#                 pl[idx_win] = stake[idx_win] * (1.0 - commission_rate)
#                 # losing lay (selection wins): pay liability
#                 idx_lose = (stake > 0) & (sel_wins)
#                 pl[idx_lose] = -liability[idx_lose]
#
#                 n_bets = int(np.count_nonzero(stake > 0))
#                 total_pl = float(pl.sum()); avg_pl = float(total_pl / max(1, n_bets))
#
#                 lays_as_preds = (stake > 0).astype(np.uint8)
#                 prc_test = precision_score(y_test, lays_as_preds, zero_division=0)
#                 acc_test = accuracy_score(y_test, lays_as_preds)
#
#                 bet_idx = np.where(stake > 0)[0]
#                 name_cols = _name_cols(df_test.iloc[bet_idx])
#                 bets_df = pd.DataFrame({
#                     **name_cols,
#                     'selection': sel_name,
#                     'market_odds': mkt_odds[bet_idx],
#                     'fair_odds': fair_odds[bet_idx],
#                     'edge_ratio': np.where(mkt_odds[bet_idx] > 0, fair_odds[bet_idx] / mkt_odds[bet_idx], np.nan),
#                     'liability': liability[bet_idx],
#                     'stake': stake[bet_idx],
#                     'commission_rate': float(commission_rate),
#                     'selection_won': sel_wins[bet_idx].astype(int),
#                     'target': y_test.values[bet_idx],
#                     'pl': pl[bet_idx],
#                 })
#                 if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
#                 bets_df['cum_pl'] = bets_df['pl'].cumsum()
#
#                 pv = _pvalue_break_even(bets_df, mode='VALUE_LAY')
#
#                 enough = n_bets >= int(min_test_samples)
#                 not_collapsed = prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop))
#                 pass_gate = bool(enough and not_collapsed)
#                 reason = "" if pass_gate else ("insufficient_test_bets" if not enough else "precision_collapse")
#
#                 if len(bets_df):
#                     meta = {
#                         'candidate_id': cand_id,
#                         'passed_test_gate': bool(pass_gate),
#                         'mode': 'VALUE_LAY',
#                         'market': market,
#                         'threshold': np.nan,
#                         'edge_param': edge,
#                         'staking_plan_lay': plan,
#                         'val_precision': float(cand['val_precision']),
#                         'val_precision_lcb': float(cand['val_precision_lcb']),
#                         'n_value_bets_test': int(n_bets),
#                         'total_pl': float(total_pl),
#                         'avg_pl': float(avg_pl),
#                         'p_value': pv['p_value'],
#                         'zscore': pv['z'],
#                         'commission_rate': float(commission_rate),
#                         'params_json': json.dumps(best_params, default=float),
#                     }
#                     bdf = bets_df.copy()
#                     for k, v in meta.items(): bdf[k] = v
#                     all_bets_collector.append(bdf)
#
#                 records_all.append({
#                     **best_params, 'threshold': np.nan, 'edge_param': edge,
#                     'val_precision_lcb': cand['val_precision_lcb'], 'val_precision': cand['val_precision'],
#                     'val_accuracy': cand['val_accuracy'],
#                     'n_value_bets_test': n_bets, 'test_precision_bets': float(prc_test),
#                     'test_accuracy_bets': float(acc_test), 'total_pl': total_pl, 'avg_pl': avg_pl,
#                     'p_value': pv['p_value'], 'zscore': pv['z'],
#                     'pass_test_gate': pass_gate, 'fail_reason': reason, 'model_obj': final_calibrator if pass_gate else None,
#                     'mode': 'VALUE_LAY', 'bets': bets_df if pass_gate else None,
#                     'staking_plan_lay': plan,
#                     'commission_rate': float(commission_rate),
#                 })
#
#         elif _USE_VALUE_BACK:
#             if market == "BACK_AWAY":
#                 p_sel_win = p_pos_test; mkt_odds = df_test['away_odds'].values; sel_name = 'AWAY'
#             elif market == "BACK_HOME":
#                 p_sel_win = p_pos_test; mkt_odds = df_test['home_odds'].values; sel_name = 'HOME'
#             elif market == "BACK_DRAW":
#                 p_sel_win = p_pos_test; mkt_odds = df_test['draw_odds'].values; sel_name = 'DRAW'
#             else:
#                 raise RuntimeError("Internal: BACK mode only for BACK_*")
#
#             fair_odds = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
#             valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01)
#             edge = float(cand.get('edge_param', 0.0))
#             edge_mask = valid & (mkt_odds >= (1.0 + edge) * fair_odds)
#
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 edge_ratio_minus1 = np.where(fair_odds > 0, mkt_odds / fair_odds - 1.0, 0.0)
#
#             for plan in staking_plan_back_options:
#                 stake = _back_stakes(mkt_odds, edge_ratio_minus1, plan, p_sel_win)
#                 stake = np.where(edge_mask, stake, 0.0)
#
#                 sel_wins = (y_test.values == 1)
#                 pl = np.zeros_like(stake)
#                 win_idx = (stake > 0) & sel_wins
#                 lose_idx = (stake > 0) & (~sel_wins)
#                 pl[win_idx]  = (mkt_odds[win_idx] - 1.0) * stake[win_idx] * (1.0 - commission_rate)
#                 pl[lose_idx] = -stake[lose_idx]
#
#                 n_bets = int(np.count_nonzero(stake > 0))
#                 total_pl = float(pl.sum()); avg_pl = float(total_pl / max(1, n_bets))
#
#                 backs_as_preds = (stake > 0).astype(np.uint8)
#                 prc_test = precision_score(y_test, backs_as_preds, zero_division=0)
#                 acc_test = accuracy_score(y_test, backs_as_preds)
#
#                 bet_idx = np.where(stake > 0)[0]
#                 name_cols = _name_cols(df_test.iloc[bet_idx])
#                 bets_df = pd.DataFrame({
#                     **name_cols,
#                     'selection': sel_name,
#                     'market_odds': mkt_odds[bet_idx],
#                     'fair_odds': fair_odds[bet_idx],
#                     'edge_ratio': np.where(fair_odds[bet_idx] > 0, mkt_odds[bet_idx] / fair_odds[bet_idx], np.nan),
#                     'stake': stake[bet_idx],
#                     'commission_rate': float(commission_rate),
#                     'selection_won': sel_wins[bet_idx].astype(int),
#                     'target': y_test.values[bet_idx],
#                     'pl': pl[bet_idx],
#                 })
#                 if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
#                 bets_df['cum_pl'] = bets_df['pl'].cumsum()
#
#                 pv = _pvalue_break_even(bets_df, mode='VALUE_BACK')
#
#                 enough = n_bets >= int(min_test_samples)
#                 not_collapsed = prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop))
#                 pass_gate = bool(enough and not_collapsed)
#                 reason = "" if pass_gate else ("insufficient_test_bets" if not enough else "precision_collapse")
#
#                 if len(bets_df):
#                     meta = {
#                         'candidate_id': cand_id,
#                         'passed_test_gate': bool(pass_gate),
#                         'mode': 'VALUE_BACK',
#                         'market': market,
#                         'threshold': np.nan,
#                         'edge_param': edge,
#                         'staking_plan_back': plan,
#                         'val_precision': float(cand['val_precision']),
#                         'val_precision_lcb': float(cand['val_precision_lcb']),
#                         'n_value_bets_test': int(n_bets),
#                         'total_pl': float(total_pl),
#                         'avg_pl': float(avg_pl),
#                         'p_value': pv['p_value'],
#                         'zscore': pv['z'],
#                         'commission_rate': float(commission_rate),
#                         'params_json': json.dumps(best_params, default=float),
#                     }
#                     bdf = bets_df.copy()
#                     for k, v in meta.items(): bdf[k] = v
#                     all_bets_collector.append(bdf)
#
#                 records_all.append({
#                     **best_params, 'threshold': np.nan, 'edge_param': edge,
#                     'val_precision_lcb': cand['val_precision_lcb'], 'val_precision': cand['val_precision'],
#                     'val_accuracy': cand['val_accuracy'],
#                     'n_value_bets_test': n_bets, 'test_precision_bets': float(prc_test),
#                     'test_accuracy_bets': float(acc_test), 'total_pl': total_pl, 'avg_pl': avg_pl,
#                     'p_value': pv['p_value'], 'zscore': pv['z'],
#                     'pass_test_gate': pass_gate, 'fail_reason': reason, 'model_obj': final_calibrator if pass_gate else None,
#                     'mode': 'VALUE_BACK', 'bets': bets_df if pass_gate else None,
#                     'staking_plan_back': plan,
#                     'commission_rate': float(commission_rate),
#                 })
#
#         else:
#             # CLASSIFY markets: threshold sweep chosen in validation
#             thr = float(cand['threshold'])
#             y_pred = (p_pos_test >= thr).astype(np.uint8)
#             n_preds_test = int(y_pred.sum())
#             prc_test = precision_score(y_test, y_pred, zero_division=0)
#             acc_test = accuracy_score(y_test, y_pred)
#             enough = n_preds_test >= int(min_test_samples)
#             not_collapsed = prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop))
#             pass_gate = bool(enough and not_collapsed)
#             reason = "" if pass_gate else ("insufficient_test_preds" if not enough else "precision_collapse")
#             records_all.append({
#                 **best_params, 'threshold': thr, 'edge_param': np.nan,
#                 'val_precision_lcb': cand['val_precision_lcb'], 'val_precision': cand['val_precision'],
#                 'val_accuracy': cand['val_accuracy'],
#                 'n_preds_test': n_preds_test, 'test_precision': float(prc_test), 'test_accuracy': float(acc_test),
#                 'pass_test_gate': pass_gate, 'fail_reason': reason, 'model_obj': final_calibrator if pass_gate else None,
#                 'mode': 'CLASSIFY', 'bets': None,
#             })
#
#     survivors_df = pd.DataFrame(records_all)
#     passers = survivors_df[survivors_df['pass_test_gate']].copy()
#
#     # ---------------- save / rank ----------------
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     tag = "xgb" if base_model == "xgb" else "mlp"
#
#     if passers.empty:
#         fail_csv = None
#         if save_diagnostics_on_fail:
#             sort_cols = (['p_value','total_pl','val_precision_lcb'] if _USE_VALUE
#                          else ['val_precision_lcb','val_precision','n_preds_test','val_accuracy'])
#             asc = ([True, False, False] if _USE_VALUE else [False, False, False, False])
#             diag = (survivors_df
#                     .drop(columns=['model_obj','bets'], errors='ignore')
#                     .sort_values(by=sort_cols, ascending=asc)
#                     .assign(market=market))
#             fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv")
#             diag.to_csv(fail_csv, index=False); summary_df = diag
#         else:
#             summary_df = survivors_df.drop(columns=['model_obj','bets'], errors='ignore')
#
#         all_bets_csv_path = None
#         if save_all_bets_csv and _USE_VALUE and all_bets_collector:
#             all_bets_df = pd.concat(all_bets_collector, ignore_index=True)
#             if not all_bets_include_failed:
#                 all_bets_df = all_bets_df[all_bets_df['passed_test_gate'] == True]
#             all_bets_csv_path = os.path.join(all_bets_dir, f"all_bets_{market}_{timestamp}.csv")
#             all_bets_df.to_csv(all_bets_csv_path, index=False)
#
#         msg = "All Top-K failed the TEST gate."
#         if on_fail == "raise": raise RuntimeError(msg)
#         if on_fail == "warn": print("[WARN]", msg)
#         return {'status':'failed_test_gate','csv':fail_csv,'model_pkl':None,
#                 'summary_df':summary_df,'validation_table':ranked,
#                 'bets_csv':None,'pl_plot':None,'all_bets_csv':all_bets_csv_path}
#
#     # Final ranking
#     if _USE_VALUE:
#         passers_sorted = passers.sort_values(
#             by=['p_value','total_pl','avg_pl','val_precision_lcb','val_precision'],
#             ascending=[True, False, False, False, False]
#         ).reset_index(drop=True)
#     else:
#         passers_sorted = passers.sort_values(
#             by=['val_precision_lcb','val_precision','test_precision','n_preds_test','val_accuracy'],
#             ascending=[False, False, False, False, False]
#         ).reset_index(drop=True)
#
#     # Save PKL + CSV
#     pkl_path = os.path.join(model_dir, f"best_model_{market}_{tag}_calibrated_{timestamp}.pkl")
#     csv_df = passers_sorted.drop(columns=['model_obj','bets'], errors='ignore').copy()
#     csv_df['model_pkl'] = ""; csv_df.loc[0, 'model_pkl'] = pkl_path
#     csv_df['market'] = market
#     csv_path = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}.csv")
#     csv_df.to_csv(csv_path, index=False)
#
#     # Save top model (with metadata to reproduce staking during inference)
#     top_row = passers_sorted.iloc[0]
#     chosen_model = top_row['model_obj']
#     if base_model == "xgb":
#         param_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']
#     else:
#         param_keys = ['hidden_layer_sizes','alpha','learning_rate_init','batch_size','max_iter']
#     chosen_params = {k: top_row[k] for k in param_keys if k in passers_sorted.columns}
#     chosen_threshold = float(top_row.get('threshold', np.nan))
#     chosen_edge = float(top_row.get('edge_param', np.nan))
#
#     joblib.dump(
#         {
#             'model': chosen_model,
#             'threshold': chosen_threshold,            # NaN in VALUE modes; meaningful in CLASSIFY
#             'edge_param': chosen_edge,                # chosen edge (VALUE)
#             'features': features,
#             'base_model': base_model,
#             'best_params': chosen_params,
#             'precision_test_threshold': float(precision_test_threshold),
#             'min_samples': int(min_samples),
#             'min_test_samples': int(min_test_samples),
#             'val_conf_level': float(val_conf_level),
#             'max_precision_drop': float(max_precision_drop),
#             'market': market,
#             'mode': top_row['mode'],
#             # staking plan that WON:
#             'staking_plan_lay': top_row.get('staking_plan_lay', None) if _USE_VALUE_LAY else None,
#             'staking_plan_back': top_row.get('staking_plan_back', None) if _USE_VALUE_BACK else None,
#             # numeric staking params (for reproducibility)
#             'liability_test': float(liability_test) if _USE_VALUE_LAY else None,
#             'lay_flat_stake': float(lay_flat_stake) if _USE_VALUE_LAY else None,
#             'lay_edge_scale': float(lay_edge_scale) if _USE_VALUE_LAY else None,
#             'kelly_fraction_lay': float(kelly_fraction_lay) if _USE_VALUE_LAY else None,
#             'min_lay_stake': float(min_lay_stake) if _USE_VALUE_LAY else None,
#             'max_lay_stake': float(max_lay_stake) if _USE_VALUE_LAY else None,
#             'min_lay_liability': float(min_lay_liability) if _USE_VALUE_LAY else None,
#             'max_lay_liability': float(max_lay_liability) if _USE_VALUE_LAY else None,
#             'back_stake_test': float(back_stake_test) if _USE_VALUE_BACK else None,
#             'back_edge_scale': float(back_edge_scale) if _USE_VALUE_BACK else None,
#             'kelly_fraction_back': float(kelly_fraction_back) if _USE_VALUE_BACK else None,
#             'bankroll_back': float(bankroll_back) if _USE_VALUE_BACK else None,
#             'min_back_stake': float(min_back_stake) if _USE_VALUE_BACK else None,
#             'max_back_stake': float(max_back_stake) if _USE_VALUE_BACK else None,
#             # commission saved for inference
#             'commission_rate': float(commission_rate),
#             'notes': ('Commission applied to winning returns in VALUE modes; '
#                       'p-value uses commission-adjusted pay-offs; '
#                       'edge sweep + optional staking-plan search; '
#                       'per-bet stakes/liabilities in bets CSVs.'),
#             'run_seed': int(RUN_SEED),
#         },
#         pkl_path
#     )
#
#     # chosen bets CSV / plot
#     bets_path = None
#     plot_path = None
#     if _USE_VALUE and (save_bets_csv or plot_pl):
#         top_bets_df = survivors_df.loc[passers_sorted.index[0], 'bets'] if 'bets' in survivors_df.columns else None
#         if isinstance(top_bets_df, pd.DataFrame) and len(top_bets_df):
#             if save_bets_csv:
#                 bets_name = f"bets_{market}_{timestamp}.csv"
#                 bets_path = os.path.join(bets_csv_dir, bets_name)
#                 top_bets_df.to_csv(bets_path, index=False)
#             if plot_pl:
#                 try:
#                     import matplotlib.pyplot as plt
#                     fig = plt.figure()
#                     x = top_bets_df['date'] if 'date' in top_bets_df.columns else np.arange(len(top_bets_df))
#                     plt.plot(x, top_bets_df['cum_pl'])
#                     title = f"{market} cumulative P/L ({'VALUE_LAY' if _USE_VALUE_LAY else 'VALUE_BACK'})"
#                     if plot_title_suffix: title += f" — {plot_title_suffix}"
#                     plt.title(title); plt.xlabel('Date' if 'date' in top_bets_df.columns else 'Bet #'); plt.ylabel('Cumulative P/L')
#                     plt.tight_layout()
#                     plot_name = f"cum_pl_{market}_{timestamp}.png"
#                     plot_path = os.path.join(plot_dir, plot_name)
#                     plt.savefig(plot_path, dpi=160); plt.close(fig)
#                 except Exception as e:
#                     print(f"[WARN] Failed to create plot: {e}")
#
#     # ALL bets export (across all candidates and staking plans)
#     all_bets_csv_path = None
#     if save_all_bets_csv and _USE_VALUE and all_bets_collector:
#         all_bets_df = pd.concat(all_bets_collector, ignore_index=True)
#         if not all_bets_include_failed:
#             all_bets_df = all_bets_df[all_bets_df['passed_test_gate'] == True]
#         preferred = [c for c in [
#             'date','league','country','home_team','away_team','match_id','event_name','selection',
#             'market_odds','fair_odds','edge_ratio','stake','liability','commission_rate',
#             'selection_won','target','pl','cum_pl',
#             'candidate_id','passed_test_gate','mode','market','threshold','edge_param',
#             'staking_plan_lay','staking_plan_back',
#             'val_precision','val_precision_lcb','n_value_bets_test','total_pl','avg_pl','p_value','zscore','params_json'
#         ] if c in all_bets_df.columns]
#         all_bets_df = all_bets_df[preferred + [c for c in all_bets_df.columns if c not in preferred]]
#         all_bets_csv_path = os.path.join(all_bets_dir, f"all_bets_{market}_{timestamp}.csv")
#         all_bets_df.to_csv(all_bets_csv_path, index=False)
#
#     return {
#         'status': 'ok',
#         'csv': csv_path,
#         'model_pkl': pkl_path,
#         'summary_df': csv_df,
#         'validation_table': ranked,
#         'bets_csv': bets_path,
#         'pl_plot': plot_path,
#         'all_bets_csv': all_bets_csv_path,
#     }


# def run_models_outcome(
#     matches_filtered: pd.DataFrame,
#     features: list,
#     # ── gates ──────────────────────────────────────────────────────────────
#     min_samples: int = 200,
#     min_test_samples: int = 100,
#     precision_test_threshold: float = 0.80,
#     # ── model/search ───────────────────────────────────────────────────────
#     base_model: str = "xgb",
#     search_mode: str = "random",
#     n_random_param_sets: int = 10,
#     cpu_jobs: int = 6,
#     top_k: int = 10,
#     thresholds: np.ndarray | None = None,        # USED only for CLASSIFY markets
#     out_dir: str | None = None,
#     # ── anti-overfitting ──────────────────────────────────────────────────
#     val_conf_level: float = 0.99,
#     max_precision_drop: float = 0.05,
#     # ── failure handling ───────────────────────────────────────────────────
#     on_fail: str = "return",                     # "return" | "warn" | "raise"
#     save_diagnostics_on_fail: bool = True,
#     # ── market ─────────────────────────────────────────────────────────────
#     market: str = "LAY_AWAY",                    # LAY_* | BACK_* | OVER | UNDER (or other classify markets)
#
#     # ── VALUE LAY controls ────────────────────────────────────────────────
#     use_value_for_lay: bool = True,
#     value_edge_grid_lay: np.ndarray | None = None,   # e.g. np.round(np.arange(0.00,0.201,0.01),2)
#
#     # Staking plan search toggle + options (VALUE modes)
#     enable_staking_plan_search: bool = True,
#     staking_plan_lay_options: list[str] | None = None,
#     staking_plan_back_options: list[str] | None = None,
#
#     # Single-plan (used when enable_staking_plan_search=False)
#     staking_plan_lay: str = "liability",             # "liability" | "flat_stake" | "edge_prop" | "kelly_approx"
#     staking_plan_back: str = "flat",                 # "flat" | "edge_prop" | "kelly"
#
#     # ── LAY staking parameters (balanced defaults) ────────────────────────
#     liability_test: float = 1.0,
#     lay_flat_stake: float = 0.50,
#     lay_edge_scale: float = 0.05,
#     kelly_fraction_lay: float = 1.0,
#     min_lay_stake: float = 0.0,
#     max_lay_stake: float = 1.0,
#     min_lay_liability: float = 0.0,
#     max_lay_liability: float = 2.0,
#
#     # ── VALUE BACK controls ────────────────────────────────────────────────
#     use_value_for_back: bool = False,
#     value_edge_grid_back: np.ndarray | None = None,
#
#     # BACK staking parameters
#     back_stake_test: float = 1.0,
#     back_edge_scale: float = 0.10,
#     kelly_fraction_back: float = 0.25,
#     bankroll_back: float = 100.0,
#     min_back_stake: float = 0.0,
#     max_back_stake: float = 10.0,
#
#     # ── CLASSIFY staking / odds (for p-value & monetary P/L) ──────────────
#     classify_stake: float = 1.0,                 # flat stake for each positive (p>=threshold)
#     classify_odds_column: str | None = None,     # e.g. 'away_odds', 'over25_odds'; if None → pseudo P/L, no p-value
#
#     # ── COMMISSION (applied to net winning returns) ───────────────────────
#     commission_rate: float = 0.02,  # 2% commission on winnings
#
#     # ── OUTPUTS: chosen model ─────────────────────────────────────────────
#     save_bets_csv: bool = False,
#     bets_csv_dir: str | None = None,
#     plot_pl: bool = False,
#     plot_dir: str | None = None,
#     plot_title_suffix: str = "",
#     # ── OUTPUTS: ALL candidates ───────────────────────────────────────────
#     save_all_bets_csv: bool = False,
#     all_bets_dir: str | None = None,
#     all_bets_include_failed: bool = True,
# ):
#     """
#     Rolling time-ordered CV with calibration.
#
#     VALUE LAY (market starts with LAY_):
#       fair_odds = 1 / P(selection wins). Place a lay if fair ≥ market × (1+edge).
#       Commission on winning lay: profit = stake * (1 - commission_rate).
#       Loss if selection wins: −liability.
#
#     VALUE BACK (market starts with BACK_):
#       Place a back if market ≥ fair × (1+edge).
#       Commission on winning back: profit = (odds−1)*stake * (1 - commission_rate).
#       Loss if selection loses: −stake.
#
#     CLASSIFY (e.g., OVER/UNDER):
#       Sweep probability thresholds (argument `thresholds`) during validation and
#       carry the best threshold into test. Build bet history for positives (p≥thr).
#       If `classify_odds_column` is given, compute monetary P/L with commission and
#       p-value vs break-even; otherwise pseudo P/L (diagnostic), p-value = NaN.
#     """
#     # ---------------- setup ----------------
#     import os, secrets, hashlib, json
#     from datetime import datetime
#     import numpy as np
#     import pandas as pd
#     from itertools import product
#     from sklearn.model_selection import ParameterSampler
#     from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, log_loss, brier_score_loss
#     from sklearn.calibration import CalibratedClassifierCV
#     from sklearn.pipeline import Pipeline, make_pipeline
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.neural_network import MLPClassifier
#     from joblib import Parallel, delayed, parallel_backend
#     from tqdm import tqdm
#     from tqdm_joblib import tqdm_joblib
#     import joblib
#
#     # --- xgboost import (optional)
#     try:
#         import xgboost as xgb
#         _HAS_XGB_LOCAL = True
#     except Exception:
#         _HAS_XGB_LOCAL = False
#     _HAS_XGB = globals().get("_HAS_XGB", _HAS_XGB_LOCAL)
#     if base_model == "xgb" and not _HAS_XGB:
#         raise ImportError("XGBoost not available; set base_model='mlp' or install xgboost.")
#
#     # --- random dists
#     try:
#         _randint; _uniform; _loguniform
#     except NameError:
#         from scipy.stats import randint as _randint
#         from scipy.stats import uniform as _uniform
#         from scipy.stats import loguniform as _loguniform
#
#     # --- Wilson LCB & normal CDF
#     try:
#         from scipy.stats import norm
#         _Z = lambda conf: float(norm.ppf(1 - (1 - conf) / 2))
#         _Phi = lambda z: float(norm.cdf(z))
#     except Exception:
#         import math
#         _Z = lambda conf: 1.96 if abs(conf - 0.95) < 1e-6 else 1.64
#         _Phi = lambda z: 0.5 * (1.0 + math.erf(z / (2**0.5)))
#
#     def _wilson_lcb(tp: int, fp: int, conf: float) -> float:
#         n = tp + fp
#         if n <= 0: return 0.0
#         p = tp / n
#         z = _Z(conf)
#         denom = 1.0 + (z*z)/n
#         centre = p + (z*z)/(2*n)
#         rad = z * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
#         return max(0.0, (centre - rad) / denom)
#
#     # defaults
#     if thresholds is None:
#         thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)  # CLASSIFY only
#     if value_edge_grid_lay is None:
#         value_edge_grid_lay = np.round(np.arange(0.00, 0.201, 0.01), 2)
#     if value_edge_grid_back is None:
#         value_edge_grid_back = np.round(np.arange(0.00, 0.201, 0.01), 2)
#
#     # normalise staking-plan options
#     if staking_plan_lay_options is None:
#         staking_plan_lay_options = ["liability", "flat_stake", "edge_prop", "kelly_approx"]
#     if staking_plan_back_options is None:
#         staking_plan_back_options = ["flat", "edge_prop", "kelly"]
#     if not enable_staking_plan_search:
#         staking_plan_lay_options = [staking_plan_lay]
#         staking_plan_back_options = [staking_plan_back]
#
#     # --- paths
#     BASE = r"C:\Users\leere\PycharmProjects\Football_ML3\FT Results"
#     PKL_DIRS = {
#         "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "model_file"),
#         "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "model_file"),
#         "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "model_file"),
#         "BACK_HOME": os.path.join(BASE, "Back_Home", "model_file"),
#         "BACK_AWAY": os.path.join(BASE, "Back_Away", "model_file"),
#         "BACK_DRAW": os.path.join(BASE, "Back_Draw", "model_file"),
#         "OVER":      os.path.join(BASE, "Over_2_5",  "model_file"),
#         "UNDER":     os.path.join(BASE, "Under_2_5", "model_file"),
#     }
#     CSV_DIRS = {
#         "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "best_model_metrics"),
#         "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "best_model_metrics"),
#         "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "best_model_metrics"),
#         "BACK_HOME": os.path.join(BASE, "Back_Home", "best_model_metrics"),
#         "BACK_AWAY": os.path.join(BASE, "Back_Away", "best_model_metrics"),
#         "BACK_DRAW": os.path.join(BASE, "Back_Draw", "best_model_metrics"),
#         "OVER":      os.path.join(BASE, "Over_2_5",  "best_model_metrics"),
#         "UNDER":     os.path.join(BASE, "Under_2_5", "best_model_metrics"),
#     }
#
#     market = str(market).upper().strip()
#     if market not in PKL_DIRS: raise ValueError(f"Unsupported market '{market}'.")
#     _IS_LAY  = market.startswith("LAY_")
#     _IS_BACK = market.startswith("BACK_")
#     _USE_VALUE_LAY  = bool(use_value_for_lay and _IS_LAY)
#     _USE_VALUE_BACK = bool(use_value_for_back and _IS_BACK)
#     _USE_VALUE = _USE_VALUE_LAY or _USE_VALUE_BACK
#     _IS_CLASSIFY = not _USE_VALUE
#
#     csv_save_dir = out_dir if (out_dir and len(str(out_dir)) > 0) else CSV_DIRS[market]
#     os.makedirs(csv_save_dir, exist_ok=True)
#     model_dir = PKL_DIRS[market]; os.makedirs(model_dir, exist_ok=True)
#     if bets_csv_dir is None: bets_csv_dir = csv_save_dir
#     if plot_dir is None: plot_dir = csv_save_dir
#     os.makedirs(bets_csv_dir, exist_ok=True)
#     os.makedirs(plot_dir, exist_ok=True)
#     if all_bets_dir is None:
#         # IMPORTANT: keep "all_bets" OUTSIDE the best_model_metrics dir
#         all_bets_dir = os.path.join(os.path.dirname(CSV_DIRS[market]), "all_bets")
#     os.makedirs(all_bets_dir, exist_ok=True)
#
#     RUN_SEED = secrets.randbits(32)
#     def _seed_from(*vals) -> int:
#         h = hashlib.blake2b(digest_size=8)
#         h.update(int(RUN_SEED).to_bytes(8,'little',signed=False))
#         for v in vals: h.update(str(v).encode('utf-8'))
#         return int.from_bytes(h.digest(),'little') & 0x7FFFFFFF
#
#     def _as_float(x):
#         try: return float(x)
#         except Exception: return float(str(x))
#     def _as_int(x): return int(float(x))
#
#     # ---------------- data ----------------
#     req_cols = {'date','target'}
#     if _USE_VALUE: req_cols |= {'home_odds','draw_odds','away_odds'}
#     # CLASSIFY odds are optional; required only if you want p-value in CLASSIFY
#     missing = req_cols - set(matches_filtered.columns)
#     if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
#
#     df = matches_filtered.copy()
#     df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     df = df.sort_values('date').reset_index(drop=True)
#
#     cols_needed = list(set(features) | {'target'} | ({'home_odds','draw_odds','away_odds'} if _USE_VALUE else set()))
#     if _IS_CLASSIFY and classify_odds_column is not None:
#         cols_needed = list(set(cols_needed) | {classify_odds_column})
#     df = df.dropna(subset=cols_needed).reset_index(drop=True)
#
#     X = df[features].copy()
#     y = df['target'].astype(int).reset_index(drop=True)
#
#     n = len(X)
#     if n < max(min_samples * 3, 500): raise RuntimeError(f"Not enough rows: {n}")
#
#     # temporal split
#     test_start = int(0.85 * n)
#     pretest_end = test_start
#     X_test = X.iloc[test_start:].reset_index(drop=True)
#     y_test = y.iloc[test_start:].reset_index(drop=True)
#     df_test = df.iloc[test_start:].reset_index(drop=True)
#
#     # rolling validation folds
#     N_FOLDS = 5
#     total_val_len = max(1, int(0.15 * n))
#     val_len = max(1, total_val_len // N_FOLDS)
#     fold_val_ends = [pretest_end - total_val_len + (i + 1) * val_len for i in range(N_FOLDS)]
#     fold_val_starts = [end - val_len for end in fold_val_ends]
#     if fold_val_ends:
#         fold_val_ends[-1] = min(fold_val_ends[-1], pretest_end)
#         fold_val_starts[-1] = max(0, fold_val_ends[-1] - val_len)
#
#     # final small validation slice (for calibration before test)
#     final_val_len = max(1, val_len)
#     final_val_start = max(0, test_start - final_val_len)
#     X_train_final = X.iloc[:final_val_start]
#     y_train_final = y.iloc[:final_val_start]
#     X_val_final   = X.iloc[final_val_start:test_start]
#     y_val_final   = y.iloc[final_val_start:test_start]
#
#     # ---------------- param spaces ----------------
#     xgb_param_grid = {'n_estimators':[200],'max_depth':[5],'learning_rate':[0.1],'subsample':[0.7],
#                       'colsample_bytree':[1.0],'min_child_weight':[5],'reg_lambda':[1.0]}
#     xgb_param_distributions = {'n_estimators':_randint(100,1001),'max_depth':_randint(3,8),
#                                'learning_rate':_loguniform(0.01,0.2),'min_child_weight':_randint(3,13),
#                                'subsample':_uniform(0.7,0.3),'colsample_bytree':_uniform(0.6,0.4),
#                                'reg_lambda':_loguniform(0.1,10.0)}
#     mlp_param_grid = {'hidden_layer_sizes':[(128,),(256,),(128,64)],'alpha':[1e-4],
#                       'learning_rate_init':[1e-3],'batch_size':['auto'],'max_iter':[200]}
#     mlp_param_distributions = {'hidden_layer_sizes':[(64,),(128,),(256,),(128,64),(256,128)],
#                                'alpha':_loguniform(1e-5,1e-2),'learning_rate_init':_loguniform(5e-4,5e-2),
#                                'batch_size':_randint(32,257),'max_iter':_randint(150,401)}
#
#     def cast_params(p: dict) -> dict:
#         q = dict(p)
#         if base_model == "xgb":
#             for k in ['n_estimators','max_depth','min_child_weight']:
#                 if k in q: q[k] = _as_int(q[k])
#             for k in ['learning_rate','subsample','colsample_bytree','reg_lambda']:
#                 if k in q: q[k] = _as_float(q[k])
#         else:
#             if 'max_iter' in q: q['max_iter'] = _as_int(q['max_iter'])
#             if 'batch_size' in q and q['batch_size'] != 'auto': q['batch_size'] = _as_int(q['batch_size'])
#             if 'alpha' in q: q['alpha'] = _as_float(q['alpha'])
#             if 'learning_rate_init' in q: q['learning_rate_init'] = _as_float(q['learning_rate_init'])
#             if 'hidden_layer_sizes' in q:
#                 h = q['hidden_layer_sizes']
#                 if isinstance(h, str):
#                     parts = [pp.strip() for pp in h.strip("()").split(",") if pp.strip()!='']
#                     q['hidden_layer_sizes'] = tuple(_as_int(pp) for pp in parts) if parts else (128,)
#                 elif isinstance(h, (list, tuple, np.ndarray)):
#                     q['hidden_layer_sizes'] = tuple(int(v) for v in h)
#                 else:
#                     q['hidden_layer_sizes'] = (int(h),)
#         return q
#
#     def _final_step_name(estimator):
#         try:
#             if isinstance(estimator, Pipeline): return estimator.steps[-1][0]
#         except Exception:
#             pass
#         return None
#
#     def build_model(params: dict, spw: float):
#         model_seed = _seed_from("model", base_model, tuple(sorted(params.items())))
#         if base_model == "xgb":
#             return xgb.XGBClassifier(
#                 objective='binary:logistic',
#                 eval_metric='auc',
#                 random_state=model_seed,
#                 scale_pos_weight=spw,
#                 n_jobs=1,
#                 tree_method="hist",
#                 verbosity=0,
#                 **params
#             )
#         else:
#             mlp = MLPClassifier(
#                 random_state=model_seed,
#                 early_stopping=True,
#                 n_iter_no_change=20,
#                 validation_fraction=0.1,
#                 solver="adam",
#                 **params
#             )
#             return make_pipeline(StandardScaler(with_mean=True, with_std=True), mlp)
#
#     def fit_model(model, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None):
#         if base_model == "xgb":
#             try:
#                 model.set_params(verbosity=0, early_stopping_rounds=50)
#                 if X_va is not None and y_va is not None:
#                     model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
#                 else:
#                     model.fit(X_tr, y_tr, verbose=False)
#             except Exception:
#                 model.fit(X_tr, y_tr, verbose=False)
#         else:
#             fit_kwargs = {}
#             if sample_weight is not None:
#                 stepname = _final_step_name(model)
#                 if stepname is not None:
#                     fit_kwargs[f"{stepname}__sample_weight"] = sample_weight
#             try:
#                 model.fit(X_tr, y_tr, **fit_kwargs)
#             except TypeError:
#                 model.fit(X_tr, y_tr)
#
#     def fit_calibrator(fitted, X_va, y_va):
#         try:
#             from sklearn.calibration import FrozenEstimator
#             frozen = FrozenEstimator(fitted)
#             cal = CalibratedClassifierCV(frozen, method='sigmoid', cv=None)
#             cal.fit(X_va, y_va)
#             return cal
#         except Exception:
#             try:
#                 cal = CalibratedClassifierCV(fitted, method='sigmoid', cv='prefit')
#                 cal.fit(X_va, y_va); return cal
#             except Exception:
#                 return fitted
#
#     def _unwrap_estimator(est):
#         if isinstance(est, Pipeline): return est.steps[-1][1]
#         return est
#
#     def predict_proba_pos(model_or_cal, X_):
#         proba = model_or_cal.predict_proba(X_)
#         if proba.ndim == 2:
#             classes = getattr(model_or_cal, "classes_", None)
#             if classes is None:
#                 base = _unwrap_estimator(model_or_cal); classes = getattr(base, "classes_", None)
#             if classes is not None and len(classes) == proba.shape[1]:
#                 try:
#                     idx = int(np.where(np.asarray(classes) == 1)[0][0])
#                     return proba[:, idx].astype(np.float32)
#                 except Exception:
#                     pass
#             if proba.shape[1] == 2: return proba[:, 1].astype(np.float32)
#             if proba.shape[1] == 1:
#                 only = getattr(model_or_cal, "classes_", [0])[0]
#                 return (np.ones_like(proba[:,0]) if only==1 else np.zeros_like(proba[:,0])).astype(np.float32)
#         return np.asarray(proba, dtype=np.float32)
#
#     # --- p-value helper (VALUE & CLASSIFY-back-like) -----------------------
#     def _pvalue_break_even(bdf: pd.DataFrame, mode: str) -> dict:
#         if not isinstance(bdf, pd.DataFrame) or bdf.empty:
#             return {'z': 0.0, 'p_value': 1.0, 'var_sum': 0.0, 'n': 0, 'total_pl': 0.0}
#         o = np.asarray(bdf['market_odds'].values, dtype=float)
#         o = np.where(o <= 1.0, np.nan, o)
#         p = 1.0 / o  # null prob (selection wins)
#         if mode == 'VALUE_BACK':
#             S = np.asarray(bdf['stake'].values, dtype=float)
#             win = (o - 1.0) * S * (1.0 - commission_rate)
#             lose = -S
#         else:  # VALUE_LAY
#             L = np.asarray(bdf['liability'].values, dtype=float)
#             stake = np.asarray(bdf['stake'].values, dtype=float)
#             win  = stake * (1.0 - commission_rate)   # selection loses
#             lose = -L                                 # selection wins
#         var_i = p * (win ** 2) + (1.0 - p) * (lose ** 2)
#         var_i = np.where(np.isfinite(var_i), var_i, 0.0)
#         pl = np.asarray(bdf['pl'].values, dtype=float)
#         total_pl = float(np.nansum(np.where(np.isfinite(pl), pl, 0.0)))
#         var_sum = float(np.nansum(var_i))
#         z = total_pl / (np.sqrt(var_sum) + 1e-12)
#         p_val = max(0.0, 1.0 - _Phi(z))  # one-sided
#         return {'z': float(z), 'p_value': float(p_val), 'var_sum': var_sum, 'n': int(len(pl)), 'total_pl': total_pl}
#
#     # --- staking calculators (VALUE) ---------------------------------------
#     def _lay_stakes(odds: np.ndarray, fair_over_market_minus1: np.ndarray, plan: str):
#         o = np.asarray(odds, dtype=float)
#         edge_plus = np.maximum(fair_over_market_minus1, 0.0)
#         if plan == "liability":
#             L = np.full_like(o, float(liability_test), dtype=float)
#             stake = np.divide(L, np.maximum(o - 1.0, 1e-9))
#         elif plan == "flat_stake":
#             stake = np.full_like(o, float(lay_flat_stake), dtype=float)
#             L = stake * (o - 1.0)
#         elif plan == "edge_prop":
#             L = float(liability_test) * np.divide(edge_plus, max(1e-9, float(lay_edge_scale)))
#             L = np.clip(L, float(min_lay_liability), float(max_lay_liability))
#             stake = np.divide(L, np.maximum(o - 1.0, 1e-9))
#         elif plan == "kelly_approx":
#             L = float(liability_test) * float(kelly_fraction_lay) * edge_plus
#             L = np.clip(L, float(min_lay_liability), float(max_lay_liability))
#             stake = np.divide(L, np.maximum(o - 1.0, 1e-9))
#         else:
#             raise ValueError(f"Unknown staking_plan_lay: {plan}")
#         stake = np.clip(stake, float(min_lay_stake), float(max_lay_stake))
#         L = np.clip(L, float(min_lay_liability), float(max_lay_liability))
#         return stake, L
#
#     def _back_stakes(odds: np.ndarray, fair_over_market_minus1: np.ndarray, plan: str, p_win: np.ndarray):
#         o = np.asarray(odds, dtype=float)
#         p = np.clip(np.asarray(p_win, dtype=float), 0.0, 1.0)
#         edge_plus = np.maximum(fair_over_market_minus1, 0.0)
#         if plan == "flat":
#             stake = np.full_like(o, float(back_stake_test), dtype=float)
#         elif plan == "edge_prop":
#             stake = float(back_stake_test) * np.divide(edge_plus, max(1e-9, float(back_edge_scale)))
#         elif plan == "kelly":
#             b = np.maximum(o - 1.0, 1e-9)
#             f = (b * p - (1.0 - p)) / b
#             f = np.maximum(f, 0.0)
#             stake = float(bankroll_back) * float(kelly_fraction_back) * f
#         else:
#             raise ValueError(f"Unknown staking_plan_back: {plan}")
#         stake = np.clip(stake, float(min_back_stake), float(max_back_stake))
#         return stake
#
#     # ---------------- search space ----------------
#     if search_mode.lower() == "grid":
#         grid = xgb_param_grid if base_model == "xgb" else mlp_param_grid
#         all_param_dicts = [dict(zip(grid.keys(), combo)) for combo in product(*grid.values())]
#     else:
#         dists = xgb_param_distributions if base_model == "xgb" else mlp_param_distributions
#         sampler_seed = _seed_from("sampler")
#         all_param_dicts = list(ParameterSampler(dists, n_iter=n_random_param_sets, random_state=sampler_seed))
#
#     # ---------------- validation eval ----------------
#     def evaluate_param_set(param_dict, *_):
#         safe = cast_params(param_dict)
#         rows = []; val_prob_all=[]; val_true_all=[]
#
#         for vstart, vend in zip(fold_val_starts, fold_val_ends):
#             if vstart is None or vend is None or vstart <= 0 or vend <= vstart: continue
#             X_tr, y_tr = X.iloc[:vstart], y.iloc[:vstart]
#             X_va, y_va = X.iloc[vstart:vend], y.iloc[vstart:vend]
#             df_va = df.iloc[vstart:vend]
#             if y_tr.nunique() < 2: continue
#
#             pos = int(y_tr.sum()); neg = len(y_tr) - pos
#             spw = (neg/pos) if pos > 0 else 1.0
#
#             sample_weight = None
#             if base_model == "mlp":
#                 w_pos = spw
#                 sample_weight = np.where(y_tr.values==1, w_pos, 1.0).astype(np.float32)
#
#             model = build_model(safe, spw)
#             fit_model(model, X_tr, y_tr, X_va, y_va, sample_weight=sample_weight)
#             cal = fit_calibrator(model, X_va, y_va)
#
#             p_pos = predict_proba_pos(cal, X_va)
#             val_prob_all.append(p_pos)
#             y_true = y_va.values.astype(np.uint8); val_true_all.append(y_true)
#
#             if _IS_CLASSIFY:
#                 for thr in thresholds:
#                     thr = float(thr)
#                     y_pred = (p_pos >= thr).astype(np.uint8)
#                     n_preds = int(y_pred.sum())
#                     tp = int(((y_true == 1) & (y_pred == 1)).sum())
#                     fp = int(((y_true == 0) & (y_pred == 1)).sum())
#                     prc = precision_score(y_va, y_pred, zero_division=0)
#                     acc = accuracy_score(y_va, y_pred)
#                     rows.append({
#                         **safe,
#                         'threshold': thr,
#                         'edge_param': np.nan,
#                         'fold_vstart': int(vstart),
#                         'fold_vend': int(vend),
#                         'n_preds_val': n_preds,
#                         'tp_val': tp,
#                         'fp_val': fp,
#                         'val_precision': float(prc),
#                         'val_accuracy': float(acc),
#                         'n_value_bets_val': 0,
#                         'val_edge_ratio_mean': np.nan,
#                         'val_edge_ratio_mean_back': np.nan,
#                     })
#             else:
#                 edge_grid = value_edge_grid_lay if _USE_VALUE_LAY else value_edge_grid_back
#                 for edge_param in edge_grid:
#                     if _USE_VALUE_LAY:
#                         if market == "LAY_AWAY":
#                             p_sel_win = 1.0 - p_pos; mkt = df_va['away_odds'].values
#                         elif market == "LAY_HOME":
#                             p_sel_win = 1.0 - p_pos; mkt = df_va['home_odds'].values
#                         else:
#                             p_sel_win = 1.0 - p_pos; mkt = df_va['draw_odds'].values
#                         fair = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
#                         edge_mask = (fair >= (1.0 + float(edge_param)) * mkt) & np.isfinite(mkt)
#                         with np.errstate(divide='ignore', invalid='ignore'):
#                             edge_ratio = fair / mkt
#                         val_edge_mean = float(np.nanmean(np.where(np.isfinite(edge_ratio), edge_ratio, np.nan)))
#                         n_bets = int(np.nansum(edge_mask))
#                     else:
#                         if market == "BACK_AWAY":
#                             p_sel_win = p_pos; mkt = df_va['away_odds'].values
#                         elif market == "BACK_HOME":
#                             p_sel_win = p_pos; mkt = df_va['home_odds'].values
#                         else:
#                             p_sel_win = p_pos; mkt = df_va['draw_odds'].values
#                         fair = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
#                         edge_mask = (mkt >= (1.0 + float(edge_param)) * fair) & np.isfinite(mkt)
#                         with np.errstate(divide='ignore', invalid='ignore'):
#                             edge_ratio = mkt / fair
#                         val_edge_mean = float(np.nanmean(np.where(np.isfinite(edge_ratio), edge_ratio, np.nan)))
#                         n_bets = int(np.nansum(edge_mask))
#
#                     y_pred = np.zeros_like(y_true, dtype=np.uint8)
#                     y_pred[np.where(edge_mask)[0]] = 1
#                     tp = int(((y_true == 1) & (y_pred == 1)).sum())
#                     fp = int(((y_true == 0) & (y_pred == 1)).sum())
#                     prc = precision_score(y_true, y_pred, zero_division=0)
#                     acc = accuracy_score(y_true, y_pred)
#
#                     rows.append({
#                         **safe,
#                         'threshold': np.nan,
#                         'edge_param': float(edge_param),
#                         'fold_vstart': int(vstart),
#                         'fold_vend': int(vend),
#                         'n_preds_val': int(y_pred.sum()),
#                         'tp_val': tp,
#                         'fp_val': fp,
#                         'val_precision': float(prc),
#                         'val_accuracy': float(acc),
#                         'n_value_bets_val': int(n_bets),
#                         'val_edge_ratio_mean': val_edge_mean if _USE_VALUE_LAY else np.nan,
#                         'val_edge_ratio_mean_back': val_edge_mean if _USE_VALUE_BACK else np.nan,
#                     })
#
#         # pooled diagnostics
#         if val_prob_all:
#             vp = np.concatenate(val_prob_all, axis=0)
#             vt = np.concatenate(val_true_all, axis=0)
#             try: val_auc = float(roc_auc_score(vt, vp))
#             except Exception: val_auc = np.nan
#             try: val_ll  = float(log_loss(vt, vp, labels=[0, 1]))
#             except Exception: val_ll = np.nan
#             try: val_bri = float(brier_score_loss(vt, vp))
#             except Exception: val_bri = np.nan
#         else:
#             val_auc = val_ll = val_bri = np.nan
#
#         for r in rows:
#             r['val_auc'] = val_auc
#             r['val_logloss'] = val_ll
#             r['val_brier'] = val_bri
#
#         return rows
#
#     # ---------------- search ----------------
#     if base_model == "mlp":
#         eff_jobs = min(max(1, cpu_jobs), 4); prefer = "threads"; backend = "threading"; pre_dispatch = eff_jobs
#         ctx = parallel_backend(backend, n_jobs=eff_jobs)
#     else:
#         eff_jobs = max(1, min(cpu_jobs, 4)) if cpu_jobs != -1 else 4
#         prefer = "processes"; backend = "loky"; pre_dispatch = f"{2*eff_jobs}"
#         ctx = parallel_backend(backend, n_jobs=eff_jobs, inner_max_num_threads=1)
#
#     with ctx:
#         try:
#             with tqdm_joblib(tqdm(total=len(all_param_dicts), desc=f"Param search ({search_mode}, {base_model})")) as _:
#                 out = Parallel(n_jobs=eff_jobs, batch_size=1, prefer=prefer, pre_dispatch=pre_dispatch)(
#                     delayed(evaluate_param_set)(pd_) for pd_ in all_param_dicts
#                 )
#         except OSError as e:
#             print(f"[WARN] Parallel failed with {e}. Falling back to serial search...")
#             out = []
#             for pd_ in tqdm(all_param_dicts, desc=f"Param search (serial, {base_model})"):
#                 out.append(evaluate_param_set(pd_))
#
#     val_rows = [r for sub in out for r in sub]
#     if not val_rows: raise RuntimeError("No validation rows produced (check folds and input data).")
#     val_df = pd.DataFrame(val_rows)
#
#     # ---------------- validation aggregate ----------------
#     if base_model == "xgb":
#         param_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']
#     else:
#         param_keys = ['hidden_layer_sizes','alpha','learning_rate_init','batch_size','max_iter']
#
#     group_cols = param_keys + (['threshold'] if _IS_CLASSIFY else ['edge_param'])
#
#     agg_dict = {
#         'n_preds_val': 'sum',
#         'tp_val': 'sum',
#         'fp_val': 'sum',
#         'val_precision': 'mean',
#         'val_accuracy': 'mean',
#         'val_auc': 'mean',
#         'val_logloss': 'mean',
#         'val_brier': 'mean',
#         'n_value_bets_val': 'sum',
#     }
#     if 'val_edge_ratio_mean' in val_df.columns: agg_dict['val_edge_ratio_mean'] = 'mean'
#     if 'val_edge_ratio_mean_back' in val_df.columns: agg_dict['val_edge_ratio_mean_back'] = 'mean'
#
#     agg = val_df.groupby(group_cols, as_index=False).agg(agg_dict)
#     agg['val_precision_pooled'] = agg.apply(lambda r: (r['tp_val'] / max(1, (r['tp_val'] + r['fp_val']))), axis=1)
#     agg['val_precision_lcb'] = agg.apply(lambda r: _wilson_lcb(int(r['tp_val']), int(r['fp_val']), conf=val_conf_level), axis=1)
#
#     qual_mask = (
#         (agg['val_precision'] >= float(precision_test_threshold)) &
#         (agg['n_preds_val'] >= int(min_samples))
#     )
#     if _USE_VALUE_LAY:  qual_mask &= (agg['n_value_bets_val'] >= int(min_samples))
#     if _USE_VALUE_BACK: qual_mask &= (agg['n_value_bets_val'] >= int(min_samples))
#     qual = agg[qual_mask].copy()
#
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     if qual.empty:
#         fail_csv = None
#         if save_diagnostics_on_fail:
#             diag = (agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                                     ascending=[False, False, False, False])
#                     .assign(fail_reason="failed_validation_gate", market=market))
#             fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv")
#             os.makedirs(csv_save_dir, exist_ok=True); diag.to_csv(fail_csv, index=False)
#         msg = "No strategy met validation gates."
#         if on_fail == "raise": raise RuntimeError(msg)
#         if on_fail == "warn": print("[WARN]", msg)
#         return {'status':'failed_validation_gate','csv':fail_csv,'model_pkl':None,
#                 'summary_df':None,'validation_table':agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                 ascending=[False,False,False,False]).reset_index(drop=True)}
#
#     ranked = qual.sort_values(by=['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                               ascending=[False, False, False, False]).reset_index(drop=True)
#     topk_val = ranked.head(top_k).reset_index(drop=True)
#
#     def _extract_params_from_row(row):
#         return cast_params({k: row[k] for k in param_keys if k in row.index})
#
#     candidates = []
#     for _, row in topk_val.iterrows():
#         c = {
#             'params': _extract_params_from_row(row),
#             'val_precision': float(row['val_precision']),
#             'val_precision_lcb': float(row['val_precision_lcb']),
#             'val_accuracy': float(row['val_accuracy']),
#             'n_preds_val': int(row['n_preds_val']),
#         }
#         if _IS_CLASSIFY:
#             c['threshold'] = float(row['threshold'])
#         else:
#             c['edge_param'] = float(row['edge_param'])
#         candidates.append(c)
#
#     # ---------------- test eval ----------------
#     records_all = []
#     all_bets_collector = []  # across all candidates (to all_bets_dir, not metrics dir)
#
#     def _name_cols(subdf):
#         cols = {}
#         for c in ['date','league','country','home_team','away_team','match_id']:
#             if c in subdf.columns: cols[c] = subdf[c].values
#         if {'home_team','away_team'}.issubset(subdf.columns):
#             cols['event_name'] = (subdf['home_team'] + ' v ' + subdf['away_team']).values
#         return cols
#
#     for cand_id, cand in enumerate(candidates):
#         best_params = cast_params(cand['params'])
#         pos = int(y_train_final.sum()); neg = len(y_train_final) - pos
#         spw_final = (neg/pos) if pos > 0 else 1.0
#
#         final_model = build_model(best_params, spw_final)
#         final_sample_weight = None
#         if base_model == "mlp":
#             w_pos = spw_final
#             final_sample_weight = np.where(y_train_final.values==1, w_pos, 1.0).astype(np.float32)
#
#         fit_model(final_model, X_train_final, y_train_final, X_val_final, y_val_final, sample_weight=final_sample_weight)
#         final_calibrator = fit_calibrator(final_model, X_val_final, y_val_final)
#         p_pos_test = predict_proba_pos(final_calibrator, X_test)
#
#         if _USE_VALUE_LAY:
#             if market == "LAY_AWAY":
#                 p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['away_odds'].values; sel_name = 'AWAY'
#             elif market == "LAY_HOME":
#                 p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['home_odds'].values; sel_name = 'HOME'
#             elif market == "LAY_DRAW":
#                 p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['draw_odds'].values; sel_name = 'DRAW'
#             else:
#                 raise RuntimeError("Internal: LAY mode only for LAY_*")
#
#             fair_odds = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
#             valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01)
#             edge = float(cand.get('edge_param', 0.0))
#             edge_mask = valid & (fair_odds >= (1.0 + edge) * mkt_odds)
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 edge_ratio_minus1 = np.where(mkt_odds > 0, fair_odds / mkt_odds - 1.0, 0.0)
#
#             for plan in staking_plan_lay_options:
#                 stake, liability = _lay_stakes(mkt_odds, edge_ratio_minus1, plan)
#                 stake = np.where(edge_mask, stake, 0.0)
#                 liability = np.where(edge_mask, liability, 0.0)
#
#                 sel_wins = (y_test.values == 0)
#                 pl = np.zeros_like(stake)
#                 idx_win = (stake > 0) & (~sel_wins)
#                 pl[idx_win] = stake[idx_win] * (1.0 - commission_rate)
#                 idx_lose = (stake > 0) & (sel_wins)
#                 pl[idx_lose] = -liability[idx_lose]
#
#                 n_bets = int(np.count_nonzero(stake > 0))
#                 total_pl = float(pl.sum()); avg_pl = float(total_pl / max(1, n_bets))
#
#                 lays_as_preds = (stake > 0).astype(np.uint8)
#                 prc_test = precision_score(y_test, lays_as_preds, zero_division=0)
#                 acc_test = accuracy_score(y_test, lays_as_preds)
#
#                 bet_idx = np.where(stake > 0)[0]
#                 name_cols = _name_cols(df_test.iloc[bet_idx])
#                 bets_df = pd.DataFrame({
#                     **name_cols,
#                     'selection': sel_name,
#                     'market_odds': mkt_odds[bet_idx],
#                     'fair_odds': fair_odds[bet_idx],
#                     'edge_ratio': np.where(mkt_odds[bet_idx] > 0, fair_odds[bet_idx] / mkt_odds[bet_idx], np.nan),
#                     'liability': liability[bet_idx],
#                     'stake': stake[bet_idx],
#                     'commission_rate': float(commission_rate),
#                     'selection_won': sel_wins[bet_idx].astype(int),
#                     'target': y_test.values[bet_idx],
#                     'pl': pl[bet_idx],
#                 })
#                 if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
#                 bets_df['cum_pl'] = bets_df['pl'].cumsum()
#
#                 pv = _pvalue_break_even(bets_df, mode='VALUE_LAY')
#
#                 enough = n_bets >= int(min_test_samples)
#                 not_collapsed = prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop))
#                 pass_gate = bool(enough and not_collapsed)
#                 reason = "" if pass_gate else ("insufficient_test_bets" if not enough else "precision_collapse")
#
#                 if len(bets_df):
#                     meta = {
#                         'candidate_id': cand_id,
#                         'passed_test_gate': bool(pass_gate),
#                         'mode': 'VALUE_LAY',
#                         'market': market,
#                         'threshold': np.nan,
#                         'edge_param': edge,
#                         'staking_plan_lay': plan,
#                         'val_precision': float(cand['val_precision']),
#                         'val_precision_lcb': float(cand['val_precision_lcb']),
#                         'n_value_bets_test': int(n_bets),
#                         'total_pl': float(total_pl),
#                         'avg_pl': float(avg_pl),
#                         'p_value': pv['p_value'],
#                         'zscore': pv['z'],
#                         'commission_rate': float(commission_rate),
#                         'params_json': json.dumps(best_params, default=float),
#                     }
#                     bdf = bets_df.copy()
#                     for k, v in meta.items(): bdf[k] = v
#                     all_bets_collector.append(bdf)

    #             records_all.append({
    #                 **best_params, 'threshold': np.nan, 'edge_param': edge,
    #                 'val_precision_lcb': cand['val_precision_lcb'], 'val_precision': cand['val_precision'],
    #                 'val_accuracy': cand['val_accuracy'],
    #                 'n_value_bets_test': n_bets, 'test_precision_bets': float(prc_test),
    #                 'test_accuracy_bets': float(acc_test), 'total_pl': total_pl, 'avg_pl': avg_pl,
    #                 'p_value': pv['p_value'], 'zscore': pv['z'],
    #                 'pass_test_gate': pass_gate, 'fail_reason': reason, 'model_obj': final_calibrator if pass_gate else None,
    #                 'mode': 'VALUE_LAY', 'bets': bets_df if pass_gate else None,
    #                 'staking_plan_lay': plan,
    #                 'commission_rate': float(commission_rate),
    #             })
    #
    #     elif _USE_VALUE_BACK:
    #         if market == "BACK_AWAY":
    #             p_sel_win = p_pos_test; mkt_odds = df_test['away_odds'].values; sel_name = 'AWAY'
    #         elif market == "BACK_HOME":
    #             p_sel_win = p_pos_test; mkt_odds = df_test['home_odds'].values; sel_name = 'HOME'
    #         elif market == "BACK_DRAW":
    #             p_sel_win = p_pos_test; mkt_odds = df_test['draw_odds'].values; sel_name = 'DRAW'
    #         else:
    #             raise RuntimeError("Internal: BACK mode only for BACK_*")
    #
    #         fair_odds = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
    #         valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01)
    #         edge = float(cand.get('edge_param', 0.0))
    #         edge_mask = valid & (mkt_odds >= (1.0 + edge) * fair_odds)
    #         with np.errstate(divide='ignore', invalid='ignore'):
    #             edge_ratio_minus1 = np.where(fair_odds > 0, mkt_odds / fair_odds - 1.0, 0.0)
    #
    #         for plan in staking_plan_back_options:
    #             stake = _back_stakes(mkt_odds, edge_ratio_minus1, plan, p_sel_win)
    #             stake = np.where(edge_mask, stake, 0.0)
    #
    #             sel_wins = (y_test.values == 1)
    #             pl = np.zeros_like(stake)
    #             win_idx = (stake > 0) & sel_wins
    #             lose_idx = (stake > 0) & (~sel_wins)
    #             pl[win_idx]  = (mkt_odds[win_idx] - 1.0) * stake[win_idx] * (1.0 - commission_rate)
    #             pl[lose_idx] = -stake[lose_idx]
    #
    #             n_bets = int(np.count_nonzero(stake > 0))
    #             total_pl = float(pl.sum()); avg_pl = float(total_pl / max(1, n_bets))
    #
    #             backs_as_preds = (stake > 0).astype(np.uint8)
    #             prc_test = precision_score(y_test, backs_as_preds, zero_division=0)
    #             acc_test = accuracy_score(y_test, backs_as_preds)
    #
    #             bet_idx = np.where(stake > 0)[0]
    #             name_cols = _name_cols(df_test.iloc[bet_idx])
    #             bets_df = pd.DataFrame({
    #                 **name_cols,
    #                 'selection': sel_name,
    #                 'market_odds': mkt_odds[bet_idx],
    #                 'fair_odds': fair_odds[bet_idx],
    #                 'edge_ratio': np.where(fair_odds[bet_idx] > 0, mkt_odds[bet_idx] / fair_odds[bet_idx], np.nan),
    #                 'stake': stake[bet_idx],
    #                 'commission_rate': float(commission_rate),
    #                 'selection_won': sel_wins[bet_idx].astype(int),
    #                 'target': y_test.values[bet_idx],
    #                 'pl': pl[bet_idx],
    #             })
    #             if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
    #             bets_df['cum_pl'] = bets_df['pl'].cumsum()
    #
    #             pv = _pvalue_break_even(bets_df, mode='VALUE_BACK')
    #
    #             enough = n_bets >= int(min_test_samples)
    #             not_collapsed = prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop))
    #             pass_gate = bool(enough and not_collapsed)
    #             reason = "" if pass_gate else ("insufficient_test_bets" if not enough else "precision_collapse")
    #
    #             if len(bets_df):
    #                 meta = {
    #                     'candidate_id': cand_id,
    #                     'passed_test_gate': bool(pass_gate),
    #                     'mode': 'VALUE_BACK',
    #                     'market': market,
    #                     'threshold': np.nan,
    #                     'edge_param': edge,
    #                     'staking_plan_back': plan,
    #                     'val_precision': float(cand['val_precision']),
    #                     'val_precision_lcb': float(cand['val_precision_lcb']),
    #                     'n_value_bets_test': int(n_bets),
    #                     'total_pl': float(total_pl),
    #                     'avg_pl': float(avg_pl),
    #                     'p_value': pv['p_value'],
    #                     'zscore': pv['z'],
    #                     'commission_rate': float(commission_rate),
    #                     'params_json': json.dumps(best_params, default=float),
    #                 }
    #                 bdf = bets_df.copy()
    #                 for k, v in meta.items(): bdf[k] = v
    #                 all_bets_collector.append(bdf)
    #
    #             records_all.append({
    #                 **best_params, 'threshold': np.nan, 'edge_param': edge,
    #                 'val_precision_lcb': cand['val_precision_lcb'], 'val_precision': cand['val_precision'],
    #                 'val_accuracy': cand['val_accuracy'],
    #                 'n_value_bets_test': n_bets, 'test_precision_bets': float(prc_test),
    #                 'test_accuracy_bets': float(acc_test), 'total_pl': total_pl, 'avg_pl': avg_pl,
    #                 'p_value': pv['p_value'], 'zscore': pv['z'],
    #                 'pass_test_gate': pass_gate, 'fail_reason': reason, 'model_obj': final_calibrator if pass_gate else None,
    #                 'mode': 'VALUE_BACK', 'bets': bets_df if pass_gate else None,
    #                 'staking_plan_back': plan,
    #                 'commission_rate': float(commission_rate),
    #             })
    #
    #     else:
    #         # ===== CLASSIFY TEST EVAL WITH P/L + P-VALUE (BACK-LIKE) =====
    #         thr = float(cand['threshold'])
    #         y_pred = (p_pos_test >= thr).astype(np.uint8)
    #         n_preds_test = int(y_pred.sum())
    #         prc_test = precision_score(y_test, y_pred, zero_division=0)
    #         acc_test = accuracy_score(y_test, y_pred)
    #
    #         enough = n_preds_test >= int(min_test_samples)
    #         not_collapsed = prc_test >= max(float(precision_test_threshold),
    #                                         float(cand['val_precision']) - float(max_precision_drop))
    #         pass_gate = bool(enough and not_collapsed)
    #         reason = "" if pass_gate else ("insufficient_test_preds" if not enough else "precision_collapse")
    #
    #         # Build bet-level P/L for positives
    #         bets_df = None
    #         total_pl = float('nan')
    #         avg_pl = float('nan')
    #         p_value = float('nan')
    #         zscore = float('nan')
    #
    #         bet_idx = np.where(y_pred == 1)[0]
    #         if len(bet_idx):
    #             name_cols = {}
    #             subdf = df_test.iloc[bet_idx]
    #             for c in ['date','league','country','home_team','away_team','match_id']:
    #                 if c in subdf.columns: name_cols[c] = subdf[c].values
    #             if {'home_team','away_team'}.issubset(subdf.columns):
    #                 name_cols['event_name'] = (subdf['home_team'] + ' v ' + subdf['away_team']).values
    #
    #             if classify_odds_column is not None and classify_odds_column in df_test.columns:
    #                 mkt_odds = df_test[classify_odds_column].values[bet_idx].astype(float)
    #                 sel_wins = (y_test.values[bet_idx] == 1)
    #                 stake = np.full_like(mkt_odds, float(classify_stake), dtype=float)
    #
    #                 pl = np.zeros_like(stake, dtype=float)
    #                 win_idx = sel_wins
    #                 lose_idx = ~sel_wins
    #                 pl[win_idx]  = (mkt_odds[win_idx] - 1.0) * stake[win_idx] * (1.0 - commission_rate)
    #                 pl[lose_idx] = -stake[lose_idx]
    #
    #                 bets_df = pd.DataFrame({
    #                     **name_cols,
    #                     'selection': 'CLASSIFY_POS',
    #                     'market_odds': mkt_odds,
    #                     'stake': stake,
    #                     'commission_rate': float(commission_rate),
    #                     'selection_won': sel_wins.astype(int),
    #                     'target': y_test.values[bet_idx],
    #                     'pl': pl,
    #                     'threshold': thr,
    #                 })
    #                 if 'date' in bets_df.columns:
    #                     bets_df = bets_df.sort_values('date').reset_index(drop=True)
    #                 bets_df['cum_pl'] = bets_df['pl'].cumsum()
    #
    #                 total_pl = float(bets_df['pl'].sum())
    #                 avg_pl = float(total_pl / max(1, len(bets_df)))
    #
    #                 pv = _pvalue_break_even(bets_df, mode='VALUE_BACK')
    #                 p_value = float(pv['p_value'])
    #                 zscore = float(pv['z'])
    #             else:
    #                 # PSEUDO P/L (no odds) — diagnostic only, no p-value
    #                 sel_wins = (y_test.values[bet_idx] == 1)
    #                 stake = np.full(len(bet_idx), float(classify_stake), dtype=float)
    #                 pl = np.where(sel_wins, stake, -stake)
    #                 bets_df = pd.DataFrame({
    #                     **name_cols,
    #                     'selection': 'CLASSIFY_POS',
    #                     'stake': stake,
    #                     'selection_won': sel_wins.astype(int),
    #                     'target': y_test.values[bet_idx],
    #                     'pl': pl,
    #                     'threshold': thr,
    #                 })
    #                 if 'date' in bets_df.columns:
    #                     bets_df = bets_df.sort_values('date').reset_index(drop=True)
    #                 bets_df['cum_pl'] = bets_df['pl'].cumsum()
    #                 total_pl = float(bets_df['pl'].sum())
    #                 avg_pl = float(total_pl / max(1, len(bets_df)))
    #                 p_value = float('nan'); zscore = float('nan')
    #
    #         records_all.append({
    #             **best_params,
    #             'threshold': thr,
    #             'edge_param': np.nan,
    #             'val_precision_lcb': cand['val_precision_lcb'],
    #             'val_precision': cand['val_precision'],
    #             'val_accuracy': cand['val_accuracy'],
    #             'n_preds_test': n_preds_test,
    #             'test_precision': float(prc_test),
    #             'test_accuracy': float(acc_test),
    #             'total_pl': total_pl,
    #             'avg_pl': avg_pl,
    #             'p_value': p_value,
    #             'zscore': zscore,
    #             'pass_test_gate': pass_gate,
    #             'fail_reason': reason,
    #             'model_obj': final_calibrator if pass_gate else None,
    #             'mode': 'CLASSIFY',
    #             'bets': bets_df if (bets_df is not None) else None,
    #         })
    #
    # survivors_df = pd.DataFrame(records_all)
    # passers = survivors_df[survivors_df['pass_test_gate']].copy()
    #
    # # ---------------- save / rank ----------------
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # tag = "xgb" if base_model == "xgb" else "mlp"
    #
    # if passers.empty:
    #     fail_csv = None
    #     if save_diagnostics_on_fail:
    #         # VALUE and CLASSIFY with odds: rank by p_value first (asc)
    #         if 'p_value' in survivors_df.columns and survivors_df['p_value'].notna().any():
    #             sort_cols = ['p_value','total_pl','val_precision_lcb']
    #             asc = [True, False, False]
    #         else:
    #             sort_cols = ['val_precision_lcb','val_precision','n_preds_test','val_accuracy']
    #             asc = [False, False, False, False]
    #         diag = (survivors_df
    #                 .drop(columns=['model_obj','bets'], errors='ignore')
    #                 .sort_values(by=sort_cols, ascending=asc)
    #                 .assign(market=market))
    #         fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv")
    #         diag.to_csv(fail_csv, index=False); summary_df = diag
    #     else:
    #         summary_df = survivors_df.drop(columns=['model_obj','bets'], errors='ignore')
    #
    #     all_bets_csv_path = None
    #     if save_all_bets_csv and ( _USE_VALUE or (_IS_CLASSIFY and classify_odds_column is not None) ) and all_bets_collector:
    #         all_bets_df = pd.concat(all_bets_collector, ignore_index=True)
    #         if not all_bets_include_failed:
    #             all_bets_df = all_bets_df[all_bets_df['passed_test_gate'] == True]
    #         all_bets_csv_path = os.path.join(all_bets_dir, f"all_bets_{market}_{timestamp}.csv")
    #         all_bets_df.to_csv(all_bets_csv_path, index=False)
    #
    #     msg = "All Top-K failed the TEST gate."
    #     if on_fail == "raise": raise RuntimeError(msg)
    #     if on_fail == "warn": print("[WARN]", msg)
    #     return {'status':'failed_test_gate','csv':fail_csv,'model_pkl':None,
    #             'summary_df':summary_df,'validation_table':ranked,
    #             'bets_csv':None,'pl_plot':None,'all_bets_csv':all_bets_csv_path}
    #
    # # Final ranking
    # if ('p_value' in passers.columns) and passers['p_value'].notna().any():
    #     passers_sorted = passers.sort_values(
    #         by=['p_value','total_pl','avg_pl','val_precision_lcb','val_precision'],
    #         ascending=[True, False, False, False, False]
    #     ).reset_index(drop=True)
    # else:
    #     passers_sorted = passers.sort_values(
    #         by=['val_precision_lcb','val_precision','test_precision','n_preds_test','val_accuracy'],
    #         ascending=[False, False, False, False, False]
    #     ).reset_index(drop=True)
    #
    # # Save PKL + CSV
    # pkl_path = os.path.join(model_dir, f"best_model_{market}_{tag}_calibrated_{timestamp}.pkl")
    # csv_df = passers_sorted.drop(columns=['model_obj','bets'], errors='ignore').copy()
    # csv_df['model_pkl'] = ""; csv_df.loc[0, 'model_pkl'] = pkl_path
    # csv_df['market'] = market
    # csv_path = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}.csv")
    # csv_df.to_csv(csv_path, index=False)
    #
    # # Save top model
    # top_row = passers_sorted.iloc[0]
    # chosen_model = top_row['model_obj']
    # if base_model == "xgb":
    #     param_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']
    # else:
    #     param_keys = ['hidden_layer_sizes','alpha','learning_rate_init','batch_size','max_iter']
    # chosen_params = {k: top_row[k] for k in param_keys if k in passers_sorted.columns}
    # chosen_threshold = float(top_row.get('threshold', np.nan))
    # chosen_edge = float(top_row.get('edge_param', np.nan))
    #
    # joblib.dump(
    #     {
    #         'model': chosen_model,
    #         'threshold': chosen_threshold,            # NaN in VALUE modes; meaningful in CLASSIFY
    #         'edge_param': chosen_edge,                # chosen edge (VALUE)
    #         'features': features,
    #         'base_model': base_model,
    #         'best_params': chosen_params,
    #         'precision_test_threshold': float(precision_test_threshold),
    #         'min_samples': int(min_samples),
    #         'min_test_samples': int(min_test_samples),
    #         'val_conf_level': float(val_conf_level),
    #         'max_precision_drop': float(max_precision_drop),
    #         'market': market,
    #         'mode': top_row['mode'],
    #         # winning staking plan (VALUE modes)
    #         'staking_plan_lay': top_row.get('staking_plan_lay', None) if _USE_VALUE_LAY else None,
    #         'staking_plan_back': top_row.get('staking_plan_back', None) if _USE_VALUE_BACK else None,
    #         # numeric staking params (for reproducibility)
    #         'liability_test': float(liability_test) if _USE_VALUE_LAY else None,
    #         'lay_flat_stake': float(lay_flat_stake) if _USE_VALUE_LAY else None,
    #         'lay_edge_scale': float(lay_edge_scale) if _USE_VALUE_LAY else None,
    #         'kelly_fraction_lay': float(kelly_fraction_lay) if _USE_VALUE_LAY else None,
    #         'min_lay_stake': float(min_lay_stake) if _USE_VALUE_LAY else None,
    #         'max_lay_stake': float(max_lay_stake) if _USE_VALUE_LAY else None,
    #         'min_lay_liability': float(min_lay_liability) if _USE_VALUE_LAY else None,
    #         'max_lay_liability': float(max_lay_liability) if _USE_VALUE_LAY else None,
    #         'back_stake_test': float(back_stake_test) if _USE_VALUE_BACK else None,
    #         'back_edge_scale': float(back_edge_scale) if _USE_VALUE_BACK else None,
    #         'kelly_fraction_back': float(kelly_fraction_back) if _USE_VALUE_BACK else None,
    #         'bankroll_back': float(bankroll_back) if _USE_VALUE_BACK else None,
    #         'min_back_stake': float(min_back_stake) if _USE_VALUE_BACK else None,
    #         'max_back_stake': float(max_back_stake) if _USE_VALUE_BACK else None,
    #         # CLASSIFY specifics (so you know how to apply in live use)
    #         'classify_stake': float(classify_stake) if _IS_CLASSIFY else None,
    #         'classify_odds_column': classify_odds_column if _IS_CLASSIFY else None,
    #         # commission saved
    #         'commission_rate': float(commission_rate),
    #         'notes': ('Commission applied to winning returns; '
    #                   'VALUE & CLASSIFY(with-odds) ranked by smallest p-value; '
    #                   'CLASSIFY builds bet history for positives and saves P/L plot.'),
    #         'run_seed': int(RUN_SEED),
    #     },
    #     pkl_path
    # )
    #
    # # chosen bets CSV / plot (VALUE and CLASSIFY)
    # bets_path = None
    # plot_path = None
    # bets_df = top_row.get('bets', None)
    # if (save_bets_csv or plot_pl) and isinstance(bets_df, pd.DataFrame) and len(bets_df):
    #     if save_bets_csv:
    #         bets_name = f"bets_{market}_{timestamp}.csv"
    #         bets_path = os.path.join(bets_csv_dir, bets_name)
    #         bets_df.to_csv(bets_path, index=False)
    #     if plot_pl:
    #         try:
    #             import matplotlib.pyplot as plt
    #             fig = plt.figure()
    #             x = bets_df['date'] if 'date' in bets_df.columns else np.arange(len(bets_df))
    #             plt.plot(x, bets_df['cum_pl'])
    #             title = f"{market} cumulative P/L ({top_row['mode']})"
    #             if plot_title_suffix: title += f" — {plot_title_suffix}"
    #             plt.title(title)
    #             plt.xlabel('Date' if 'date' in bets_df.columns else 'Bet #')
    #             plt.ylabel('Cumulative P/L')
    #             plt.tight_layout()
    #             plot_name = f"cum_pl_{market}_{timestamp}.png"
    #             plot_path = os.path.join(plot_dir, plot_name)
    #             plt.savefig(plot_path, dpi=160); plt.close(fig)
    #         except Exception as e:
    #             print(f"[WARN] Failed to create plot: {e}")
    #
    # # ALL bets export (across all candidates) — lives in sibling all_bets dir
    # all_bets_csv_path = None
    # if save_all_bets_csv and ( _USE_VALUE or (_IS_CLASSIFY and classify_odds_column is not None) ) and all_bets_collector:
    #     all_bets_df = pd.concat(all_bets_collector, ignore_index=True)
    #     if not all_bets_include_failed:
    #         all_bets_df = all_bets_df[all_bets_df['passed_test_gate'] == True]
    #     preferred = [c for c in [
    #         'date','league','country','home_team','away_team','match_id','event_name','selection',
    #         'market_odds','fair_odds','edge_ratio','stake','liability','commission_rate',
    #         'selection_won','target','pl','cum_pl',
    #         'candidate_id','passed_test_gate','mode','market','threshold','edge_param',
    #         'staking_plan_lay','staking_plan_back',
    #         'val_precision','val_precision_lcb','n_value_bets_test','total_pl','avg_pl','p_value','zscore','params_json'
    #     ] if c in all_bets_df.columns]
    #     all_bets_df = all_bets_df[preferred + [c for c in all_bets_df.columns if c not in preferred]]
    #     all_bets_csv_path = os.path.join(all_bets_dir, f"all_bets_{market}_{timestamp}.csv")
    #     all_bets_df.to_csv(all_bets_csv_path, index=False)
    #
    # return {
    #     'status': 'ok',
    #     'csv': csv_path,
    #     'model_pkl': pkl_path,
    #     'summary_df': csv_df,
    #     'validation_table': ranked,
    #     'bets_csv': bets_path,
    #     'pl_plot': plot_path,
    #     'all_bets_csv': all_bets_csv_path,
    # }

# def run_models_outcome(
#     matches_filtered: pd.DataFrame,
#     features: list,
#     # ── gates ──────────────────────────────────────────────────────────────
#     min_samples: int = 200,
#     min_test_samples: int = 100,
#     precision_test_threshold: float = 0.80,
#     # ── model/search ───────────────────────────────────────────────────────
#     base_model: str = "xgb",
#     search_mode: str = "random",
#     n_random_param_sets: int = 10,
#     cpu_jobs: int = 6,
#     top_k: int = 10,
#     thresholds: np.ndarray | None = None,        # USED for CLASSIFY markets
#     out_dir: str | None = None,
#     # ── anti-overfitting ──────────────────────────────────────────────────
#     val_conf_level: float = 0.99,
#     max_precision_drop: float = 0.05,
#     # ── failure handling ───────────────────────────────────────────────────
#     on_fail: str = "return",                     # "return" | "warn" | "raise"
#     save_diagnostics_on_fail: bool = True,
#     # ── market ─────────────────────────────────────────────────────────────
#     market: str = "LAY_AWAY",                    # LAY_* | BACK_* | OVER | UNDER (or other classify markets)
#
#     # ── VALUE LAY controls ────────────────────────────────────────────────
#     use_value_for_lay: bool = True,
#     value_edge_grid_lay: np.ndarray | None = None,   # e.g. np.round(np.arange(0.00,0.201,0.01),2)
#
#     # Staking plan search toggle + options (VALUE modes)
#     enable_staking_plan_search: bool = False,
#     staking_plan_lay_options: list[str] | None = None,
#     staking_plan_back_options: list[str] | None = None,
#
#     # Single-plan (used when enable_staking_plan_search=False)
#     staking_plan_lay: str = "liability",             # "liability" | "flat_stake" | "edge_prop" | "kelly_approx"
#     staking_plan_back: str = "flat",                 # "flat" | "edge_prop" | "kelly"
#
#     # ── LAY staking parameters (balanced defaults) ────────────────────────
#     liability_test: float = 1.0,
#     lay_flat_stake: float = 1.0,
#     lay_edge_scale: float = 0.05,
#     kelly_fraction_lay: float = 1.0,
#     min_lay_stake: float = 0.0,
#     max_lay_stake: float = 1.0,
#     min_lay_liability: float = 0.0,
#     max_lay_liability: float = 2.0,
#
#     # ── VALUE BACK controls ────────────────────────────────────────────────
#     use_value_for_back: bool = True,
#     value_edge_grid_back: np.ndarray | None = None,
#
#     # BACK staking parameters
#     back_stake_test: float = 1.0,
#     back_edge_scale: float = 0.10,
#     kelly_fraction_back: float = 0.25,
#     bankroll_back: float = 100.0,
#     min_back_stake: float = 0.0,
#     max_back_stake: float = 10.0,
#
#     # ── CLASSIFY staking / odds (adds odds-band grid sweep) ───────────────
#     classify_stake: float = 1.0,                      # flat stake for each positive bet
#     classify_odds_column: str | None = None,          # e.g. 'away_odds', 'over25_odds'
#     classify_side: str = "back",                      # "back" or "lay" for classify bets
#     classify_odds_min_grid: np.ndarray | None = None, # e.g. np.arange(1.00, 10.01, 0.25)
#     classify_odds_max_grid: np.ndarray | None = None, # e.g. np.arange(1.00, 10.01, 0.25)
#
#     # ── COMMISSION (applied to net winning returns) ───────────────────────
#     commission_rate: float = 0.02,  # 2% commission on winnings
#
#     # ── OUTPUTS: chosen model ─────────────────────────────────────────────
#     save_bets_csv: bool = False,
#     bets_csv_dir: str | None = None,
#     plot_pl: bool = False,
#     plot_dir: str | None = None,
#     plot_title_suffix: str = "",
#     # ── OUTPUTS: ALL candidates ───────────────────────────────────────────
#     save_all_bets_csv: bool = False,
#     all_bets_dir: str | None = None,
#     all_bets_include_failed: bool = True,
# ):
#     """
#     VALUE modes behave as before (commission-adjusted). CLASSIFY now also:
#       • Sweeps thresholds
#       • If `classify_odds_column` provided, sweeps (odds_min, odds_max) bands over [1.00, 10.00] step 0.25
#       • Supports `classify_side="back"| "lay"`:
#           - back: bet when p>=thr within odds band, P/L like back bets
#           - lay:  bet when p<=1-thr within odds band, P/L like lay bets
#       • Computes commission-adjusted P/L and p-value vs break-even
#       • Ranks by smallest p-value (then P/L), saves bets CSV and P/L plot
#     """
#     # ---------------- setup ----------------
#     import os, secrets, hashlib, json
#     from datetime import datetime
#     import numpy as np
#     import pandas as pd
#     from itertools import product
#     from sklearn.model_selection import ParameterSampler
#     from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, log_loss, brier_score_loss
#     from sklearn.calibration import CalibratedClassifierCV
#     from sklearn.pipeline import Pipeline, make_pipeline
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.neural_network import MLPClassifier
#     from joblib import Parallel, delayed, parallel_backend
#     from tqdm import tqdm
#     from tqdm_joblib import tqdm_joblib
#     import joblib
#
#     # --- xgboost import (optional)
#     try:
#         import xgboost as xgb
#         _HAS_XGB_LOCAL = True
#     except Exception:
#         _HAS_XGB_LOCAL = False
#     _HAS_XGB = globals().get("_HAS_XGB", _HAS_XGB_LOCAL)
#     if base_model == "xgb" and not _HAS_XGB:
#         raise ImportError("XGBoost not available; set base_model='mlp' or install xgboost.")
#
#     # --- random dists
#     try:
#         _randint; _uniform; _loguniform
#     except NameError:
#         from scipy.stats import randint as _randint
#         from scipy.stats import uniform as _uniform
#         from scipy.stats import loguniform as _loguniform
#
#     # --- Wilson LCB & normal CDF
#     try:
#         from scipy.stats import norm
#         _Z = lambda conf: float(norm.ppf(1 - (1 - conf) / 2))
#         _Phi = lambda z: float(norm.cdf(z))
#     except Exception:
#         import math
#         _Z = lambda conf: 1.96 if abs(conf - 0.95) < 1e-6 else 1.64
#         _Phi = lambda z: 0.5 * (1.0 + math.erf(z / (2**0.5)))
#
#     def _wilson_lcb(tp: int, fp: int, conf: float) -> float:
#         n = tp + fp
#         if n <= 0: return 0.0
#         p = tp / n
#         z = _Z(conf)
#         denom = 1.0 + (z*z)/n
#         centre = p + (z*z)/(2*n)
#         rad = z * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
#         return max(0.0, (centre - rad) / denom)
#
#     # defaults
#     if thresholds is None:
#         thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)  # CLASSIFY only
#     if value_edge_grid_lay is None:
#         value_edge_grid_lay = np.round(np.arange(0.00, 0.201, 0.01), 2)
#     if value_edge_grid_back is None:
#         value_edge_grid_back = np.round(np.arange(0.00, 0.201, 0.01), 2)
#     if classify_odds_min_grid is None:
#         classify_odds_min_grid = np.round(np.arange(1.00, 10.01, 0.25), 2)
#     if classify_odds_max_grid is None:
#         classify_odds_max_grid = np.round(np.arange(1.00, 10.01, 0.25), 2)
#     classify_side = str(classify_side).lower().strip()
#     if classify_side not in ("back","lay"):
#         raise ValueError("classify_side must be 'back' or 'lay'")
#
#     # normalise staking-plan options
#     if staking_plan_lay_options is None:
#         staking_plan_lay_options = ["liability", "flat_stake", "edge_prop", "kelly_approx"]
#     if staking_plan_back_options is None:
#         staking_plan_back_options = ["flat", "edge_prop", "kelly"]
#     if not enable_staking_plan_search:
#         staking_plan_lay_options = [staking_plan_lay]
#         staking_plan_back_options = [staking_plan_back]
#
#     # --- paths
#     BASE = r"C:\Users\leere\PycharmProjects\Football_ML3\FT Results"
#     PKL_DIRS = {
#         "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "model_file"),
#         "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "model_file"),
#         "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "model_file"),
#         "BACK_HOME": os.path.join(BASE, "Back_Home", "model_file"),
#         "BACK_AWAY": os.path.join(BASE, "Back_Away", "model_file"),
#         "BACK_DRAW": os.path.join(BASE, "Back_Draw", "model_file"),
#         "OVER":      os.path.join(BASE, "Over_2_5",  "model_file"),
#         "UNDER":     os.path.join(BASE, "Under_2_5", "model_file"),
#     }
#     CSV_DIRS = {
#         "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "best_model_metrics"),
#         "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "best_model_metrics"),
#         "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "best_model_metrics"),
#         "BACK_HOME": os.path.join(BASE, "Back_Home", "best_model_metrics"),
#         "BACK_AWAY": os.path.join(BASE, "Back_Away", "best_model_metrics"),
#         "BACK_DRAW": os.path.join(BASE, "Back_Draw", "best_model_metrics"),
#         "OVER":      os.path.join(BASE, "Over_2_5",  "best_model_metrics"),
#         "UNDER":     os.path.join(BASE, "Under_2_5", "best_model_metrics"),
#     }
#
#     market = str(market).upper().strip()
#     if market not in PKL_DIRS: raise ValueError(f"Unsupported market '{market}'.")
#     _IS_LAY  = market.startswith("LAY_")
#     _IS_BACK = market.startswith("BACK_")
#     _USE_VALUE_LAY  = bool(use_value_for_lay and _IS_LAY)
#     _USE_VALUE_BACK = bool(use_value_for_back and _IS_BACK)
#     _USE_VALUE = _USE_VALUE_LAY or _USE_VALUE_BACK
#     _IS_CLASSIFY = not _USE_VALUE
#
#     csv_save_dir = out_dir if (out_dir and len(str(out_dir)) > 0) else CSV_DIRS[market]
#     os.makedirs(csv_save_dir, exist_ok=True)
#     model_dir = PKL_DIRS[market]; os.makedirs(model_dir, exist_ok=True)
#     if bets_csv_dir is None: bets_csv_dir = csv_save_dir
#     if plot_dir is None: plot_dir = csv_save_dir
#     os.makedirs(bets_csv_dir, exist_ok=True)
#     os.makedirs(plot_dir, exist_ok=True)
#     if all_bets_dir is None:
#         all_bets_dir = os.path.join(os.path.dirname(CSV_DIRS[market]), "all_bets")
#     os.makedirs(all_bets_dir, exist_ok=True)
#
#     RUN_SEED = secrets.randbits(32)
#     def _seed_from(*vals) -> int:
#         h = hashlib.blake2b(digest_size=8)
#         h.update(int(RUN_SEED).to_bytes(8,'little',signed=False))
#         for v in vals: h.update(str(v).encode('utf-8'))
#         return int.from_bytes(h.digest(),'little') & 0x7FFFFFFF
#
#     def _as_float(x):
#         try: return float(x)
#         except Exception: return float(str(x))
#     def _as_int(x): return int(float(x))
#
#     # ---------------- data ----------------
#     req_cols = {'date','target'}
#     if _USE_VALUE: req_cols |= {'home_odds','draw_odds','away_odds'}
#     missing = req_cols - set(matches_filtered.columns)
#     if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
#
#     df = matches_filtered.copy()
#     df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     df = df.sort_values('date').reset_index(drop=True)
#
#     cols_needed = list(set(features) | {'target'} | ({'home_odds','draw_odds','away_odds'} if _USE_VALUE else set()))
#     if _IS_CLASSIFY and classify_odds_column is not None:
#         cols_needed = list(set(cols_needed) | {classify_odds_column})
#     df = df.dropna(subset=cols_needed).reset_index(drop=True)
#
#     X = df[features].copy()
#     y = df['target'].astype(int).reset_index(drop=True)
#
#     n = len(X)
#     if n < max(min_samples * 3, 500): raise RuntimeError(f"Not enough rows: {n}")
#
#     # temporal split
#     test_start = int(0.85 * n)
#     pretest_end = test_start
#     X_test = X.iloc[test_start:].reset_index(drop=True)
#     y_test = y.iloc[test_start:].reset_index(drop=True)
#     df_test = df.iloc[test_start:].reset_index(drop=True)
#
#     # rolling validation folds
#     N_FOLDS = 5
#     total_val_len = max(1, int(0.15 * n))
#     val_len = max(1, total_val_len // N_FOLDS)
#     fold_val_ends = [pretest_end - total_val_len + (i + 1) * val_len for i in range(N_FOLDS)]
#     fold_val_starts = [end - val_len for end in fold_val_ends]
#     if fold_val_ends:
#         fold_val_ends[-1] = min(fold_val_ends[-1], pretest_end)
#         fold_val_starts[-1] = max(0, fold_val_ends[-1] - val_len)
#
#     # final small validation slice (for calibration before test)
#     final_val_len = max(1, val_len)
#     final_val_start = max(0, test_start - final_val_len)
#     X_train_final = X.iloc[:final_val_start]
#     y_train_final = y.iloc[:final_val_start]
#     X_val_final   = X.iloc[final_val_start:test_start]
#     y_val_final   = y.iloc[final_val_start:test_start]
#
#     # ---------------- param spaces ----------------
#     xgb_param_grid = {'n_estimators':[200],'max_depth':[5],'learning_rate':[0.1],'subsample':[0.7],
#                       'colsample_bytree':[1.0],'min_child_weight':[5],'reg_lambda':[1.0]}
#     xgb_param_distributions = {'n_estimators':_randint(100,1001),'max_depth':_randint(3,8),
#                                'learning_rate':_loguniform(0.01,0.2),'min_child_weight':_randint(3,13),
#                                'subsample':_uniform(0.7,0.3),'colsample_bytree':_uniform(0.6,0.4),
#                                'reg_lambda':_loguniform(0.1,10.0)}
#     mlp_param_grid = {'hidden_layer_sizes':[(128,),(256,),(128,64)],'alpha':[1e-4],
#                       'learning_rate_init':[1e-3],'batch_size':['auto'],'max_iter':[200]}
#     mlp_param_distributions = {'hidden_layer_sizes':[(64,),(128,),(256,),(128,64),(256,128)],
#                                'alpha':_loguniform(1e-5,1e-2),'learning_rate_init':_loguniform(5e-4,5e-2),
#                                'batch_size':_randint(32,257),'max_iter':_randint(150,401)}
#
#     def cast_params(p: dict) -> dict:
#         q = dict(p)
#         if base_model == "xgb":
#             for k in ['n_estimators','max_depth','min_child_weight']:
#                 if k in q: q[k] = _as_int(q[k])
#             for k in ['learning_rate','subsample','colsample_bytree','reg_lambda']:
#                 if k in q: q[k] = _as_float(q[k])
#         else:
#             if 'max_iter' in q: q['max_iter'] = _as_int(q['max_iter'])
#             if 'batch_size' in q and q['batch_size'] != 'auto': q['batch_size'] = _as_int(q['batch_size'])
#             if 'alpha' in q: q['alpha'] = _as_float(q['alpha'])
#             if 'learning_rate_init' in q: q['learning_rate_init'] = _as_float(q['learning_rate_init'])
#             if 'hidden_layer_sizes' in q:
#                 h = q['hidden_layer_sizes']
#                 if isinstance(h, str):
#                     parts = [pp.strip() for pp in h.strip("()").split(",") if pp.strip()!='']
#                     q['hidden_layer_sizes'] = tuple(_as_int(pp) for pp in parts) if parts else (128,)
#                 elif isinstance(h, (list, tuple, np.ndarray)):
#                     q['hidden_layer_sizes'] = tuple(int(v) for v in h)
#                 else:
#                     q['hidden_layer_sizes'] = (int(h),)
#         return q
#
#     def _final_step_name(estimator):
#         try:
#             if isinstance(estimator, Pipeline): return estimator.steps[-1][0]
#         except Exception:
#             pass
#         return None
#
#     def build_model(params: dict, spw: float):
#         model_seed = _seed_from("model", base_model, tuple(sorted(params.items())))
#         if base_model == "xgb":
#             return xgb.XGBClassifier(
#                 objective='binary:logistic',
#                 eval_metric='auc',
#                 random_state=model_seed,
#                 scale_pos_weight=spw,
#                 n_jobs=1,
#                 tree_method="hist",
#                 verbosity=0,
#                 **params
#             )
#         else:
#             mlp = MLPClassifier(
#                 random_state=model_seed,
#                 early_stopping=True,
#                 n_iter_no_change=20,
#                 validation_fraction=0.1,
#                 solver="adam",
#                 **params
#             )
#             return make_pipeline(StandardScaler(with_mean=True, with_std=True), mlp)
#
#     def fit_model(model, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None):
#         if base_model == "xgb":
#             try:
#                 model.set_params(verbosity=0, early_stopping_rounds=50)
#                 if X_va is not None and y_va is not None:
#                     model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
#                 else:
#                     model.fit(X_tr, y_tr, verbose=False)
#             except Exception:
#                 model.fit(X_tr, y_tr, verbose=False)
#         else:
#             fit_kwargs = {}
#             if sample_weight is not None:
#                 stepname = _final_step_name(model)
#                 if stepname is not None:
#                     fit_kwargs[f"{stepname}__sample_weight"] = sample_weight
#             try:
#                 model.fit(X_tr, y_tr, **fit_kwargs)
#             except TypeError:
#                 model.fit(X_tr, y_tr)
#
#     def fit_calibrator(fitted, X_va, y_va):
#         try:
#             from sklearn.calibration import FrozenEstimator
#             frozen = FrozenEstimator(fitted)
#             cal = CalibratedClassifierCV(frozen, method='sigmoid', cv=None)
#             cal.fit(X_va, y_va)
#             return cal
#         except Exception:
#             try:
#                 cal = CalibratedClassifierCV(fitted, method='sigmoid', cv='prefit')
#                 cal.fit(X_va, y_va); return cal
#             except Exception:
#                 return fitted
#
#     def _unwrap_estimator(est):
#         if isinstance(est, Pipeline): return est.steps[-1][1]
#         return est
#
#     def predict_proba_pos(model_or_cal, X_):
#         proba = model_or_cal.predict_proba(X_)
#         if proba.ndim == 2:
#             classes = getattr(model_or_cal, "classes_", None)
#             if classes is None:
#                 base = _unwrap_estimator(model_or_cal); classes = getattr(base, "classes_", None)
#             if classes is not None and len(classes) == proba.shape[1]:
#                 try:
#                     idx = int(np.where(np.asarray(classes) == 1)[0][0])
#                     return proba[:, idx].astype(np.float32)
#                 except Exception:
#                     pass
#             if proba.shape[1] == 2: return proba[:, 1].astype(np.float32)
#             if proba.shape[1] == 1:
#                 only = getattr(model_or_cal, "classes_", [0])[0]
#                 return (np.ones_like(proba[:,0]) if only==1 else np.zeros_like(proba[:,0])).astype(np.float32)
#         return np.asarray(proba, dtype=np.float32)
#
#     # --- p-value helper (commission-adjusted) ------------------------------
#     def _pvalue_break_even(bdf: pd.DataFrame, mode: str) -> dict:
#         if not isinstance(bdf, pd.DataFrame) or bdf.empty:
#             return {'z': 0.0, 'p_value': 1.0, 'var_sum': 0.0, 'n': 0, 'total_pl': 0.0}
#         o = np.asarray(bdf['market_odds'].values, dtype=float)
#         o = np.where(o <= 1.0, np.nan, o)
#         p = 1.0 / o  # null win prob
#         if mode == 'VALUE_BACK':
#             S = np.asarray(bdf['stake'].values, dtype=float)
#             win = (o - 1.0) * S * (1.0 - commission_rate)
#             lose = -S
#         else:  # VALUE_LAY
#             L = np.asarray(bdf.get('liability', np.nan*np.ones_like(o))).astype(float)
#             stake = np.asarray(bdf['stake'].values, dtype=float)
#             win  = stake * (1.0 - commission_rate)   # selection loses
#             lose = -L                                 # selection wins
#         var_i = p * (win ** 2) + (1.0 - p) * (lose ** 2)
#         var_i = np.where(np.isfinite(var_i), var_i, 0.0)
#         pl = np.asarray(bdf['pl'].values, dtype=float)
#         total_pl = float(np.nansum(np.where(np.isfinite(pl), pl, 0.0)))
#         var_sum = float(np.nansum(var_i))
#         z = total_pl / (np.sqrt(var_sum) + 1e-12)
#         p_val = max(0.0, 1.0 - _Phi(z))  # one-sided
#         return {'z': float(z), 'p_value': float(p_val), 'var_sum': var_sum, 'n': int(len(pl)), 'total_pl': total_pl}
#
#     def _lay_stakes(odds: np.ndarray, fair_over_market_minus1: np.ndarray, plan: str):
#         """
#         Compute (stake, liability) for lay bets with the exact identity:
#             liability = stake * (odds - 1)
#         while satisfying BOTH stake and liability clamps.
#
#         Parameters
#         ----------
#         odds : np.ndarray           # market decimal odds o
#         fair_over_market_minus1 : np.ndarray  # edge ratio - 1 (>=0 when fair >= market)
#         plan : str                  # "liability" | "flat_stake" | "edge_prop" | "kelly_approx"
#         """
#         o = np.asarray(odds, dtype=float)
#         edge_plus = np.maximum(np.asarray(fair_over_market_minus1, dtype=float), 0.0)
#         # avoid divide-by-zero when o≈1
#         denom = np.maximum(o - 1.0, 1e-12)
#
#         # --- helpers: apply joint bounds coherently ---------------------------
#         # Enforce BOTH stake and liability bounds at once, keeping L == S*(o-1)
#         def _apply_joint_bounds_from_stake(stake_desired: np.ndarray):
#             # stake must satisfy its own [min,max] AND the implicit liability bounds:
#             #   L = S*denom ∈ [min_lay_liability, max_lay_liability]
#             # => S ∈ [min_lay_liability/denom, max_lay_liability/denom]
#             stake_min_joint = np.maximum(float(min_lay_stake), float(min_lay_liability) / denom)
#             stake_max_joint = np.minimum(float(max_lay_stake), float(max_lay_liability) / denom)
#             stake = np.clip(stake_desired, stake_min_joint, stake_max_joint)
#             L = stake * denom
#             return stake, L
#
#         def _apply_joint_bounds_from_liability(L_desired: np.ndarray):
#             # liability must satisfy its own [min,max] AND the implicit stake bounds:
#             #   S = L/denom ∈ [min_lay_stake, max_lay_stake]
#             # => L ∈ [min_lay_stake*denom, max_lay_stake*denom]
#             L_min_joint = np.maximum(float(min_lay_liability), float(min_lay_stake) * denom)
#             L_max_joint = np.minimum(float(max_lay_liability), float(max_lay_stake) * denom)
#             L = np.clip(L_desired, L_min_joint, L_max_joint)
#             stake = L / denom
#             return stake, L
#
#         # --- sizing per plan ---------------------------------------------------
#         if plan == "liability":
#             # PRIMARY: target liability
#             L_desired = np.full_like(o, float(liability_test), dtype=float)
#             stake, L = _apply_joint_bounds_from_liability(L_desired)
#
#         elif plan == "flat_stake":
#             # PRIMARY: flat stake
#             stake_desired = np.full_like(o, float(lay_flat_stake), dtype=float)
#             stake, L = _apply_joint_bounds_from_stake(stake_desired)
#
#         elif plan == "edge_prop":
#             # PRIMARY: liability proportional to edge
#             # L_desired = liability_test * (edge_plus / lay_edge_scale)
#             scale = max(1e-12, float(lay_edge_scale))
#             L_desired = float(liability_test) * (edge_plus / scale)
#             stake, L = _apply_joint_bounds_from_liability(L_desired)
#
#         elif plan == "kelly_approx":
#             # PRIMARY: liability ≈ k * edge
#             L_desired = float(liability_test) * float(kelly_fraction_lay) * edge_plus
#             stake, L = _apply_joint_bounds_from_liability(L_desired)
#
#         else:
#             raise ValueError(f"Unknown staking_plan_lay: {plan}")
#
#         # Final safety: ensure finite & non-negative
#         stake = np.where(np.isfinite(stake), np.maximum(stake, 0.0), 0.0)
#         L = np.where(np.isfinite(L), np.maximum(L, 0.0), 0.0)
#         return stake, L
#
#     def _back_stakes(odds: np.ndarray, fair_over_market_minus1: np.ndarray, plan: str, p_win: np.ndarray):
#         o = np.asarray(odds, dtype=float)
#         p = np.clip(np.asarray(p_win, dtype=float), 0.0, 1.0)
#         edge_plus = np.maximum(fair_over_market_minus1, 0.0)
#         if plan == "flat":
#             stake = np.full_like(o, float(back_stake_test), dtype=float)
#         elif plan == "edge_prop":
#             stake = float(back_stake_test) * np.divide(edge_plus, max(1e-9, float(back_edge_scale)))
#         elif plan == "kelly":
#             b = np.maximum(o - 1.0, 1e-9)
#             f = (b * p - (1.0 - p)) / b
#             f = np.maximum(f, 0.0)
#             stake = float(bankroll_back) * float(kelly_fraction_back) * f
#         else:
#             raise ValueError(f"Unknown staking_plan_back: {plan}")
#         stake = np.clip(stake, float(min_back_stake), float(max_back_stake))
#         return stake
#
#     # ---------------- search space ----------------
#     if search_mode.lower() == "grid":
#         grid = xgb_param_grid if base_model == "xgb" else mlp_param_grid
#         all_param_dicts = [dict(zip(grid.keys(), combo)) for combo in product(*grid.values())]
#     else:
#         dists = xgb_param_distributions if base_model == "xgb" else mlp_param_distributions
#         sampler_seed = _seed_from("sampler")
#         all_param_dicts = list(ParameterSampler(dists, n_iter=n_random_param_sets, random_state=sampler_seed))
#
#     # ---------------- validation eval ----------------
#     def evaluate_param_set(param_dict, *_):
#         safe = cast_params(param_dict)
#         rows = []; val_prob_all=[]; val_true_all=[]
#
#         for vstart, vend in zip(fold_val_starts, fold_val_ends):
#             if vstart is None or vend is None or vstart <= 0 or vend <= vstart: continue
#             X_tr, y_tr = X.iloc[:vstart], y.iloc[:vstart]
#             X_va, y_va = X.iloc[vstart:vend], y.iloc[vstart:vend]
#             df_va = df.iloc[vstart:vend]
#             if y_tr.nunique() < 2: continue
#
#             pos = int(y_tr.sum()); neg = len(y_tr) - pos
#             spw = (neg/pos) if pos > 0 else 1.0
#
#             sample_weight = None
#             if base_model == "mlp":
#                 w_pos = spw
#                 sample_weight = np.where(y_tr.values==1, w_pos, 1.0).astype(np.float32)
#
#             model = build_model(safe, spw)
#             fit_model(model, X_tr, y_tr, X_va, y_va, sample_weight=sample_weight)
#             cal = fit_calibrator(model, X_va, y_va)
#
#             p_pos = predict_proba_pos(cal, X_va)
#             val_prob_all.append(p_pos)
#             y_true = y_va.values.astype(np.uint8); val_true_all.append(y_true)
#
#             if _IS_CLASSIFY:
#                 if classify_odds_column is None or classify_odds_column not in df_va.columns:
#                     # No odds band sweep; just threshold sweep
#                     for thr in thresholds:
#                         thr = float(thr)
#                         if classify_side == "back":
#                             take = p_pos >= thr
#                         else:  # lay
#                             take = p_pos <= (1.0 - thr)
#                         y_pred = (take).astype(np.uint8)
#                         n_preds = int(y_pred.sum())
#                         tp = int(((y_true == 1) & (y_pred == 1)).sum())
#                         fp = int(((y_true == 0) & (y_pred == 1)).sum())
#                         prc = precision_score(y_va, y_pred, zero_division=0)
#                         acc = accuracy_score(y_va, y_pred)
#                         rows.append({
#                             **safe,
#                             'threshold': thr,
#                             'odds_min': np.nan, 'odds_max': np.nan,
#                             'fold_vstart': int(vstart),
#                             'fold_vend': int(vend),
#                             'n_preds_val': n_preds,
#                             'tp_val': tp,
#                             'fp_val': fp,
#                             'val_precision': float(prc),
#                             'val_accuracy': float(acc),
#                             'n_value_bets_val': n_preds,
#                             'val_edge_ratio_mean': np.nan,
#                             'val_edge_ratio_mean_back': np.nan,
#                         })
#                 else:
#                     mkt = df_va[classify_odds_column].values.astype(float)
#                     valid = np.isfinite(mkt) & (mkt > 1.01)
#                     for thr in thresholds:
#                         thr = float(thr)
#                         if classify_side == "back":
#                             pred_mask = (p_pos >= thr)
#                         else:
#                             pred_mask = (p_pos <= (1.0 - thr))
#                         for omin in classify_odds_min_grid:
#                             for omax in classify_odds_max_grid:
#                                 omin = float(omin); omax = float(omax)
#                                 if omin > omax: continue
#                                 odds_mask = valid & (mkt >= omin) & (mkt <= omax)
#                                 take = pred_mask & odds_mask
#                                 y_pred = take.astype(np.uint8)
#                                 n_preds = int(y_pred.sum())
#                                 tp = int(((y_true == 1) & (y_pred == 1)).sum())
#                                 fp = int(((y_true == 0) & (y_pred == 1)).sum())
#                                 prc = precision_score(y_va, y_pred, zero_division=0)
#                                 acc = accuracy_score(y_va, y_pred)
#                                 rows.append({
#                                     **safe,
#                                     'threshold': thr,
#                                     'odds_min': omin, 'odds_max': omax,
#                                     'fold_vstart': int(vstart),
#                                     'fold_vend': int(vend),
#                                     'n_preds_val': n_preds,
#                                     'tp_val': tp,
#                                     'fp_val': fp,
#                                     'val_precision': float(prc),
#                                     'val_accuracy': float(acc),
#                                     'n_value_bets_val': n_preds,
#                                     'val_edge_ratio_mean': np.nan,
#                                     'val_edge_ratio_mean_back': np.nan,
#                                 })
#             else:
#                 # VALUE modes: (unchanged) edge sweep
#                 if _IS_LAY:
#                     mkt = df_va['away_odds'].values if market=="LAY_AWAY" else (df_va['home_odds'].values if market=="LAY_HOME" else df_va['draw_odds'].values)
#                     p_sel_win = 1.0 - p_pos
#                     fair = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
#                 else:
#                     mkt = df_va['away_odds'].values if market=="BACK_AWAY" else (df_va['home_odds'].values if market=="BACK_HOME" else df_va['draw_odds'].values)
#                     p_sel_win = p_pos
#                     fair = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
#                 edge_grid = value_edge_grid_lay if _IS_LAY else value_edge_grid_back
#                 for edge_param in edge_grid:
#                     if _IS_LAY:
#                         edge_mask = (fair >= (1.0 + float(edge_param)) * mkt) & np.isfinite(mkt)
#                         with np.errstate(divide='ignore', invalid='ignore'):
#                             edge_ratio = fair / mkt
#                         val_edge_mean = float(np.nanmean(np.where(np.isfinite(edge_ratio), edge_ratio, np.nan)))
#                     else:
#                         edge_mask = (mkt >= (1.0 + float(edge_param)) * fair) & np.isfinite(mkt)
#                         with np.errstate(divide='ignore', invalid='ignore'):
#                             edge_ratio = mkt / fair
#                         val_edge_mean = float(np.nanmean(np.where(np.isfinite(edge_ratio), edge_ratio, np.nan)))
#                     y_pred = edge_mask.astype(np.uint8)
#                     tp = int(((y_true == 1) & (y_pred == 1)).sum())
#                     fp = int(((y_true == 0) & (y_pred == 1)).sum())
#                     prc = precision_score(y_true, y_pred, zero_division=0)
#                     acc = accuracy_score(y_true, y_pred)
#                     rows.append({
#                         **safe,
#                         'threshold': np.nan,
#                         'odds_min': np.nan, 'odds_max': np.nan,
#                         'edge_param': float(edge_param),
#                         'fold_vstart': int(vstart),
#                         'fold_vend': int(vend),
#                         'n_preds_val': int(y_pred.sum()),
#                         'tp_val': tp,
#                         'fp_val': fp,
#                         'val_precision': float(prc),
#                         'val_accuracy': float(acc),
#                         'n_value_bets_val': int(y_pred.sum()),
#                         'val_edge_ratio_mean': val_edge_mean if _IS_LAY else np.nan,
#                         'val_edge_ratio_mean_back': val_edge_mean if _IS_BACK else np.nan,
#                     })
#
#         # pooled diagnostics
#         if val_prob_all:
#             vp = np.concatenate(val_prob_all, axis=0)
#             vt = np.concatenate(val_true_all, axis=0)
#             try: val_auc = float(roc_auc_score(vt, vp))
#             except Exception: val_auc = np.nan
#             try: val_ll  = float(log_loss(vt, vp, labels=[0, 1]))
#             except Exception: val_ll = np.nan
#             try: val_bri = float(brier_score_loss(vt, vp))
#             except Exception: val_bri = np.nan
#         else:
#             val_auc = val_ll = val_bri = np.nan
#
#         for r in rows:
#             r['val_auc'] = val_auc
#             r['val_logloss'] = val_ll
#             r['val_brier'] = val_bri
#
#         return rows
#
#     # ---------------- search ----------------
#     if base_model == "mlp":
#         eff_jobs = min(max(1, cpu_jobs), 4); prefer = "threads"; backend = "threading"; pre_dispatch = eff_jobs
#         ctx = parallel_backend(backend, n_jobs=eff_jobs)
#     else:
#         eff_jobs = max(1, min(cpu_jobs, 4)) if cpu_jobs != -1 else 4
#         prefer = "processes"; backend = "loky"; pre_dispatch = f"{2*eff_jobs}"
#         ctx = parallel_backend(backend, n_jobs=eff_jobs, inner_max_num_threads=1)
#
#     with ctx:
#         try:
#             with tqdm_joblib(tqdm(total=len(all_param_dicts), desc=f"Param search ({search_mode}, {base_model})")) as _:
#                 out = Parallel(n_jobs=eff_jobs, batch_size=1, prefer=prefer, pre_dispatch=pre_dispatch)(
#                     delayed(evaluate_param_set)(pd_) for pd_ in all_param_dicts
#                 )
#         except OSError as e:
#             print(f"[WARN] Parallel failed with {e}. Falling back to serial search...")
#             out = []
#             for pd_ in tqdm(all_param_dicts, desc=f"Param search (serial, {base_model})"):
#                 out.append(evaluate_param_set(pd_))
#
#     val_rows = [r for sub in out for r in sub]
#     if not val_rows: raise RuntimeError("No validation rows produced (check folds and input data).")
#     val_df = pd.DataFrame(val_rows)
#
#     # ---------------- validation aggregate ----------------
#     if base_model == "xgb":
#         param_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']
#     else:
#         param_keys = ['hidden_layer_sizes','alpha','learning_rate_init','batch_size','max_iter']
#
#     if _IS_CLASSIFY:
#         # include odds_min/odds_max in grouping when odds are used
#         if (classify_odds_column is not None) and (classify_odds_column in df.columns):
#             group_cols = param_keys + ['threshold','odds_min','odds_max']
#         else:
#             group_cols = param_keys + ['threshold']
#     else:
#         group_cols = param_keys + ['edge_param']
#
#     agg_dict = {
#         'n_preds_val': 'sum',
#         'tp_val': 'sum',
#         'fp_val': 'sum',
#         'val_precision': 'mean',
#         'val_accuracy': 'mean',
#         'val_auc': 'mean',
#         'val_logloss': 'mean',
#         'val_brier': 'mean',
#         'n_value_bets_val': 'sum',
#     }
#     if 'val_edge_ratio_mean' in val_df.columns: agg_dict['val_edge_ratio_mean'] = 'mean'
#     if 'val_edge_ratio_mean_back' in val_df.columns: agg_dict['val_edge_ratio_mean_back'] = 'mean'
#
#     agg = val_df.groupby(group_cols, as_index=False).agg(agg_dict)
#     agg['val_precision_pooled'] = agg.apply(lambda r: (r['tp_val'] / max(1, (r['tp_val'] + r['fp_val']))), axis=1)
#     agg['val_precision_lcb'] = agg.apply(lambda r: _wilson_lcb(int(r['tp_val']), int(r['fp_val']), conf=val_conf_level), axis=1)
#
#     qual_mask = (
#         (agg['val_precision'] >= float(precision_test_threshold)) &
#         (agg['n_preds_val'] >= int(min_samples))
#     )
#     if _USE_VALUE_LAY or _USE_VALUE_BACK:
#         qual_mask &= (agg['n_value_bets_val'] >= int(min_samples))
#     qual = agg[qual_mask].copy()
#
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     if qual.empty:
#         fail_csv = None
#         if save_diagnostics_on_fail:
#             diag = (agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                                     ascending=[False, False, False, False])
#                     .assign(fail_reason="failed_validation_gate", market=market))
#             fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv")
#             os.makedirs(csv_save_dir, exist_ok=True); diag.to_csv(fail_csv, index=False)
#         msg = "No strategy met validation gates."
#         if on_fail == "raise": raise RuntimeError(msg)
#         if on_fail == "warn": print("[WARN]", msg)
#         return {'status':'failed_validation_gate','csv':fail_csv,'model_pkl':None,
#                 'summary_df':None,'validation_table':agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                 ascending=[False,False,False,False]).reset_index(drop=True)}
#
#     ranked = qual.sort_values(by=['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                               ascending=[False, False, False, False]).reset_index(drop=True)
#     topk_val = ranked.head(top_k).reset_index(drop=True)
#
#     def _extract_params_from_row(row):
#         return cast_params({k: row[k] for k in param_keys if k in row.index})
#
#     candidates = []
#     for _, row in topk_val.iterrows():
#         c = {
#             'params': _extract_params_from_row(row),
#             'val_precision': float(row['val_precision']),
#             'val_precision_lcb': float(row['val_precision_lcb']),
#             'val_accuracy': float(row['val_accuracy']),
#             'n_preds_val': int(row['n_preds_val']),
#         }
#         if _IS_CLASSIFY:
#             c['threshold'] = float(row['threshold'])
#             c['odds_min'] = float(row['odds_min']) if 'odds_min' in row.index else np.nan
#             c['odds_max'] = float(row['odds_max']) if 'odds_max' in row.index else np.nan
#         else:
#             c['edge_param'] = float(row['edge_param'])
#         candidates.append(c)
#
#     # ---------------- test eval ----------------
#     records_all = []
#     all_bets_collector = []
#
#     def _name_cols(subdf):
#         cols = {}
#         for c in ['date','league','country','home_team','away_team','match_id']:
#             if c in subdf.columns: cols[c] = subdf[c].values
#         if {'home_team','away_team'}.issubset(subdf.columns):
#             cols['event_name'] = (subdf['home_team'] + ' v ' + subdf['away_team']).values
#         return cols
#
#     for cand_id, cand in enumerate(candidates):
#         best_params = cast_params(cand['params'])
#         pos = int(y_train_final.sum()); neg = len(y_train_final) - pos
#         spw_final = (neg/pos) if pos > 0 else 1.0
#
#         final_model = build_model(best_params, spw_final)
#         final_sample_weight = None
#         if base_model == "mlp":
#             w_pos = spw_final
#             final_sample_weight = np.where(y_train_final.values==1, w_pos, 1.0).astype(np.float32)
#
#         fit_model(final_model, X_train_final, y_train_final, X_val_final, y_val_final, sample_weight=final_sample_weight)
#         final_calibrator = fit_calibrator(final_model, X_val_final, y_val_final)
#         p_pos_test = predict_proba_pos(final_calibrator, X_test)
#
#         if _USE_VALUE:
#             # ===== VALUE modes (unchanged core) =====
#             if _IS_LAY:
#                 if market == "LAY_AWAY":
#                     p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['away_odds'].values; sel_name = 'AWAY'
#                 elif market == "LAY_HOME":
#                     p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['home_odds'].values; sel_name = 'HOME'
#                 else:
#                     p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['draw_odds'].values; sel_name = 'DRAW'
#                 fair_odds = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
#                 valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01)
#                 edge = float(cand.get('edge_param', 0.0))
#                 edge_mask = valid & (fair_odds >= (1.0 + edge) * mkt_odds)
#                 with np.errstate(divide='ignore', invalid='ignore'):
#                     edge_ratio_minus1 = np.where(mkt_odds > 0, fair_odds / mkt_odds - 1.0, 0.0)
#
#                 for plan in staking_plan_lay_options:
#                     stake = np.zeros_like(mkt_odds, dtype=float)
#                     liability = np.zeros_like(mkt_odds, dtype=float)
#                     s, L = _lay_stakes(mkt_odds, edge_ratio_minus1, plan)
#                     stake[edge_mask] = s[edge_mask]; liability[edge_mask] = L[edge_mask]
#
#                     sel_wins = (y_test.values == 0)
#                     pl = np.zeros_like(stake)
#                     idx_win = (stake > 0) & (~sel_wins)
#                     idx_lose = (stake > 0) & (sel_wins)
#                     pl[idx_win]  = stake[idx_win] * (1.0 - commission_rate)
#                     pl[idx_lose] = -liability[idx_lose]
#
#                     n_bets = int(np.count_nonzero(stake > 0))
#                     total_pl = float(pl.sum()); avg_pl = float(total_pl / max(1, n_bets))
#
#                     lays_as_preds = (stake > 0).astype(np.uint8)
#                     prc_test = precision_score(y_test, lays_as_preds, zero_division=0)
#                     acc_test = accuracy_score(y_test, lays_as_preds)
#
#                     bet_idx = np.where(stake > 0)[0]
#                     name_cols = _name_cols(df_test.iloc[bet_idx])
#                     bets_df = pd.DataFrame({
#                         **name_cols,
#                         'selection': sel_name,
#                         'market_odds': mkt_odds[bet_idx],
#                         'fair_odds': fair_odds[bet_idx],
#                         'edge_ratio': np.where(mkt_odds[bet_idx] > 0, fair_odds[bet_idx] / mkt_odds[bet_idx], np.nan),
#                         'liability': liability[bet_idx],
#                         'stake': stake[bet_idx],
#                         'commission_rate': float(commission_rate),
#                         'selection_won': sel_wins[bet_idx].astype(int),
#                         'target': y_test.values[bet_idx],
#                         'pl': pl[bet_idx],
#                     })
#                     if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
#                     bets_df['cum_pl'] = bets_df['pl'].cumsum()
#
#                     pv = _pvalue_break_even(bets_df, mode='VALUE_LAY')
#                     enough = n_bets >= int(min_test_samples)
#                     not_collapsed = prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop))
#                     pass_gate = bool(enough and not_collapsed)
#                     reason = "" if pass_gate else ("insufficient_test_bets" if not enough else "precision_collapse")
#
#                     if len(bets_df):
#                         meta = {
#                             'candidate_id': cand_id,'passed_test_gate': bool(pass_gate),'mode': 'VALUE_LAY','market': market,
#                             'threshold': np.nan,'edge_param': edge,'staking_plan_lay': plan,
#                             'val_precision': float(cand['val_precision']),'val_precision_lcb': float(cand['val_precision_lcb']),
#                             'n_value_bets_test': int(n_bets),'total_pl': float(total_pl),'avg_pl': float(avg_pl),
#                             'p_value': pv['p_value'],'zscore': pv['z'],'commission_rate': float(commission_rate),
#                             'params_json': json.dumps(best_params, default=float),
#                         }
#                         bdf = bets_df.copy()
#                         for k, v in meta.items(): bdf[k] = v
#                         all_bets_collector.append(bdf)
#
#                     records_all.append({
#                         **best_params, 'threshold': np.nan, 'odds_min': np.nan, 'odds_max': np.nan, 'edge_param': edge,
#                         'val_precision_lcb': cand['val_precision_lcb'], 'val_precision': cand['val_precision'],
#                         'val_accuracy': cand['val_accuracy'],
#                         'n_value_bets_test': n_bets, 'test_precision_bets': float(prc_test),
#                         'test_accuracy_bets': float(acc_test), 'total_pl': total_pl, 'avg_pl': avg_pl,
#                         'p_value': pv['p_value'], 'zscore': pv['z'],
#                         'pass_test_gate': pass_gate, 'fail_reason': reason, 'model_obj': final_calibrator if pass_gate else None,
#                         'mode': 'VALUE_LAY', 'bets': bets_df if pass_gate else None,
#                         'staking_plan_lay': plan,'commission_rate': float(commission_rate),
#                     })
#
#             else:  # VALUE BACK
#                 if market == "BACK_AWAY":
#                     p_sel_win = p_pos_test; mkt_odds = df_test['away_odds'].values; sel_name = 'AWAY'
#                 elif market == "BACK_HOME":
#                     p_sel_win = p_pos_test; mkt_odds = df_test['home_odds'].values; sel_name = 'HOME'
#                 else:
#                     p_sel_win = p_pos_test; mkt_odds = df_test['draw_odds'].values; sel_name = 'DRAW'
#                 fair_odds = np.divide(1.0, np.clip(p_sel_win, 1.0e-9, 1.0))
#                 valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01)
#                 edge = float(cand.get('edge_param', 0.0))
#                 edge_mask = valid & (mkt_odds >= (1.0 + edge) * fair_odds)
#                 with np.errstate(divide='ignore', invalid='ignore'):
#                     edge_ratio_minus1 = np.where(fair_odds > 0, mkt_odds / fair_odds - 1.0, 0.0)
#
#                 for plan in staking_plan_back_options:
#                     stake = np.zeros_like(mkt_odds, dtype=float)
#                     s = _back_stakes(mkt_odds, edge_ratio_minus1, plan, p_sel_win)
#                     stake[edge_mask] = s[edge_mask]
#
#                     sel_wins = (y_test.values == 1)
#                     pl = np.zeros_like(stake)
#                     win_idx = (stake > 0) & sel_wins
#                     lose_idx = (stake > 0) & (~sel_wins)
#                     pl[win_idx]  = (mkt_odds[win_idx] - 1.0) * stake[win_idx] * (1.0 - commission_rate)
#                     pl[lose_idx] = -stake[lose_idx]
#
#                     n_bets = int(np.count_nonzero(stake > 0))
#                     total_pl = float(pl.sum()); avg_pl = float(total_pl / max(1, n_bets))
#
#                     backs_as_preds = (stake > 0).astype(np.uint8)
#                     prc_test = precision_score(y_test, backs_as_preds, zero_division=0)
#                     acc_test = accuracy_score(y_test, backs_as_preds)
#
#                     bet_idx = np.where(stake > 0)[0]
#                     name_cols = _name_cols(df_test.iloc[bet_idx])
#                     bets_df = pd.DataFrame({
#                         **name_cols,
#                         'selection': sel_name,
#                         'market_odds': mkt_odds[bet_idx],
#                         'fair_odds': fair_odds[bet_idx],
#                         'edge_ratio': np.where(fair_odds[bet_idx] > 0, mkt_odds[bet_idx] / fair_odds[bet_idx], np.nan),
#                         'stake': stake[bet_idx],
#                         'commission_rate': float(commission_rate),
#                         'selection_won': sel_wins[bet_idx].astype(int),
#                         'target': y_test.values[bet_idx],
#                         'pl': pl[bet_idx],
#                     })
#                     if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
#                     bets_df['cum_pl'] = bets_df['pl'].cumsum()
#
#                     pv = _pvalue_break_even(bets_df, mode='VALUE_BACK')
#                     enough = n_bets >= int(min_test_samples)
#                     not_collapsed = prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop))
#                     pass_gate = bool(enough and not_collapsed)
#                     reason = "" if pass_gate else ("insufficient_test_bets" if not enough else "precision_collapse")
#
#                     if len(bets_df):
#                         meta = {
#                             'candidate_id': cand_id,'passed_test_gate': bool(pass_gate),'mode': 'VALUE_BACK','market': market,
#                             'threshold': np.nan,'edge_param': edge,'staking_plan_back': plan,
#                             'val_precision': float(cand['val_precision']),'val_precision_lcb': float(cand['val_precision_lcb']),
#                             'n_value_bets_test': int(n_bets),'total_pl': float(total_pl),'avg_pl': float(avg_pl),
#                             'p_value': pv['p_value'],'zscore': pv['z'],'commission_rate': float(commission_rate),
#                             'params_json': json.dumps(best_params, default=float),
#                         }
#                         bdf = bets_df.copy()
#                         for k, v in meta.items(): bdf[k] = v
#                         all_bets_collector.append(bdf)
#
#                     records_all.append({
#                         **best_params, 'threshold': np.nan, 'odds_min': np.nan, 'odds_max': np.nan, 'edge_param': edge,
#                         'val_precision_lcb': cand['val_precision_lcb'], 'val_precision': cand['val_precision'],
#                         'val_accuracy': cand['val_accuracy'],
#                         'n_value_bets_test': n_bets, 'test_precision_bets': float(prc_test),
#                         'test_accuracy_bets': float(acc_test), 'total_pl': total_pl, 'avg_pl': avg_pl,
#                         'p_value': pv['p_value'], 'zscore': pv['z'],
#                         'pass_test_gate': pass_gate, 'fail_reason': reason, 'model_obj': final_calibrator if pass_gate else None,
#                         'mode': 'VALUE_BACK', 'bets': bets_df if pass_gate else None,
#                         'staking_plan_back': plan,'commission_rate': float(commission_rate),
#                     })
#
#         else:
#             # ===== CLASSIFY TEST EVAL with odds band & bet side =====
#             thr = float(cand['threshold'])
#             if (classify_odds_column is not None) and (classify_odds_column in df_test.columns):
#                 o = df_test[classify_odds_column].values.astype(float)
#                 valid = np.isfinite(o) & (o > 1.01)
#                 omin = cand.get('odds_min', np.nan); omax = cand.get('odds_max', np.nan)
#                 if np.isnan(omin) or np.isnan(omax):
#                     # if training had no odds bands, accept all valid odds
#                     odds_mask = valid
#                     omin, omax = np.nan, np.nan
#                 else:
#                     odds_mask = valid & (o >= float(omin)) & (o <= float(omax))
#             else:
#                 o = None
#                 odds_mask = np.ones(len(X_test), dtype=bool)
#                 omin = np.nan; omax = np.nan
#
#             if classify_side == "back":
#                 pred_mask = (p_pos_test >= thr)
#             else:
#                 pred_mask = (p_pos_test <= (1.0 - thr))
#
#             take = pred_mask & odds_mask
#             y_pred = take.astype(np.uint8)
#             n_preds_test = int(y_pred.sum())
#             prc_test = precision_score(y_test, y_pred, zero_division=0)
#             acc_test = accuracy_score(y_test, y_pred)
#             enough = n_preds_test >= int(min_test_samples)
#             not_collapsed = prc_test >= max(float(precision_test_threshold),
#                                             float(cand['val_precision']) - float(max_precision_drop))
#             pass_gate = bool(enough and not_collapsed)
#             reason = "" if pass_gate else ("insufficient_test_preds" if not enough else "precision_collapse")
#
#             # Bet-level P/L + p-value
#             bets_df = None
#             total_pl = float('nan'); avg_pl = float('nan'); p_value = float('nan'); zscore = float('nan')
#
#             bet_idx = np.where(take)[0]
#             if len(bet_idx):
#                 name_cols = _name_cols(df_test.iloc[bet_idx])
#                 sel_wins = (y_test.values[bet_idx] == 1)
#                 stake = np.full(len(bet_idx), float(classify_stake), dtype=float)
#
#                 if o is not None:
#                     mkt_odds = o[bet_idx].astype(float)
#                     if classify_side == "back":
#                         pl = np.zeros_like(stake, dtype=float)
#                         win_idx = sel_wins
#                         lose_idx = ~sel_wins
#                         pl[win_idx]  = (mkt_odds[win_idx] - 1.0) * stake[win_idx] * (1.0 - commission_rate)
#                         pl[lose_idx] = -stake[lose_idx]
#                         mode_for_p = 'VALUE_BACK'
#                         extra_cols = {}
#                     else:  # lay classify
#                         liability = stake * (mkt_odds - 1.0)
#                         pl = np.zeros_like(stake, dtype=float)
#                         win_idx = ~sel_wins  # selection loses
#                         lose_idx = sel_wins  # selection wins
#                         pl[win_idx]  = stake[win_idx] * (1.0 - commission_rate)
#                         pl[lose_idx] = -liability[lose_idx]
#                         mode_for_p = 'VALUE_LAY'
#                         extra_cols = {'liability': liability}
#                     bets_df = pd.DataFrame({
#                         **name_cols,
#                         'selection': f'CLASSIFY_{classify_side.upper()}',
#                         'market_odds': mkt_odds,
#                         'stake': stake,
#                         'commission_rate': float(commission_rate),
#                         'selection_won': sel_wins.astype(int),
#                         'target': y_test.values[bet_idx],
#                         'pl': pl,
#                         'threshold': thr,
#                         'odds_min': omin, 'odds_max': omax,
#                     } | extra_cols)
#                     if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
#                     bets_df['cum_pl'] = bets_df['pl'].cumsum()
#
#                     total_pl = float(bets_df['pl'].sum())
#                     avg_pl = float(total_pl / max(1, len(bets_df)))
#                     pv = _pvalue_break_even(bets_df, mode=mode_for_p)
#                     p_value = float(pv['p_value']); zscore = float(pv['z'])
#                 else:
#                     # PSEUDO P/L only
#                     pl = np.where(sel_wins, stake, -stake)
#                     bets_df = pd.DataFrame({
#                         **name_cols,
#                         'selection': f'CLASSIFY_{classify_side.upper()}',
#                         'stake': stake,
#                         'selection_won': sel_wins.astype(int),
#                         'target': y_test.values[bet_idx],
#                         'pl': pl,
#                         'threshold': thr,
#                         'odds_min': omin, 'odds_max': omax,
#                     })
#                     if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
#                     bets_df['cum_pl'] = bets_df['pl'].cumsum()
#                     total_pl = float(bets_df['pl'].sum())
#                     avg_pl = float(total_pl / max(1, len(bets_df)))
#
#             records_all.append({
#                 **best_params,
#                 'threshold': thr,
#                 'odds_min': omin, 'odds_max': omax,
#                 'val_precision_lcb': cand['val_precision_lcb'],
#                 'val_precision': cand['val_precision'],
#                 'val_accuracy': cand['val_accuracy'],
#                 'n_preds_test': n_preds_test,
#                 'test_precision': float(prc_test),
#                 'test_accuracy': float(acc_test),
#                 'total_pl': total_pl,
#                 'avg_pl': avg_pl,
#                 'p_value': p_value,
#                 'zscore': zscore,
#                 'pass_test_gate': pass_gate,
#                 'fail_reason': reason,
#                 'model_obj': final_calibrator if pass_gate else None,
#                 'mode': f'CLASSIFY_{classify_side.upper()}',
#                 'bets': bets_df if (bets_df is not None) else None,
#             })
#
#     survivors_df = pd.DataFrame(records_all)
#     passers = survivors_df[survivors_df['pass_test_gate']].copy()
#
#     # ---------------- save / rank ----------------
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     tag = "xgb" if base_model == "xgb" else "mlp"
#
#     if passers.empty:
#         fail_csv = None
#         if save_diagnostics_on_fail:
#             if 'p_value' in survivors_df.columns and survivors_df['p_value'].notna().any():
#                 sort_cols = ['p_value','total_pl','val_precision_lcb']; asc = [True, False, False]
#             else:
#                 sort_cols = ['val_precision_lcb','val_precision','n_preds_test','val_accuracy']; asc = [False, False, False, False]
#             diag = (survivors_df.drop(columns=['model_obj','bets'], errors='ignore')
#                     .sort_values(by=sort_cols, ascending=asc)
#                     .assign(market=market))
#             fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv")
#             diag.to_csv(fail_csv, index=False); summary_df = diag
#         else:
#             summary_df = survivors_df.drop(columns=['model_obj','bets'], errors='ignore')
#
#         all_bets_csv_path = None
#         if save_all_bets_csv and ((_USE_VALUE) or (_IS_CLASSIFY and classify_odds_column is not None)) and all_bets_collector:
#             all_bets_df = pd.concat(all_bets_collector, ignore_index=True)
#             if not all_bets_include_failed:
#                 all_bets_df = all_bets_df[all_bets_df['passed_test_gate'] == True]
#             all_bets_csv_path = os.path.join(all_bets_dir, f"all_bets_{market}_{timestamp}.csv")
#             all_bets_df.to_csv(all_bets_csv_path, index=False)
#
#         if on_fail == "raise": raise RuntimeError("All Top-K failed the TEST gate.")
#         if on_fail == "warn": print("[WARN] All Top-K failed the TEST gate.")
#         return {'status':'failed_test_gate','csv':fail_csv,'model_pkl':None,
#                 'summary_df':summary_df,'validation_table':ranked,
#                 'bets_csv':None,'pl_plot':None,'all_bets_csv':all_bets_csv_path}
#
#     # Final ranking
#     if ('p_value' in passers.columns) and passers['p_value'].notna().any():
#         passers_sorted = passers.sort_values(
#             by=['p_value','total_pl','avg_pl','val_precision_lcb','val_precision'],
#             ascending=[True, False, False, False, False]
#         ).reset_index(drop=True)
#     else:
#         passers_sorted = passers.sort_values(
#             by=['val_precision_lcb','val_precision','test_precision','n_preds_test','val_accuracy'],
#             ascending=[False, False, False, False, False]
#         ).reset_index(drop=True)
#
#     # Save PKL + CSV
#     pkl_path = os.path.join(model_dir, f"best_model_{market}_{tag}_calibrated_{timestamp}.pkl")
#     csv_df = passers_sorted.drop(columns=['model_obj','bets'], errors='ignore').copy()
#     csv_df['model_pkl'] = ""; csv_df.loc[0, 'model_pkl'] = pkl_path
#     csv_df['market'] = market
#     csv_path = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}.csv")
#     csv_df.to_csv(csv_path, index=False)
#
#     # Save top model
#     top_row = passers_sorted.iloc[0]
#     chosen_model = top_row['model_obj']
#     if base_model == "xgb":
#         param_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']
#     else:
#         param_keys = ['hidden_layer_sizes','alpha','learning_rate_init','batch_size','max_iter']
#     chosen_params = {k: top_row[k] for k in param_keys if k in passers_sorted.columns}
#     chosen_threshold = float(top_row.get('threshold', np.nan))
#     chosen_edge = float(top_row.get('edge_param', np.nan))
#     chosen_odds_min = float(top_row.get('odds_min', np.nan)) if 'odds_min' in top_row.index else np.nan
#     chosen_odds_max = float(top_row.get('odds_max', np.nan)) if 'odds_max' in top_row.index else np.nan
#
#     joblib.dump(
#         {
#             'model': chosen_model,
#             'threshold': chosen_threshold,            # NaN in VALUE modes; meaningful in CLASSIFY
#             'edge_param': chosen_edge,                # chosen edge (VALUE)
#             'features': features,
#             'base_model': base_model,
#             'best_params': chosen_params,
#             'precision_test_threshold': float(precision_test_threshold),
#             'min_samples': int(min_samples),
#             'min_test_samples': int(min_test_samples),
#             'val_conf_level': float(val_conf_level),
#             'max_precision_drop': float(max_precision_drop),
#             'market': market,
#             'mode': top_row['mode'],
#             # VALUE mode staking plan that won:
#             'staking_plan_lay': top_row.get('staking_plan_lay', None) if _USE_VALUE_LAY else None,
#             'staking_plan_back': top_row.get('staking_plan_back', None) if _USE_VALUE_BACK else None,
#             # numeric staking params
#             'liability_test': float(liability_test) if _USE_VALUE_LAY else None,
#             'lay_flat_stake': float(lay_flat_stake) if _USE_VALUE_LAY else None,
#             'lay_edge_scale': float(lay_edge_scale) if _USE_VALUE_LAY else None,
#             'kelly_fraction_lay': float(kelly_fraction_lay) if _USE_VALUE_LAY else None,
#             'min_lay_stake': float(min_lay_stake) if _USE_VALUE_LAY else None,
#             'max_lay_stake': float(max_lay_stake) if _USE_VALUE_LAY else None,
#             'min_lay_liability': float(min_lay_liability) if _USE_VALUE_LAY else None,
#             'max_lay_liability': float(max_lay_liability) if _USE_VALUE_LAY else None,
#             'back_stake_test': float(back_stake_test) if _USE_VALUE_BACK else None,
#             'back_edge_scale': float(back_edge_scale) if _USE_VALUE_BACK else None,
#             'kelly_fraction_back': float(kelly_fraction_back) if _USE_VALUE_BACK else None,
#             'bankroll_back': float(bankroll_back) if _USE_VALUE_BACK else None,
#             'min_back_stake': float(min_back_stake) if _USE_VALUE_BACK else None,
#             'max_back_stake': float(max_back_stake) if _USE_VALUE_BACK else None,
#             # CLASSIFY specifics for live use:
#             'classify_stake': float(classify_stake) if _IS_CLASSIFY else None,
#             'classify_odds_column': classify_odds_column if _IS_CLASSIFY else None,
#             'classify_side': classify_side if _IS_CLASSIFY else None,
#             'classify_odds_min': chosen_odds_min if _IS_CLASSIFY else None,
#             'classify_odds_max': chosen_odds_max if _IS_CLASSIFY else None,
#             # commission saved
#             'commission_rate': float(commission_rate),
#             'notes': ('Commission applied to winning returns; '
#                       'VALUE & CLASSIFY(with-odds) ranked by smallest p-value; '
#                       'CLASSIFY sweeps threshold + odds bands + side(back/lay); '
#                       'bets CSV & cumulative P/L plot saved for the winner.'),
#             'run_seed': int(RUN_SEED),
#         },
#         pkl_path
#     )
#
#     # chosen bets CSV / plot
#     bets_path = None
#     plot_path = None
#     bets_df = top_row.get('bets', None)
#     if (save_bets_csv or plot_pl) and isinstance(bets_df, pd.DataFrame) and len(bets_df):
#         if save_bets_csv:
#             bets_name = f"bets_{market}_{timestamp}.csv"
#             bets_path = os.path.join(bets_csv_dir, bets_name)
#             bets_df.to_csv(bets_path, index=False)
#         if plot_pl:
#             try:
#                 import matplotlib.pyplot as plt
#                 fig = plt.figure()
#                 x = bets_df['date'] if 'date' in bets_df.columns else np.arange(len(bets_df))
#                 plt.plot(x, bets_df['cum_pl'])
#                 title = f"{market} cumulative P/L ({top_row['mode']})"
#                 if plot_title_suffix: title += f" — {plot_title_suffix}"
#                 plt.title(title)
#                 plt.xlabel('Date' if 'date' in bets_df.columns else 'Bet #')
#                 plt.ylabel('Cumulative P/L')
#                 plt.tight_layout()
#                 plot_name = f"cum_pl_{market}_{timestamp}.png"
#                 plot_path = os.path.join(plot_dir, plot_name)
#                 plt.savefig(plot_path, dpi=160); plt.close(fig)
#             except Exception as e:
#                 print(f"[WARN] Failed to create plot: {e}")
#
#     # ALL bets export (across all candidates) — sibling all_bets dir
#     all_bets_csv_path = None
#     if save_all_bets_csv and ((_USE_VALUE) or (_IS_CLASSIFY and classify_odds_column is not None)) and all_bets_collector:
#         all_bets_df = pd.concat(all_bets_collector, ignore_index=True)
#         if not all_bets_include_failed:
#             all_bets_df = all_bets_df[all_bets_df['passed_test_gate'] == True]
#         preferred = [c for c in [
#             'date','league','country','home_team','away_team','match_id','event_name','selection',
#             'market_odds','fair_odds','edge_ratio','stake','liability','commission_rate',
#             'selection_won','target','pl','cum_pl',
#             'candidate_id','passed_test_gate','mode','market','threshold',
#             'odds_min','odds_max','edge_param',
#             'staking_plan_lay','staking_plan_back',
#             'val_precision','val_precision_lcb','n_value_bets_test','total_pl','avg_pl','p_value','zscore','params_json'
#         ] if c in all_bets_df.columns]
#         all_bets_df = all_bets_df[preferred + [c for c in all_bets_df.columns if c not in preferred]]
#         all_bets_csv_path = os.path.join(all_bets_dir, f"all_bets_{market}_{timestamp}.csv")
#         all_bets_df.to_csv(all_bets_csv_path, index=False)
#
#     return {
#         'status': 'ok',
#         'csv': csv_path,
#         'model_pkl': pkl_path,
#         'summary_df': csv_df,
#         'validation_table': ranked,
#         'bets_csv': bets_path,
#         'pl_plot': plot_path,
#         'all_bets_csv': all_bets_csv_path,
#     }

def run_models_outcome(
    matches_filtered: pd.DataFrame,
    features: list,
    # ── gates ──────────────────────────────────────────────────────────────
    min_samples: int = 200,
    min_test_samples: int = 100,
    precision_test_threshold: float = 0.80,
    # ── model/search ───────────────────────────────────────────────────────
    base_model: str = "xgb",
    search_mode: str = "random",
    n_random_param_sets: int = 10,
    cpu_jobs: int = 6,
    top_k: int = 10,
    thresholds: np.ndarray | None = None,        # USED for CLASSIFY markets
    out_dir: str | None = None,
    # ── anti-overfitting ──────────────────────────────────────────────────
    val_conf_level: float = 0.99,
    max_precision_drop: float = 1,
    # ── failure handling ───────────────────────────────────────────────────
    on_fail: str = "return",                     # "return" | "warn" | "raise"
    save_diagnostics_on_fail: bool = True,
    # ── market ─────────────────────────────────────────────────────────────
    market: str = "LAY_AWAY",                    # LAY_* | BACK_* | OVER | UNDER (or other classify markets)

    # ── VALUE LAY controls ────────────────────────────────────────────────
    use_value_for_lay: bool = True,
    value_edge_grid_lay: np.ndarray | None = None,   # e.g. np.round(np.arange(0.00,0.201,0.01),2)

    # Staking plan search toggle + options (VALUE modes)
    enable_staking_plan_search: bool = False,
    staking_plan_lay_options: list[str] | None = None,
    staking_plan_back_options: list[str] | None = None,

    # Single-plan (used when enable_staking_plan_search=False)
    staking_plan_lay: str = "liability",             # "liability" | "flat_stake" | "edge_prop" | "kelly_approx"
    staking_plan_back: str = "flat",                 # "flat" | "edge_prop" | "kelly"

    # ── LAY staking parameters (balanced defaults) ────────────────────────
    liability_test: float = 1.0,
    lay_flat_stake: float = 1.0,
    lay_edge_scale: float = 0.05,
    kelly_fraction_lay: float = 1.0,
    min_lay_stake: float = 0.0,
    max_lay_stake: float = 1.0,
    min_lay_liability: float = 0.0,
    max_lay_liability: float = 2.0,

    # ── VALUE BACK controls ────────────────────────────────────────────────
    use_value_for_back: bool = True,
    value_edge_grid_back: np.ndarray | None = None,

    # BACK staking parameters
    back_stake_test: float = 1.0,
    back_edge_scale: float = 0.10,
    kelly_fraction_back: float = 0.25,
    bankroll_back: float = 100.0,
    min_back_stake: float = 0.0,
    max_back_stake: float = 10.0,

    # ── CLASSIFY staking / odds (adds odds-band grid sweep) ───────────────
    classify_stake: float = 1.0,                      # flat stake for BACK classify
    classify_odds_column: str | None = None,          # e.g. 'away_odds', 'over25_odds'
    classify_side: str = "back",                      # "back" or "lay" for classify bets
    classify_odds_min_grid: np.ndarray | None = None, # e.g. np.arange(1.00, 10.01, 0.25)
    classify_odds_max_grid: np.ndarray | None = None, # e.g. np.arange(1.00, 10.01, 0.25)

    # ── NEW: CLASSIFY-LAY dual staking knobs ──────────────────────────────
    classify_lay_flat_stake: float = 1.0,             # stake per bet (flat-stake variant)
    classify_lay_liability: float = 1.0,              # liability per bet (flat-liability variant)

    # ── COMMISSION (applied to net winning returns) ───────────────────────
    commission_rate: float = 0.02,  # 2% commission on winnings

    # ── OUTPUTS: chosen model ─────────────────────────────────────────────
    save_bets_csv: bool = False,
    bets_csv_dir: str | None = None,
    plot_pl: bool = False,
    plot_dir: str | None = None,
    plot_title_suffix: str = "",
    # ── OUTPUTS: ALL candidates ───────────────────────────────────────────
    save_all_bets_csv: bool = False,
    all_bets_dir: str | None = None,
    all_bets_include_failed: bool = True,
):
    """
    VALUE modes behave as before (commission-adjusted). CLASSIFY now also:
      • Sweeps thresholds
      • If `classify_odds_column` provided, sweeps (odds_min, odds_max) bands over [1.00, 10.00] step 0.25
      • Supports `classify_side="back"| "lay"`:
          - back: bet when p>=thr within odds band, P/L like back bets (flat stake)
          - lay:  bet when p<=1-thr within odds band, P/L like lay bets; evaluates BOTH flat-stake & flat-liability variants
      • Computes commission-adjusted P/L and p-value vs break-even
      • Ranks by smallest p-value (then P/L), saves bets CSV and P/L plot
    """
    # ---------------- setup ----------------
    import os, secrets, hashlib, json
    from datetime import datetime
    import numpy as np
    import pandas as pd
    from itertools import product
    from sklearn.model_selection import ParameterSampler
    from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, log_loss, brier_score_loss
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from joblib import Parallel, delayed, parallel_backend
    from tqdm import tqdm
    from tqdm_joblib import tqdm_joblib
    import joblib

    # --- xgboost import (optional)
    try:
        import xgboost as xgb
        _HAS_XGB_LOCAL = True
    except Exception:
        _HAS_XGB_LOCAL = False
    _HAS_XGB = globals().get("_HAS_XGB", _HAS_XGB_LOCAL)
    if base_model == "xgb" and not _HAS_XGB:
        raise ImportError("XGBoost not available; set base_model='mlp' or install xgboost.")

    # --- random dists
    try:
        _randint; _uniform; _loguniform
    except NameError:
        from scipy.stats import randint as _randint
        from scipy.stats import uniform as _uniform
        from scipy.stats import loguniform as _loguniform

    # --- Wilson LCB & normal CDF
    try:
        from scipy.stats import norm
        _Z = lambda conf: float(norm.ppf(1 - (1 - conf) / 2))
        _Phi = lambda z: float(norm.cdf(z))
    except Exception:
        import math
        _Z = lambda conf: 1.96 if abs(conf - 0.95) < 1e-6 else 1.64
        _Phi = lambda z: 0.5 * (1.0 + math.erf(z / (2**0.5)))

    def _wilson_lcb(tp: int, fp: int, conf: float) -> float:
        n = tp + fp
        if n <= 0: return 0.0
        p = tp / n
        z = _Z(conf)
        denom = 1.0 + (z*z)/n
        centre = p + (z*z)/(2*n)
        rad = z * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
        return max(0.0, (centre - rad) / denom)

    # defaults
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)  # CLASSIFY only
    if value_edge_grid_lay is None:
        value_edge_grid_lay = np.round(np.arange(0.00, 0.201, 0.01), 2)
    if value_edge_grid_back is None:
        value_edge_grid_back = np.round(np.arange(0.00, 0.201, 0.01), 2)
    if classify_odds_min_grid is None:
        classify_odds_min_grid = np.round(np.arange(1.00, 10.01, 0.25), 2)
    if classify_odds_max_grid is None:
        classify_odds_max_grid = np.round(np.arange(1.00, 10.01, 0.25), 2)
    classify_side = str(classify_side).lower().strip()
    if classify_side not in ("back","lay"):
        raise ValueError("classify_side must be 'back' or 'lay'")

    # normalise staking-plan options
    if staking_plan_lay_options is None:
        staking_plan_lay_options = ["liability", "flat_stake", "edge_prop", "kelly_approx"]
    if staking_plan_back_options is None:
        staking_plan_back_options = ["flat", "edge_prop", "kelly"]
    if not enable_staking_plan_search:
        staking_plan_lay_options = [staking_plan_lay]
        staking_plan_back_options = [staking_plan_back]

    # --- paths
    BASE = r"C:\Users\leere\PycharmProjects\Football_ML3\FT Results"
    PKL_DIRS = {
        "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "model_file"),
        "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "model_file"),
        "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "model_file"),
        "BACK_HOME": os.path.join(BASE, "Back_Home", "model_file"),
        "BACK_AWAY": os.path.join(BASE, "Back_Away", "model_file"),
        "BACK_DRAW": os.path.join(BASE, "Back_Draw", "model_file"),
        "OVER":      os.path.join(BASE, "Over_2_5",  "model_file"),
        "UNDER":     os.path.join(BASE, "Under_2_5", "model_file"),
    }
    CSV_DIRS = {
        "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "best_model_metrics"),
        "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "best_model_metrics"),
        "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "best_model_metrics"),
        "BACK_HOME": os.path.join(BASE, "Back_Home", "best_model_metrics"),
        "BACK_AWAY": os.path.join(BASE, "Back_Away", "best_model_metrics"),
        "BACK_DRAW": os.path.join(BASE, "Back_Draw", "best_model_metrics"),
        "OVER":      os.path.join(BASE, "Over_2_5",  "best_model_metrics"),
        "UNDER":     os.path.join(BASE, "Under_2_5", "best_model_metrics"),
    }

    market = str(market).upper().strip()
    if market not in PKL_DIRS: raise ValueError(f"Unsupported market '{market}'.")
    _IS_LAY  = market.startswith("LAY_")
    _IS_BACK = market.startswith("BACK_")
    _USE_VALUE_LAY  = bool(use_value_for_lay and _IS_LAY)
    _USE_VALUE_BACK = bool(use_value_for_back and _IS_BACK)
    _USE_VALUE = _USE_VALUE_LAY or _USE_VALUE_BACK
    _IS_CLASSIFY = not _USE_VALUE

    csv_save_dir = out_dir if (out_dir and len(str(out_dir)) > 0) else CSV_DIRS[market]
    os.makedirs(csv_save_dir, exist_ok=True)
    model_dir = PKL_DIRS[market]; os.makedirs(model_dir, exist_ok=True)
    if bets_csv_dir is None: bets_csv_dir = csv_save_dir
    if plot_dir is None: plot_dir = csv_save_dir
    os.makedirs(bets_csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    if all_bets_dir is None:
        all_bets_dir = os.path.join(os.path.dirname(CSV_DIRS[market]), "all_bets")
    os.makedirs(all_bets_dir, exist_ok=True)

    RUN_SEED = secrets.randbits(32)
    def _seed_from(*vals) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(int(RUN_SEED).to_bytes(8,'little',signed=False))
        for v in vals: h.update(str(v).encode('utf-8'))
        return int.from_bytes(h.digest(),'little') & 0x7FFFFFFF

    def _as_float(x):
        try: return float(x)
        except Exception: return float(str(x))
    def _as_int(x): return int(float(x))

    # ---------------- data ----------------
    req_cols = {'date','target'}
    if _USE_VALUE: req_cols |= {'home_odds','draw_odds','away_odds'}
    missing = req_cols - set(matches_filtered.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = matches_filtered.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    cols_needed = list(set(features) | {'target'} | ({'home_odds','draw_odds','away_odds'} if _USE_VALUE else set()))
    if _IS_CLASSIFY and classify_odds_column is not None:
        cols_needed = list(set(cols_needed) | {classify_odds_column})
    df = df.dropna(subset=cols_needed).reset_index(drop=True)

    X = df[features].copy()
    y = df['target'].astype(int).reset_index(drop=True)

    n = len(X)
    if n < max(min_samples * 3, 500): raise RuntimeError(f"Not enough rows: {n}")

    # temporal split
    test_start = int(0.85 * n)
    pretest_end = test_start
    X_test = X.iloc[test_start:].reset_index(drop=True)
    y_test = y.iloc[test_start:].reset_index(drop=True)
    df_test = df.iloc[test_start:].reset_index(drop=True)

    # rolling validation folds
    N_FOLDS = 5
    total_val_len = max(1, int(0.15 * n))
    val_len = max(1, total_val_len // N_FOLDS)
    fold_val_ends = [pretest_end - total_val_len + (i + 1) * val_len for i in range(N_FOLDS)]
    fold_val_starts = [end - val_len for end in fold_val_ends]
    if fold_val_ends:
        fold_val_ends[-1] = min(fold_val_ends[-1], pretest_end)
        fold_val_starts[-1] = max(0, fold_val_ends[-1] - val_len)

    # final small validation slice (for calibration before test)
    final_val_len = max(1, val_len)
    final_val_start = max(0, test_start - final_val_len)
    X_train_final = X.iloc[:final_val_start]
    y_train_final = y.iloc[:final_val_start]
    X_val_final   = X.iloc[final_val_start:test_start]
    y_val_final   = y.iloc[final_val_start:test_start]

    # ---------------- param spaces ----------------
    xgb_param_grid = {'n_estimators':[200],'max_depth':[5],'learning_rate':[0.1],'subsample':[0.7],
                      'colsample_bytree':[1.0],'min_child_weight':[5],'reg_lambda':[1.0]}
    xgb_param_distributions = {'n_estimators':_randint(100,1001),'max_depth':_randint(3,8),
                               'learning_rate':_loguniform(0.01,0.2),'min_child_weight':_randint(3,13),
                               'subsample':_uniform(0.7,0.3),'colsample_bytree':_uniform(0.6,0.4),
                               'reg_lambda':_loguniform(0.1,10.0)}
    mlp_param_grid = {'hidden_layer_sizes':[(128,),(256,),(128,64)],'alpha':[1e-4],
                      'learning_rate_init':[1e-3],'batch_size':['auto'],'max_iter':[200]}
    mlp_param_distributions = {'hidden_layer_sizes':[(64,),(128,),(256,),(128,64),(256,128)],
                               'alpha':_loguniform(1e-5,1e-2),'learning_rate_init':_loguniform(5e-4,5e-2),
                               'batch_size':_randint(32,257),'max_iter':_randint(150,401)}

    def cast_params(p: dict) -> dict:
        q = dict(p)
        if base_model == "xgb":
            for k in ['n_estimators','max_depth','min_child_weight']:
                if k in q: q[k] = _as_int(q[k])
            for k in ['learning_rate','subsample','colsample_bytree','reg_lambda']:
                if k in q: q[k] = _as_float(q[k])
        else:
            if 'max_iter' in q: q['max_iter'] = _as_int(q['max_iter'])
            if 'batch_size' in q and q['batch_size'] != 'auto': q['batch_size'] = _as_int(q['batch_size'])
            if 'alpha' in q: q['alpha'] = _as_float(q['alpha'])
            if 'learning_rate_init' in q: q['learning_rate_init'] = _as_float(q['learning_rate_init'])
            if 'hidden_layer_sizes' in q:
                h = q['hidden_layer_sizes']
                if isinstance(h, str):
                    parts = [pp.strip() for pp in h.strip("()").split(",") if pp.strip()!='']
                    q['hidden_layer_sizes'] = tuple(_as_int(pp) for pp in parts) if parts else (128,)
                elif isinstance(h, (list, tuple, np.ndarray)):
                    q['hidden_layer_sizes'] = tuple(int(v) for v in h)
                else:
                    q['hidden_layer_sizes'] = (int(h),)
        return q

    def _final_step_name(estimator):
        try:
            if isinstance(estimator, Pipeline): return estimator.steps[-1][0]
        except Exception:
            pass
        return None

    def build_model(params: dict, spw: float):
        model_seed = _seed_from("model", base_model, tuple(sorted(params.items())))
        if base_model == "xgb":
            return xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=model_seed,
                scale_pos_weight=spw,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
                **params
            )
        else:
            mlp = MLPClassifier(
                random_state=model_seed,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
                solver="adam",
                **params
            )
            return make_pipeline(StandardScaler(with_mean=True, with_std=True), mlp)

    def fit_model(model, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None):
        if base_model == "xgb":
            try:
                model.set_params(verbosity=0, early_stopping_rounds=50)
                if X_va is not None and y_va is not None:
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                else:
                    model.fit(X_tr, y_tr, verbose=False)
            except Exception:
                model.fit(X_tr, y_tr, verbose=False)
        else:
            fit_kwargs = {}
            if sample_weight is not None:
                stepname = _final_step_name(model)
                if stepname is not None:
                    fit_kwargs[f"{stepname}__sample_weight"] = sample_weight
            try:
                model.fit(X_tr, y_tr, **fit_kwargs)
            except TypeError:
                model.fit(X_tr, y_tr)

    def fit_calibrator(fitted, X_va, y_va):
        try:
            from sklearn.calibration import FrozenEstimator
            frozen = FrozenEstimator(fitted)
            cal = CalibratedClassifierCV(frozen, method='sigmoid', cv=None)
            cal.fit(X_va, y_va)
            return cal
        except Exception:
            try:
                cal = CalibratedClassifierCV(fitted, method='sigmoid', cv='prefit')
                cal.fit(X_va, y_va); return cal
            except Exception:
                return fitted

    def _unwrap_estimator(est):
        if isinstance(est, Pipeline): return est.steps[-1][1]
        return est

    def predict_proba_pos(model_or_cal, X_):
        proba = model_or_cal.predict_proba(X_)
        if proba.ndim == 2:
            classes = getattr(model_or_cal, "classes_", None)
            if classes is None:
                base = _unwrap_estimator(model_or_cal); classes = getattr(base, "classes_", None)
            if classes is not None and len(classes) == proba.shape[1]:
                try:
                    idx = int(np.where(np.asarray(classes) == 1)[0][0])
                    return proba[:, idx].astype(np.float32)
                except Exception:
                    pass
            if proba.shape[1] == 2: return proba[:, 1].astype(np.float32)
            if proba.shape[1] == 1:
                only = getattr(model_or_cal, "classes_", [0])[0]
                return (np.ones_like(proba[:,0]) if only==1 else np.zeros_like(proba[:,0])).astype(np.float32)
        return np.asarray(proba, dtype=np.float32)

    # --- p-value helper (commission-adjusted) ------------------------------
    def _pvalue_break_even(bdf: pd.DataFrame, mode: str) -> dict:
        if not isinstance(bdf, pd.DataFrame) or bdf.empty:
            return {'z': 0.0, 'p_value': 1.0, 'var_sum': 0.0, 'n': 0, 'total_pl': 0.0}
        o = np.asarray(bdf['market_odds'].values, dtype=float)
        o = np.where(o <= 1.0, np.nan, o)
        p = 1.0 / o  # null win prob
        if mode == 'VALUE_BACK':
            S = np.asarray(bdf['stake'].values, dtype=float)
            win = (o - 1.0) * S * (1.0 - commission_rate)
            lose = -S
        else:  # VALUE_LAY
            L = np.asarray(bdf.get('liability', np.nan*np.ones_like(o))).astype(float)
            stake = np.asarray(bdf['stake'].values, dtype=float)
            win  = stake * (1.0 - commission_rate)   # selection loses
            lose = -L                                 # selection wins
        var_i = p * (win ** 2) + (1.0 - p) * (lose ** 2)
        var_i = np.where(np.isfinite(var_i), var_i, 0.0)
        pl = np.asarray(bdf['pl'].values, dtype=float)
        total_pl = float(np.nansum(np.where(np.isfinite(pl), pl, 0.0)))
        var_sum = float(np.nansum(var_i))
        z = total_pl / (np.sqrt(var_sum) + 1e-12)
        p_val = max(0.0, 1.0 - _Phi(z))  # one-sided
        return {'z': float(z), 'p_value': float(p_val), 'var_sum': var_sum, 'n': int(len(pl)), 'total_pl': total_pl}

    def _pvalue_break_even_lay_variant(mkt_odds, stake, liability, sel_wins):
        bdf = pd.DataFrame({
            'market_odds': mkt_odds,
            'stake': stake,
            'liability': liability,
            'pl': np.where(sel_wins, -liability, stake * (1.0 - commission_rate)),
        })
        return _pvalue_break_even(bdf, mode='VALUE_LAY')

    def _lay_stakes(odds: np.ndarray, fair_over_market_minus1: np.ndarray, plan: str):
        o = np.asarray(odds, dtype=float)
        edge_plus = np.maximum(np.asarray(fair_over_market_minus1, dtype=float), 0.0)
        denom = np.maximum(o - 1.0, 1e-12)

        def _apply_joint_bounds_from_stake(stake_desired: np.ndarray):
            stake_min_joint = np.maximum(float(min_lay_stake), float(min_lay_liability) / denom)
            stake_max_joint = np.minimum(float(max_lay_stake), float(max_lay_liability) / denom)
            stake = np.clip(stake_desired, stake_min_joint, stake_max_joint)
            L = stake * denom
            return stake, L

        def _apply_joint_bounds_from_liability(L_desired: np.ndarray):
            L_min_joint = np.maximum(float(min_lay_liability), float(min_lay_stake) * denom)
            L_max_joint = np.minimum(float(max_lay_liability), float(max_lay_stake) * denom)
            L = np.clip(L_desired, L_min_joint, L_max_joint)
            stake = L / denom
            return stake, L

        if plan == "liability":
            L_desired = np.full_like(o, float(liability_test), dtype=float)
            stake, L = _apply_joint_bounds_from_liability(L_desired)
        elif plan == "flat_stake":
            stake_desired = np.full_like(o, float(lay_flat_stake), dtype=float)
            stake, L = _apply_joint_bounds_from_stake(stake_desired)
        elif plan == "edge_prop":
            scale = max(1e-12, float(lay_edge_scale))
            L_desired = float(liability_test) * (edge_plus / scale)
            stake, L = _apply_joint_bounds_from_liability(L_desired)
        elif plan == "kelly_approx":
            L_desired = float(liability_test) * float(kelly_fraction_lay) * edge_plus
            stake, L = _apply_joint_bounds_from_liability(L_desired)
        else:
            raise ValueError(f"Unknown staking_plan_lay: {plan}")

        stake = np.where(np.isfinite(stake), np.maximum(stake, 0.0), 0.0)
        L = np.where(np.isfinite(L), np.maximum(L, 0.0), 0.0)
        return stake, L

    def _back_stakes(odds: np.ndarray, fair_over_market_minus1: np.ndarray, plan: str, p_win: np.ndarray):
        o = np.asarray(odds, dtype=float)
        p = np.clip(np.asarray(p_win, dtype=float), 0.0, 1.0)
        edge_plus = np.maximum(fair_over_market_minus1, 0.0)
        if plan == "flat":
            stake = np.full_like(o, float(back_stake_test), dtype=float)
        elif plan == "edge_prop":
            stake = float(back_stake_test) * np.divide(edge_plus, max(1e-9, float(back_edge_scale)))
        elif plan == "kelly":
            b = np.maximum(o - 1.0, 1e-9)
            f = (b * p - (1.0 - p)) / b
            f = np.maximum(f, 0.0)
            stake = float(bankroll_back) * float(kelly_fraction_back) * f
        else:
            raise ValueError(f"Unknown staking_plan_back: {plan}")
        stake = np.clip(stake, float(min_back_stake), float(max_back_stake))
        return stake

    # ---------------- search space ----------------
    if search_mode.lower() == "grid":
        grid = xgb_param_grid if base_model == "xgb" else mlp_param_grid
        all_param_dicts = [dict(zip(grid.keys(), combo)) for combo in product(*grid.values())]
    else:
        dists = xgb_param_distributions if base_model == "xgb" else mlp_param_distributions
        sampler_seed = _seed_from("sampler")
        all_param_dicts = list(ParameterSampler(dists, n_iter=n_random_param_sets, random_state=sampler_seed))

    # ---------------- validation eval ----------------
    def evaluate_param_set(param_dict, *_):
        safe = cast_params(param_dict)
        rows = []; val_prob_all=[]; val_true_all=[]

        for vstart, vend in zip(fold_val_starts, fold_val_ends):
            if vstart is None or vend is None or vstart <= 0 or vend <= vstart: continue
            X_tr, y_tr = X.iloc[:vstart], y.iloc[:vstart]
            X_va, y_va = X.iloc[vstart:vend], y.iloc[vstart:vend]
            df_va = df.iloc[vstart:vend]
            if y_tr.nunique() < 2: continue

            pos = int(y_tr.sum()); neg = len(y_tr) - pos
            spw = (neg/pos) if pos > 0 else 1.0

            sample_weight = None
            if base_model == "mlp":
                w_pos = spw
                sample_weight = np.where(y_tr.values==1, w_pos, 1.0).astype(np.float32)

            model = build_model(safe, spw)
            fit_model(model, X_tr, y_tr, X_va, y_va, sample_weight=sample_weight)
            cal = fit_calibrator(model, X_va, y_va)

            p_pos = predict_proba_pos(cal, X_va)
            val_prob_all.append(p_pos)
            y_true = y_va.values.astype(np.uint8); val_true_all.append(y_true)

            if _IS_CLASSIFY:
                if classify_odds_column is None or classify_odds_column not in df_va.columns:
                    for thr in thresholds:
                        thr = float(thr)
                        take = (p_pos >= thr) if classify_side == "back" else (p_pos <= (1.0 - thr))
                        y_pred = (take).astype(np.uint8)
                        n_preds = int(y_pred.sum())
                        tp = int(((y_true == 1) & (y_pred == 1)).sum())
                        fp = int(((y_true == 0) & (y_pred == 1)).sum())
                        prc = precision_score(y_va, y_pred, zero_division=0)
                        acc = accuracy_score(y_va, y_pred)
                        rows.append({
                            **safe,
                            'threshold': thr,
                            'odds_min': np.nan, 'odds_max': np.nan,
                            'fold_vstart': int(vstart),
                            'fold_vend': int(vend),
                            'n_preds_val': n_preds,
                            'tp_val': tp,
                            'fp_val': fp,
                            'val_precision': float(prc),
                            'val_accuracy': float(acc),
                            'n_value_bets_val': n_preds,
                            'val_edge_ratio_mean': np.nan,
                            'val_edge_ratio_mean_back': np.nan,
                        })
                else:
                    mkt = df_va[classify_odds_column].values.astype(float)
                    valid = np.isfinite(mkt) & (mkt > 1.01)
                    for thr in thresholds:
                        thr = float(thr)
                        pred_mask = (p_pos >= thr) if classify_side == "back" else (p_pos <= (1.0 - thr))
                        for omin in classify_odds_min_grid:
                            for omax in classify_odds_max_grid:
                                omin = float(omin); omax = float(omax)
                                if omin > omax: continue
                                odds_mask = valid & (mkt >= omin) & (mkt <= omax)
                                take = pred_mask & odds_mask
                                y_pred = take.astype(np.uint8)
                                n_preds = int(y_pred.sum())
                                tp = int(((y_true == 1) & (y_pred == 1)).sum())
                                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                                prc = precision_score(y_va, y_pred, zero_division=0)
                                acc = accuracy_score(y_va, y_pred)
                                rows.append({
                                    **safe,
                                    'threshold': thr,
                                    'odds_min': omin, 'odds_max': omax,
                                    'fold_vstart': int(vstart),
                                    'fold_vend': int(vend),
                                    'n_preds_val': n_preds,
                                    'tp_val': tp,
                                    'fp_val': fp,
                                    'val_precision': float(prc),
                                    'val_accuracy': float(acc),
                                    'n_value_bets_val': n_preds,
                                    'val_edge_ratio_mean': np.nan,
                                    'val_edge_ratio_mean_back': np.nan,
                                })
            else:
                # VALUE modes
                if _IS_LAY:
                    mkt = df_va['away_odds'].values if market=="LAY_AWAY" else (df_va['home_odds'].values if market=="LAY_HOME" else df_va['draw_odds'].values)
                    p_sel_win = 1.0 - p_pos
                    fair = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
                else:
                    mkt = df_va['away_odds'].values if market=="BACK_AWAY" else (df_va['home_odds'].values if market=="BACK_HOME" else df_va['draw_odds'].values)
                    p_sel_win = p_pos
                    fair = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
                edge_grid = value_edge_grid_lay if _IS_LAY else value_edge_grid_back
                for edge_param in edge_grid:
                    if _IS_LAY:
                        edge_mask = (fair >= (1.0 + float(edge_param)) * mkt) & np.isfinite(mkt)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            edge_ratio = fair / mkt
                        val_edge_mean = float(np.nanmean(np.where(np.isfinite(edge_ratio), edge_ratio, np.nan)))
                    else:
                        edge_mask = (mkt >= (1.0 + float(edge_param)) * fair) & np.isfinite(mkt)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            edge_ratio = mkt / fair
                        val_edge_mean = float(np.nanmean(np.where(np.isfinite(edge_ratio), edge_ratio, np.nan)))
                    y_pred = edge_mask.astype(np.uint8)
                    tp = int(((y_true == 1) & (y_pred == 1)).sum())
                    fp = int(((y_true == 0) & (y_pred == 1)).sum())
                    prc = precision_score(y_true, y_pred, zero_division=0)
                    acc = accuracy_score(y_true, y_pred)
                    rows.append({
                        **safe,
                        'threshold': np.nan,
                        'odds_min': np.nan, 'odds_max': np.nan,
                        'edge_param': float(edge_param),
                        'fold_vstart': int(vstart),
                        'fold_vend': int(vend),
                        'n_preds_val': int(y_pred.sum()),
                        'tp_val': tp,
                        'fp_val': fp,
                        'val_precision': float(prc),
                        'val_accuracy': float(acc),
                        'n_value_bets_val': int(y_pred.sum()),
                        'val_edge_ratio_mean': val_edge_mean if _IS_LAY else np.nan,
                        'val_edge_ratio_mean_back': val_edge_mean if _IS_BACK else np.nan,
                    })

        # pooled diagnostics
        if val_prob_all:
            vp = np.concatenate(val_prob_all, axis=0)
            vt = np.concatenate(val_true_all, axis=0)
            try: val_auc = float(roc_auc_score(vt, vp))
            except Exception: val_auc = np.nan
            try: val_ll  = float(log_loss(vt, vp, labels=[0, 1]))
            except Exception: val_ll = np.nan
            try: val_bri = float(brier_score_loss(vt, vp))
            except Exception: val_bri = np.nan
        else:
            val_auc = val_ll = val_bri = np.nan

        for r in rows:
            r['val_auc'] = val_auc
            r['val_logloss'] = val_ll
            r['val_brier'] = val_bri

        return rows

    # ---------------- search ----------------
    if base_model == "mlp":
        eff_jobs = min(max(1, cpu_jobs), 4); prefer = "threads"; backend = "threading"; pre_dispatch = eff_jobs
        ctx = parallel_backend(backend, n_jobs=eff_jobs)
    else:
        eff_jobs = max(1, min(cpu_jobs, 4)) if cpu_jobs != -1 else 4
        prefer = "processes"; backend = "loky"; pre_dispatch = f"{2*eff_jobs}"
        ctx = parallel_backend(backend, n_jobs=eff_jobs, inner_max_num_threads=1)

    with ctx:
        try:
            with tqdm_joblib(tqdm(total=len(all_param_dicts), desc=f"Param search ({search_mode}, {base_model})")) as _:
                out = Parallel(n_jobs=eff_jobs, batch_size=1, prefer=prefer, pre_dispatch=pre_dispatch)(
                    delayed(evaluate_param_set)(pd_) for pd_ in all_param_dicts
                )
        except OSError as e:
            print(f"[WARN] Parallel failed with {e}. Falling back to serial search...")
            out = []
            for pd_ in tqdm(all_param_dicts, desc=f"Param search (serial, {base_model})"):
                out.append(evaluate_param_set(pd_))

    val_rows = [r for sub in out for r in sub]
    if not val_rows: raise RuntimeError("No validation rows produced (check folds and input data).")
    val_df = pd.DataFrame(val_rows)

    # ---------------- validation aggregate ----------------
    if base_model == "xgb":
        param_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']
    else:
        param_keys = ['hidden_layer_sizes','alpha','learning_rate_init','batch_size','max_iter']

    if _IS_CLASSIFY:
        if (classify_odds_column is not None) and (classify_odds_column in df.columns):
            group_cols = param_keys + ['threshold','odds_min','odds_max']
        else:
            group_cols = param_keys + ['threshold']
    else:
        group_cols = param_keys + ['edge_param']

    agg_dict = {
        'n_preds_val': 'sum',
        'tp_val': 'sum',
        'fp_val': 'sum',
        'val_precision': 'mean',
        'val_accuracy': 'mean',
        'val_auc': 'mean',
        'val_logloss': 'mean',
        'val_brier': 'mean',
        'n_value_bets_val': 'sum',
    }
    if 'val_edge_ratio_mean' in val_df.columns: agg_dict['val_edge_ratio_mean'] = 'mean'
    if 'val_edge_ratio_mean_back' in val_df.columns: agg_dict['val_edge_ratio_mean_back'] = 'mean'

    agg = val_df.groupby(group_cols, as_index=False).agg(agg_dict)
    agg['val_precision_pooled'] = agg.apply(lambda r: (r['tp_val'] / max(1, (r['tp_val'] + r['fp_val']))), axis=1)
    agg['val_precision_lcb'] = agg.apply(lambda r: _wilson_lcb(int(r['tp_val']), int(r['fp_val']), conf=val_conf_level), axis=1)

    qual_mask = (
        (agg['val_precision'] >= float(precision_test_threshold)) &
        (agg['n_preds_val'] >= int(min_samples))
    )
    if _USE_VALUE_LAY or _USE_VALUE_BACK:
        qual_mask &= (agg['n_value_bets_val'] >= int(min_samples))
    qual = agg[qual_mask].copy()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if qual.empty:
        fail_csv = None
        if save_diagnostics_on_fail:
            diag = (agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
                                    ascending=[False, False, False, False])
                    .assign(fail_reason="failed_validation_gate", market=market))
            fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv")
            os.makedirs(csv_save_dir, exist_ok=True); diag.to_csv(fail_csv, index=False)
        msg = "No strategy met validation gates."
        if on_fail == "raise": raise RuntimeError(msg)
        if on_fail == "warn": print("[WARN]", msg)
        return {'status':'failed_validation_gate','csv':fail_csv,'model_pkl':None,
                'summary_df':None,'validation_table':agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
                ascending=[False,False,False,False]).reset_index(drop=True)}

    ranked = qual.sort_values(by=['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
                              ascending=[False, False, False, False]).reset_index(drop=True)
    topk_val = ranked.head(top_k).reset_index(drop=True)

    def _extract_params_from_row(row):
        return cast_params({k: row[k] for k in param_keys if k in row.index})

    candidates = []
    for _, row in topk_val.iterrows():
        c = {
            'params': _extract_params_from_row(row),
            'val_precision': float(row['val_precision']),
            'val_precision_lcb': float(row['val_precision_lcb']),
            'val_accuracy': float(row['val_accuracy']),
            'n_preds_val': int(row['n_preds_val']),
        }
        if _IS_CLASSIFY:
            c['threshold'] = float(row['threshold'])
            c['odds_min'] = float(row['odds_min']) if 'odds_min' in row.index else np.nan
            c['odds_max'] = float(row['odds_max']) if 'odds_max' in row.index else np.nan
        else:
            c['edge_param'] = float(row['edge_param'])
        candidates.append(c)

    # ---------------- test eval ----------------
    records_all = []
    all_bets_collector = []

    def _name_cols(subdf):
        cols = {}
        for c in ['date','league','country','home_team','away_team','match_id']:
            if c in subdf.columns: cols[c] = subdf[c].values
        if {'home_team','away_team'}.issubset(subdf.columns):
            cols['event_name'] = (subdf['home_team'] + ' v ' + subdf['away_team']).values
        return cols

    for cand_id, cand in enumerate(candidates):
        best_params = cast_params(cand['params'])
        pos = int(y_train_final.sum()); neg = len(y_train_final) - pos
        spw_final = (neg/pos) if pos > 0 else 1.0

        final_model = build_model(best_params, spw_final)
        final_sample_weight = None
        if base_model == "mlp":
            w_pos = spw_final
            final_sample_weight = np.where(y_train_final.values==1, w_pos, 1.0).astype(np.float32)

        fit_model(final_model, X_train_final, y_train_final, X_val_final, y_val_final, sample_weight=final_sample_weight)
        final_calibrator = fit_calibrator(final_model, X_val_final, y_val_final)
        p_pos_test = predict_proba_pos(final_calibrator, X_test)

        if _USE_VALUE:
            # ===== VALUE modes (unchanged) =====
            if _IS_LAY:
                if market == "LAY_AWAY":
                    p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['away_odds'].values; sel_name = 'AWAY'
                elif market == "LAY_HOME":
                    p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['home_odds'].values; sel_name = 'HOME'
                else:
                    p_sel_win = 1.0 - p_pos_test; mkt_odds = df_test['draw_odds'].values; sel_name = 'DRAW'
                fair_odds = np.divide(1.0, np.clip(p_sel_win, 1e-9, 1.0))
                valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01)
                edge = float(cand.get('edge_param', 0.0))
                edge_mask = valid & (fair_odds >= (1.0 + edge) * mkt_odds)
                with np.errstate(divide='ignore', invalid='ignore'):
                    edge_ratio_minus1 = np.where(mkt_odds > 0, fair_odds / mkt_odds - 1.0, 0.0)

                for plan in staking_plan_lay_options:
                    stake = np.zeros_like(mkt_odds, dtype=float)
                    liability = np.zeros_like(mkt_odds, dtype=float)
                    s, L = _lay_stakes(mkt_odds, edge_ratio_minus1, plan)
                    stake[edge_mask] = s[edge_mask]; liability[edge_mask] = L[edge_mask]

                    sel_wins = (y_test.values == 0)
                    pl = np.zeros_like(stake)
                    idx_win = (stake > 0) & (~sel_wins)
                    idx_lose = (stake > 0) & (sel_wins)
                    pl[idx_win]  = stake[idx_win] * (1.0 - commission_rate)
                    pl[idx_lose] = -liability[idx_lose]

                    n_bets = int(np.count_nonzero(stake > 0))
                    total_pl = float(pl.sum()); avg_pl = float(total_pl / max(1, n_bets))

                    lays_as_preds = (stake > 0).astype(np.uint8)
                    prc_test = precision_score(y_test, lays_as_preds, zero_division=0)
                    acc_test = accuracy_score(y_test, lays_as_preds)

                    bet_idx = np.where(stake > 0)[0]
                    name_cols = _name_cols(df_test.iloc[bet_idx])
                    bets_df = pd.DataFrame({
                        **name_cols,
                        'selection': sel_name,
                        'market_odds': mkt_odds[bet_idx],
                        'fair_odds': fair_odds[bet_idx],
                        'edge_ratio': np.where(mkt_odds[bet_idx] > 0, fair_odds[bet_idx] / mkt_odds[bet_idx], np.nan),
                        'liability': liability[bet_idx],
                        'stake': stake[bet_idx],
                        'commission_rate': float(commission_rate),
                        'selection_won': sel_wins[bet_idx].astype(int),
                        'target': y_test.values[bet_idx],
                        'pl': pl[bet_idx],
                    })
                    if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
                    bets_df['cum_pl'] = bets_df['pl'].cumsum()

                    pv = _pvalue_break_even(bets_df, mode='VALUE_LAY')
                    enough = n_bets >= int(min_test_samples)
                    not_collapsed = prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop))
                    pass_gate = bool(enough and not_collapsed)
                    reason = "" if pass_gate else ("insufficient_test_bets" if not enough else "precision_collapse")

                    if len(bets_df):
                        meta = {
                            'candidate_id': cand_id,'passed_test_gate': bool(pass_gate),'mode': 'VALUE_LAY','market': market,
                            'threshold': np.nan,'edge_param': edge,'staking_plan_lay': plan,
                            'val_precision': float(cand['val_precision']),'val_precision_lcb': float(cand['val_precision_lcb']),
                            'n_value_bets_test': int(n_bets),'total_pl': float(total_pl),'avg_pl': float(avg_pl),
                            'p_value': pv['p_value'],'zscore': pv['z'],'commission_rate': float(commission_rate),
                            'params_json': json.dumps(best_params, default=float),
                        }
                        bdf = bets_df.copy()
                        for k, v in meta.items(): bdf[k] = v
                        all_bets_collector.append(bdf)

                    records_all.append({
                        **best_params, 'threshold': np.nan, 'odds_min': np.nan, 'odds_max': np.nan, 'edge_param': edge,
                        'val_precision_lcb': cand['val_precision_lcb'], 'val_precision': cand['val_precision'],
                        'val_accuracy': cand['val_accuracy'],
                        'n_value_bets_test': n_bets, 'test_precision_bets': float(prc_test),
                        'test_accuracy_bets': float(acc_test), 'total_pl': total_pl, 'avg_pl': avg_pl,
                        'p_value': pv['p_value'], 'zscore': pv['z'],
                        'pass_test_gate': pass_gate, 'fail_reason': reason, 'model_obj': final_calibrator if pass_gate else None,
                        'mode': 'VALUE_LAY', 'bets': bets_df if pass_gate else None,
                        'staking_plan_lay': plan,'commission_rate': float(commission_rate),
                    })

            else:  # VALUE BACK
                if market == "BACK_AWAY":
                    p_sel_win = p_pos_test; mkt_odds = df_test['away_odds'].values; sel_name = 'AWAY'
                elif market == "BACK_HOME":
                    p_sel_win = p_pos_test; mkt_odds = df_test['home_odds'].values; sel_name = 'HOME'
                else:
                    p_sel_win = p_pos_test; mkt_odds = df_test['draw_odds'].values; sel_name = 'DRAW'
                fair_odds = np.divide(1.0, np.clip(p_sel_win, 1.0e-9, 1.0))
                valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01)
                edge = float(cand.get('edge_param', 0.0))
                edge_mask = valid & (mkt_odds >= (1.0 + edge) * fair_odds)
                with np.errstate(divide='ignore', invalid='ignore'):
                    edge_ratio_minus1 = np.where(fair_odds > 0, mkt_odds / fair_odds - 1.0, 0.0)

                for plan in staking_plan_back_options:
                    stake = np.zeros_like(mkt_odds, dtype=float)
                    s = _back_stakes(mkt_odds, edge_ratio_minus1, plan, p_sel_win)
                    stake[edge_mask] = s[edge_mask]

                    sel_wins = (y_test.values == 1)
                    pl = np.zeros_like(stake)
                    win_idx = (stake > 0) & sel_wins
                    lose_idx = (stake > 0) & (~sel_wins)
                    pl[win_idx]  = (mkt_odds[win_idx] - 1.0) * stake[win_idx] * (1.0 - commission_rate)
                    pl[lose_idx] = -stake[lose_idx]

                    n_bets = int(np.count_nonzero(stake > 0))
                    total_pl = float(pl.sum()); avg_pl = float(total_pl / max(1, n_bets))

                    backs_as_preds = (stake > 0).astype(np.uint8)
                    prc_test = precision_score(y_test, backs_as_preds, zero_division=0)
                    acc_test = accuracy_score(y_test, backs_as_preds)

                    bet_idx = np.where(stake > 0)[0]
                    name_cols = _name_cols(df_test.iloc[bet_idx])
                    bets_df = pd.DataFrame({
                        **name_cols,
                        'selection': sel_name,
                        'market_odds': mkt_odds[bet_idx],
                        'fair_odds': fair_odds[bet_idx],
                        'edge_ratio': np.where(fair_odds[bet_idx] > 0, mkt_odds[bet_idx] / fair_odds[bet_idx], np.nan),
                        'stake': stake[bet_idx],
                        'commission_rate': float(commission_rate),
                        'selection_won': sel_wins[bet_idx].astype(int),
                        'target': y_test.values[bet_idx],
                        'pl': pl[bet_idx],
                    })
                    if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
                    bets_df['cum_pl'] = bets_df['pl'].cumsum()

                    pv = _pvalue_break_even(bets_df, mode='VALUE_BACK')
                    enough = n_bets >= int(min_test_samples)
                    not_collapsed = prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop))
                    pass_gate = bool(enough and not_collapsed)
                    reason = "" if pass_gate else ("insufficient_test_bets" if not enough else "precision_collapse")

                    if len(bets_df):
                        meta = {
                            'candidate_id': cand_id,'passed_test_gate': bool(pass_gate),'mode': 'VALUE_BACK','market': market,
                            'threshold': np.nan,'edge_param': edge,'staking_plan_back': plan,
                            'val_precision': float(cand['val_precision']),'val_precision_lcb': float(cand['val_precision_lcb']),
                            'n_value_bets_test': int(n_bets),'total_pl': float(total_pl),'avg_pl': float(avg_pl),
                            'p_value': pv['p_value'],'zscore': pv['z'],'commission_rate': float(commission_rate),
                            'params_json': json.dumps(best_params, default=float),
                        }
                        bdf = bets_df.copy()
                        for k, v in meta.items(): bdf[k] = v
                        all_bets_collector.append(bdf)

                    records_all.append({
                        **best_params, 'threshold': np.nan, 'odds_min': np.nan, 'odds_max': np.nan, 'edge_param': edge,
                        'val_precision_lcb': cand['val_precision_lcb'], 'val_precision': cand['val_precision'],
                        'val_accuracy': cand['val_accuracy'],
                        'n_value_bets_test': n_bets, 'test_precision_bets': float(prc_test),
                        'test_accuracy_bets': float(acc_test), 'total_pl': total_pl, 'avg_pl': avg_pl,
                        'p_value': pv['p_value'], 'zscore': pv['z'],
                        'pass_test_gate': pass_gate, 'fail_reason': reason, 'model_obj': final_calibrator if pass_gate else None,
                        'mode': 'VALUE_BACK', 'bets': bets_df if pass_gate else None,
                        'staking_plan_back': plan,'commission_rate': float(commission_rate),
                    })

        else:
            # ===== CLASSIFY TEST EVAL with odds band & bet side =====
            thr = float(cand['threshold'])
            if (classify_odds_column is not None) and (classify_odds_column in df_test.columns):
                o = df_test[classify_odds_column].values.astype(float)
                valid = np.isfinite(o) & (o > 1.01)
                omin = cand.get('odds_min', np.nan); omax = cand.get('odds_max', np.nan)
                if np.isnan(omin) or np.isnan(omax):
                    odds_mask = valid
                    omin, omax = np.nan, np.nan
                else:
                    odds_mask = valid & (o >= float(omin)) & (o <= float(omax))
            else:
                o = None
                odds_mask = np.ones(len(X_test), dtype=bool)
                omin = np.nan; omax = np.nan

            pred_mask = (p_pos_test >= thr) if classify_side == "back" else (p_pos_test <= (1.0 - thr))
            take = pred_mask & odds_mask
            y_pred = take.astype(np.uint8)
            n_preds_test = int(y_pred.sum())
            prc_test = precision_score(y_test, y_pred, zero_division=0)
            acc_test = accuracy_score(y_test, y_pred)
            enough = n_preds_test >= int(min_test_samples)
            not_collapsed = prc_test >= max(float(precision_test_threshold),
                                            float(cand['val_precision']) - float(max_precision_drop))
            pass_gate = bool(enough and not_collapsed)
            reason = "" if pass_gate else ("insufficient_test_preds" if not enough else "precision_collapse")

            # Bet-level P/L + p-value
            bets_df = None
            total_pl = float('nan'); avg_pl = float('nan'); p_value = float('nan'); zscore = float('nan')

            bet_idx = np.where(take)[0]
            if len(bet_idx):
                name_cols = _name_cols(df_test.iloc[bet_idx])
                sel_wins = (y_test.values[bet_idx] == 1)   # selection WON

                if o is not None:
                    mkt_odds = o[bet_idx].astype(float)
                    base_cols = {
                        **name_cols,
                        'selection': f'CLASSIFY_{classify_side.upper()}',
                        'market_odds': mkt_odds,
                        'threshold': thr,
                        'odds_min': omin, 'odds_max': omax,
                        'commission_rate': float(commission_rate),
                        'selection_won': sel_wins.astype(int),
                        'target': y_test.values[bet_idx],
                    }

                    if classify_side == "back":
                        stake_flat = np.full(len(bet_idx), float(classify_stake), dtype=float)
                        pl_flat = np.zeros_like(stake_flat, dtype=float)
                        win_idx = sel_wins
                        lose_idx = ~sel_wins
                        pl_flat[win_idx]  = (mkt_odds[win_idx] - 1.0) * stake_flat[win_idx] * (1.0 - commission_rate)
                        pl_flat[lose_idx] = -stake_flat[lose_idx]

                        bets_df = pd.DataFrame({
                            **base_cols,
                            'stake': stake_flat,
                            'pl': pl_flat,
                        })
                        if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
                        bets_df['cum_pl'] = bets_df['pl'].cumsum()

                        pv = _pvalue_break_even(bets_df[['market_odds','stake','pl']], mode='VALUE_BACK')
                        total_pl = float(bets_df['pl'].sum()); avg_pl = float(total_pl / max(1, len(bets_df)))
                        p_value = float(pv['p_value']); zscore = float(pv['z'])

                    else:
                        # LAY CLASSIFY: evaluate BOTH variants
                        # A) Flat-stake variant
                        stake_flat = np.full(len(bet_idx), float(classify_lay_flat_stake), dtype=float)
                        liability_flat = stake_flat * (mkt_odds - 1.0)
                        pl_flat = np.where(sel_wins, -liability_flat, stake_flat * (1.0 - commission_rate))
                        pv_flat = _pvalue_break_even_lay_variant(mkt_odds, stake_flat, liability_flat, sel_wins)

                        # B) Flat-liability variant
                        liability_const = np.full(len(bet_idx), float(classify_lay_liability), dtype=float)
                        denom = np.maximum(mkt_odds - 1.0, 1e-12)
                        stake_liab = liability_const / denom
                        pl_liab = np.where(sel_wins, -liability_const, stake_liab * (1.0 - commission_rate))
                        pv_liab = _pvalue_break_even_lay_variant(mkt_odds, stake_liab, liability_const, sel_wins)

                        bets_df = pd.DataFrame({
                            **base_cols,
                            # Flat-stake variant
                            'stake_flat': stake_flat,
                            'liability_flat': liability_flat,
                            'pl_flat': pl_flat,
                            # Flat-liability variant
                            'stake_liability': stake_liab,
                            'liability_liability': liability_const,
                            'pl_liability': pl_liab,
                        })
                        if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
                        bets_df['cum_pl_flat'] = bets_df['pl_flat'].cumsum()
                        bets_df['cum_pl_liability'] = bets_df['pl_liability'].cumsum()
                        # For plotting/back-compat: expose a 'cum_pl' using flat-stake
                        bets_df['cum_pl'] = bets_df['cum_pl_flat']

                        # Legacy totals use the flat-stake variant (back-compat)
                        total_pl = float(np.sum(pl_flat))
                        avg_pl = float(total_pl / max(1, len(bets_df)))
                        p_value = float(pv_flat['p_value']); zscore = float(pv_flat['z'])

                        # Attach aggregates for both variants (useful in CSV/summary)
                        bets_df.attrs['totals'] = {
                            'total_pl_flat': float(np.sum(pl_flat)),
                            'avg_pl_flat': float(np.mean(pl_flat)) if len(pl_flat) else float('nan'),
                            'p_value_flat': float(pv_flat['p_value']),
                            'zscore_flat': float(pv_flat['z']),
                            'total_pl_liability': float(np.sum(pl_liab)),
                            'avg_pl_liability': float(np.mean(pl_liab)) if len(pl_liab) else float('nan'),
                            'p_value_liability': float(pv_liab['p_value']),
                            'zscore_liability': float(pv_liab['z']),
                        }

                else:
                    # PSEUDO P/L
                    stake = np.full(len(bet_idx), float(classify_stake), dtype=float)
                    if classify_side == "back":
                        pl = np.where(sel_wins, stake, -stake)
                    else:
                        pl = np.where(sel_wins, -stake, stake)
                    bets_df = pd.DataFrame({
                        **name_cols,
                        'selection': f'CLASSIFY_{classify_side.upper()}',
                        'stake_pseudo': stake,
                        'pl_pseudo': pl,
                        'threshold': thr,
                        'odds_min': omin, 'odds_max': omax,
                    })
                    if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
                    bets_df['cum_pl_pseudo'] = bets_df['pl_pseudo'].cumsum()
                    total_pl = float(bets_df['pl_pseudo'].sum())
                    avg_pl = float(total_pl / max(1, len(bets_df)))

            # record
            records = {
                **best_params,
                'threshold': thr,
                'odds_min': omin, 'odds_max': omax,
                'val_precision_lcb': cand['val_precision_lcb'],
                'val_precision': cand['val_precision'],
                'val_accuracy': cand['val_accuracy'],
                'n_preds_test': n_preds_test,
                'test_precision': float(prc_test),
                'test_accuracy': float(acc_test),
                'total_pl': total_pl,
                'avg_pl': avg_pl,
                'p_value': p_value,
                'zscore': zscore,
                'pass_test_gate': pass_gate,
                'fail_reason': reason,
                'model_obj': final_calibrator if pass_gate else None,
                'mode': f'CLASSIFY_{classify_side.upper()}',
                'bets': bets_df if (bets_df is not None) else None,
            }

            # include both LAY variants' aggregates if present
            if (bets_df is not None) and ('totals' in getattr(bets_df, 'attrs', {})) and (classify_side == 'lay'):
                t = bets_df.attrs['totals']
                records.update({
                    'total_pl_flat': t['total_pl_flat'],
                    'avg_pl_flat': t['avg_pl_flat'],
                    'p_value_flat': t['p_value_flat'],
                    'zscore_flat': t['zscore_flat'],
                    'total_pl_liability': t['total_pl_liability'],
                    'avg_pl_liability': t['avg_pl_liability'],
                    'p_value_liability': t['p_value_liability'],
                    'zscore_liability': t['zscore_liability'],
                })

            records_all.append(records)

            if (bets_df is not None) and len(bet_idx):
                meta = {
                    'candidate_id': cand_id,
                    'passed_test_gate': bool(pass_gate),
                    'mode': f'CLASSIFY_{classify_side.upper()}',
                    'market': market,
                    'threshold': thr,
                    'commission_rate': float(commission_rate),
                    'params_json': json.dumps(best_params, default=float),
                    'val_precision': float(cand['val_precision']),
                    'val_precision_lcb': float(cand['val_precision_lcb']),
                }
                bdf = bets_df.copy()
                for k, v in meta.items(): bdf[k] = v
                all_bets_collector.append(bdf)

    survivors_df = pd.DataFrame(records_all)
    passers = survivors_df[survivors_df['pass_test_gate']].copy()

    # ---------------- save / rank ----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "xgb" if base_model == "xgb" else "mlp"

    if passers.empty:
        fail_csv = None
        if save_diagnostics_on_fail:
            if 'p_value' in survivors_df.columns and survivors_df['p_value'].notna().any():
                sort_cols = ['p_value','total_pl','val_precision_lcb']; asc = [True, False, False]
            else:
                sort_cols = ['val_precision_lcb','val_precision','n_preds_test','val_accuracy']; asc = [False, False, False, False]
            diag = (survivors_df.drop(columns=['model_obj','bets'], errors='ignore')
                    .sort_values(by=sort_cols, ascending=asc)
                    .assign(market=market))
            fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv")
            diag.to_csv(fail_csv, index=False); summary_df = diag
        else:
            summary_df = survivors_df.drop(columns=['model_obj','bets'], errors='ignore')

        all_bets_csv_path = None
        if save_all_bets_csv and ((_USE_VALUE) or (_IS_CLASSIFY and classify_odds_column is not None)) and all_bets_collector:
            all_bets_df = pd.concat(all_bets_collector, ignore_index=True)
            if not all_bets_include_failed:
                all_bets_df = all_bets_df[all_bets_df['passed_test_gate'] == True]
            all_bets_csv_path = os.path.join(all_bets_dir, f"all_bets_{market}_{timestamp}.csv")
            all_bets_df.to_csv(all_bets_csv_path, index=False)

        if on_fail == "raise": raise RuntimeError("All Top-K failed the TEST gate.")
        if on_fail == "warn": print("[WARN] All Top-K failed the TEST gate.")
        return {'status':'failed_test_gate','csv':fail_csv,'model_pkl':None,
                'summary_df':summary_df,'validation_table':ranked,
                'bets_csv':None,'pl_plot':None,'all_bets_csv':all_bets_csv_path}

    # Final ranking (unchanged core)
    if ('p_value' in passers.columns) and passers['p_value'].notna().any():
        passers_sorted = passers.sort_values(
            by=['p_value','total_pl','avg_pl','val_precision_lcb','val_precision'],
            ascending=[True, False, False, False, False]
        ).reset_index(drop=True)
    else:
        passers_sorted = passers.sort_values(
            by=['val_precision_lcb','val_precision','test_precision','n_preds_test','val_accuracy'],
            ascending=[False, False, False, False, False]
        ).reset_index(drop=True)

    # Save PKL + CSV
    pkl_path = os.path.join(model_dir, f"best_model_{market}_{tag}_calibrated_{timestamp}.pkl")
    csv_df = passers_sorted.drop(columns=['model_obj','bets'], errors='ignore').copy()
    csv_df['model_pkl'] = ""; csv_df.loc[0, 'model_pkl'] = pkl_path
    csv_df['market'] = market
    csv_path = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}.csv")
    csv_df.to_csv(csv_path, index=False)

    # Save top model
    top_row = passers_sorted.iloc[0]
    chosen_model = top_row['model_obj']
    if base_model == "xgb":
        param_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']
    else:
        param_keys = ['hidden_layer_sizes','alpha','learning_rate_init','batch_size','max_iter']
    chosen_params = {k: top_row[k] for k in param_keys if k in passers_sorted.columns}
    chosen_threshold = float(top_row.get('threshold', np.nan))
    chosen_edge = float(top_row.get('edge_param', np.nan))
    chosen_odds_min = float(top_row.get('odds_min', np.nan)) if 'odds_min' in top_row.index else np.nan
    chosen_odds_max = float(top_row.get('odds_max', np.nan)) if 'odds_max' in top_row.index else np.nan

    joblib.dump(
        {
            'model': chosen_model,
            'threshold': chosen_threshold,            # NaN in VALUE modes; meaningful in CLASSIFY
            'edge_param': chosen_edge,                # chosen edge (VALUE)
            'features': features,
            'base_model': base_model,
            'best_params': chosen_params,
            'precision_test_threshold': float(precision_test_threshold),
            'min_samples': int(min_samples),
            'min_test_samples': int(min_test_samples),
            'val_conf_level': float(val_conf_level),
            'max_precision_drop': float(max_precision_drop),
            'market': market,
            'mode': top_row['mode'],
            # VALUE mode staking winners (if any):
            'staking_plan_lay': top_row.get('staking_plan_lay', None) if _USE_VALUE_LAY else None,
            'staking_plan_back': top_row.get('staking_plan_back', None) if _USE_VALUE_BACK else None,
            # numeric staking params (VALUE):
            'liability_test': float(liability_test) if _USE_VALUE_LAY else None,
            'lay_flat_stake': float(lay_flat_stake) if _USE_VALUE_LAY else None,
            'lay_edge_scale': float(lay_edge_scale) if _USE_VALUE_LAY else None,
            'kelly_fraction_lay': float(kelly_fraction_lay) if _USE_VALUE_LAY else None,
            'min_lay_stake': float(min_lay_stake) if _USE_VALUE_LAY else None,
            'max_lay_stake': float(max_lay_stake) if _USE_VALUE_LAY else None,
            'min_lay_liability': float(min_lay_liability) if _USE_VALUE_LAY else None,
            'max_lay_liability': float(max_lay_liability) if _USE_VALUE_LAY else None,
            'back_stake_test': float(back_stake_test) if _USE_VALUE_BACK else None,
            'back_edge_scale': float(back_edge_scale) if _USE_VALUE_BACK else None,
            'kelly_fraction_back': float(kelly_fraction_back) if _USE_VALUE_BACK else None,
            'bankroll_back': float(bankroll_back) if _USE_VALUE_BACK else None,
            'min_back_stake': float(min_back_stake) if _USE_VALUE_BACK else None,
            'max_back_stake': float(max_back_stake) if _USE_VALUE_BACK else None,
            # CLASSIFY specifics for live use:
            'classify_stake': float(classify_stake) if _IS_CLASSIFY else None,
            'classify_odds_column': classify_odds_column if _IS_CLASSIFY else None,
            'classify_side': classify_side if _IS_CLASSIFY else None,
            'classify_odds_min': chosen_odds_min if _IS_CLASSIFY else None,
            'classify_odds_max': chosen_odds_max if _IS_CLASSIFY else None,
            # NEW: store classify-lay variant knobs
            'classify_lay_flat_stake': float(classify_lay_flat_stake) if (_IS_CLASSIFY and classify_side=='lay') else None,
            'classify_lay_liability': float(classify_lay_liability) if (_IS_CLASSIFY and classify_side=='lay') else None,
            # commission saved
            'commission_rate': float(commission_rate),
            'notes': ('Commission applied to winning returns; '
                      'VALUE & CLASSIFY(with-odds) ranked by smallest p-value; '
                      'CLASSIFY sweeps threshold + odds bands + side(back/lay); '
                      'CLASSIFY_LAY evaluates flat-stake and flat-liability variants; '
                      'bets CSV & cumulative P/L plot saved for the winner.'),
            'run_seed': int(RUN_SEED),
        },
        pkl_path
    )

    # chosen bets CSV / plot
    bets_path = None
    plot_path = None
    bets_df = top_row.get('bets', None)
    if (save_bets_csv or plot_pl) and isinstance(bets_df, pd.DataFrame) and len(bets_df):
        if save_bets_csv:
            bets_name = f"bets_{market}_{timestamp}.csv"
            bets_path = os.path.join(bets_csv_dir, bets_name)
            bets_df.to_csv(bets_path, index=False)
        if plot_pl:
            try:
                import matplotlib.pyplot as plt
                fig = plt.figure()
                x = bets_df['date'] if 'date' in bets_df.columns else np.arange(len(bets_df))
                # Use 'cum_pl' (defined in VALUE/CLASSIFY_BACK; for CLASSIFY_LAY we set cum_pl := cum_pl_flat)
                y = bets_df['cum_pl'] if 'cum_pl' in bets_df.columns else (bets_df['cum_pl_flat'] if 'cum_pl_flat' in bets_df.columns else None)
                if y is None:
                    y = bets_df['pl'].cumsum()
                plt.plot(x, y)
                title = f"{market} cumulative P/L ({top_row['mode']})"
                if plot_title_suffix: title += f" — {plot_title_suffix}"
                plt.title(title)
                plt.xlabel('Date' if 'date' in bets_df.columns else 'Bet #')
                plt.ylabel('Cumulative P/L')
                plt.tight_layout()
                plot_name = f"cum_pl_{market}_{timestamp}.png"
                plot_path = os.path.join(plot_dir, plot_name)
                plt.savefig(plot_path, dpi=160); plt.close(fig)
            except Exception as e:
                print(f"[WARN] Failed to create plot: {e}")

    # ALL bets export (across all candidates) — sibling all_bets dir
    all_bets_csv_path = None
    if save_all_bets_csv and ((_USE_VALUE) or (_IS_CLASSIFY and classify_odds_column is not None)) and all_bets_collector:
        all_bets_df = pd.concat(all_bets_collector, ignore_index=True)
        if not all_bets_include_failed:
            all_bets_df = all_bets_df[all_bets_df['passed_test_gate'] == True]
        preferred = [c for c in [
            'date','league','country','home_team','away_team','match_id','event_name','selection',
            'market_odds','fair_odds','edge_ratio','stake','liability','commission_rate',
            'selection_won','target','pl','cum_pl','cum_pl_flat','cum_pl_liability',
            'candidate_id','passed_test_gate','mode','market','threshold',
            'odds_min','odds_max','edge_param',
            'staking_plan_lay','staking_plan_back',
            'val_precision','val_precision_lcb','n_value_bets_test','total_pl','avg_pl','p_value','zscore','params_json'
        ] if c in all_bets_df.columns]
        all_bets_df = all_bets_df[preferred + [c for c in all_bets_df.columns if c not in preferred]]
        all_bets_csv_path = os.path.join(all_bets_dir, f"all_bets_{market}_{timestamp}.csv")
        all_bets_df.to_csv(all_bets_csv_path, index=False)

    return {
        'status': 'ok',
        'csv': csv_path,
        'model_pkl': pkl_path,
        'summary_df': csv_df,
        'validation_table': ranked,
        'bets_csv': bets_path,
        'pl_plot': plot_path,
        'all_bets_csv': all_bets_csv_path,
    }





def simple_feature_importance(df, features, target_col='target'):
    """
    Fit a RandomForestClassifier and print its built-in feature importances.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing both features and the target column.
    features : list of str
        Names of feature columns.
    target_col : str
        Name of the target column in df.

    Returns
    -------
    importances : pd.Series
        Sorted feature importances.
    """
    X = df[features]
    y = df[target_col]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    importances = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    print("Feature importances:")
    print(importances.to_string())

    return importances


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
    data['total_goals'] = data['home_goals_ft'] + data['away_goals_ft']
    data['target'] = data['total_goals'].apply(lambda x: 1 if x > 2.5 else 0)
    return data
