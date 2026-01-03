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

    # Defensive: if any column names duplicated, keep first occurrence only
    if val_df.columns.duplicated().any():
        dupes = val_df.columns[val_df.columns.duplicated()].tolist()
        print(f"[WARN] Duplicate columns in val_df removed: {dupes}")
        val_df = val_df.loc[:, ~val_df.columns.duplicated()].copy()

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

# def run_models_25_v2(
#         matches_filtered: pd.DataFrame,
#         features: list,
#         min_samples: int = 200,  # validation gate: min predicted positives across folds
#         min_test_samples: int = 100,  # test gate: min predicted positives on hold-out test
#         precision_test_threshold: float = 0.80,
#         base_model: str = "xgb",  # "xgb" or "mlp"
#         search_mode: str = "random",  # "random" or "grid"
#         n_random_param_sets: int = 10,
#         cpu_jobs: int = 6,
#         top_k: int = 10,
#         thresholds: np.ndarray | None = None,
#         out_dir: str | None = None,
#         # --- anti-overfitting knobs ---
#         val_conf_level: float = 0.99,  # Wilson-LCB confidence for validation precision
#         max_precision_drop: float = 0.02,  # allow at most 10pp drop val → test
#         # --- optional recall gate on validation ---
#         min_val_recall: float = 0.0,
#         # --- failure handling ---
#         on_fail: str = "return",  # "return" | "warn" | "raise"
#         save_diagnostics_on_fail: bool = True,
#         market: str = "OVER"
# ):
#     """
#     Rolling time-ordered CV (no leakage) with calibration and robust failure handling.
#
#     Key v2 changes:
#       - Validation gate uses Wilson-LCB of precision (val_precision_lcb) instead of raw mean precision.
#       - Ranking uses log-loss & Brier (probability-quality metrics) plus Wilson-LCB.
#       - Recall is tracked (per-fold and pooled); optional min_val_recall gate.
#       - Test survivors are ranked with val/test log-loss & Brier plus test precision & volume.
#     """
#     # ------------------ Imports & setup ------------------
#     import os, secrets, hashlib
#     from datetime import datetime
#     import numpy as np
#     import pandas as pd
#     from itertools import product
#     from math import sqrt
#     from sklearn.model_selection import ParameterSampler
#     from sklearn.metrics import (
#         precision_score,
#         accuracy_score,
#         roc_auc_score,
#         log_loss,
#         brier_score_loss,
#     )
#     from sklearn.calibration import CalibratedClassifierCV
#     from sklearn.pipeline import Pipeline, make_pipeline
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.neural_network import MLPClassifier
#     from joblib import Parallel, delayed, parallel_backend
#     from tqdm import tqdm
#     from tqdm_joblib import tqdm_joblib
#     import joblib
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
#         denom = 1.0 + (z * z) / n
#         centre = p + (z * z) / (2 * n)
#         rad = z * sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
#         return max(0.0, (centre - rad) / denom)
#
#     if thresholds is None:
#         thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)
#
#     out_dir = out_dir or os.getcwd()
#     os.makedirs(out_dir, exist_ok=True)
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
#     X_val_final = X.iloc[final_val_start:test_start]
#     y_val_final = y.iloc[final_val_start:test_start]
#
#     # ------------------ Hyper-parameter spaces ------------------
#     xgb_param_grid = {
#         'n_estimators': [200],
#         'max_depth': [5],
#         'learning_rate': [0.1],
#         'subsample': [0.7],
#         'colsample_bytree': [1.0],
#         'min_child_weight': [5],
#         'reg_lambda': [1.0],
#     }
#     xgb_param_distributions = {
#         'n_estimators': _randint(100, 1001),
#         'max_depth': _randint(3, 8),
#         'learning_rate': _loguniform(0.01, 0.2),
#         'min_child_weight': _randint(3, 13),
#         'subsample': _uniform(0.7, 0.3),
#         'colsample_bytree': _uniform(0.6, 0.4),
#         'reg_lambda': _loguniform(0.1, 10.0),
#     }
#     mlp_param_grid = {
#         'hidden_layer_sizes': [(128,), (256,), (128, 64)],
#         'alpha': [1e-4],
#         'learning_rate_init': [1e-3],
#         'batch_size': ['auto'],
#         'max_iter': [200],
#     }
#     mlp_param_distributions = {
#         'hidden_layer_sizes': [(64,), (128,), (256,), (128, 64), (256, 128)],
#         'alpha': _loguniform(1e-5, 1e-2),
#         'learning_rate_init': _loguniform(5e-4, 5e-2),
#         'batch_size': _randint(32, 257),
#         'max_iter': _randint(150, 401),
#     }
#
#     if search_mode.lower() == "grid":
#         grid, dists = (xgb_param_grid, None) if base_model == "xgb" else (mlp_param_grid, None)
#         all_param_dicts = [dict(zip(grid.keys(), combo)) for combo in product(*grid.values())]
#     else:
#         grid, dists = (xgb_param_grid, xgb_param_distributions) if base_model == "xgb" else (mlp_param_grid,
#                                                                                              mlp_param_distributions)
#         # Random every run
#         sampler_seed = _seed_from("sampler")
#         all_param_dicts = list(ParameterSampler(dists, n_iter=n_random_param_sets, random_state=sampler_seed))
#
#     # ------------------ Helpers ------------------
#     def cast_params(p: dict) -> dict:
#         q = dict(p)
#         if base_model == "xgb":
#             for k in ['n_estimators', 'max_depth', 'min_child_weight']:
#                 if k in q:
#                     q[k] = int(q[k])
#             for k in ['learning_rate', 'subsample', 'colsample_bytree', 'reg_lambda']:
#                 if k in q:
#                     q[k] = float(q[k])
#         else:
#             if 'max_iter' in q:
#                 q['max_iter'] = int(q['max_iter'])
#             if 'batch_size' in q and q['batch_size'] != 'auto':
#                 q['batch_size'] = int(q['batch_size'])
#             if 'alpha' in q:
#                 q['alpha'] = float(q['alpha'])
#             if 'learning_rate_init' in q:
#                 q['learning_rate_init'] = float(q['learning_rate_init'])
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
#         # model seed depends on RUN_SEED and params → changes each run
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
#     def predict_proba_1(model_or_cal, X_):
#         proba = model_or_cal.predict_proba(X_)
#         return proba[:, 1].astype(np.float32) if proba.ndim == 2 else np.asarray(proba, dtype=np.float32)
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
#             pos = int(y_tr.sum())
#             neg = len(y_tr) - pos
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
#                 pos_fold = int((y_true == 1).sum())
#                 fn = max(0, pos_fold - tp)
#                 prc = precision_score(y_va, y_pred, zero_division=0)
#                 acc = accuracy_score(y_va, y_pred)
#                 rec = float(tp / pos_fold) if pos_fold > 0 else 0.0
#
#                 rows.append({
#                     **safe,
#                     'threshold': float(thr),
#                     'fold_vstart': int(vstart),
#                     'fold_vend': int(vend),
#                     'n_preds_val': n_preds,
#                     'tp_val': tp,
#                     'fp_val': fp,
#                     'fn_val': fn,
#                     'val_precision': float(prc),
#                     'val_accuracy': float(acc),
#                     'val_recall': float(rec),
#                 })
#
#         # pooled diagnostics (optional)
#         if val_prob_all:
#             vp = np.concatenate(val_prob_all, axis=0)
#             vt = np.concatenate(val_true_all, axis=0)
#             try:
#                 val_auc = float(roc_auc_score(vt, vp))
#             except Exception:
#                 val_auc = np.nan
#             try:
#                 val_ll = float(log_loss(vt, vp, labels=[0, 1]))
#             except Exception:
#                 val_ll = np.nan
#             try:
#                 val_bri = float(brier_score_loss(vt, vp))
#             except Exception:
#                 val_bri = np.nan
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
#         prefer = "threads"
#         backend = "threading"
#         pre_dispatch = eff_jobs
#         ctx = parallel_backend(backend, n_jobs=eff_jobs)
#     else:
#         eff_jobs = max(1, min(cpu_jobs, 4)) if cpu_jobs != -1 else 4
#         prefer = "processes"
#         backend = "loky"
#         pre_dispatch = f"{2 * eff_jobs}"
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
#     param_keys = list((xgb_param_grid if base_model == "xgb" else mlp_param_grid).keys())
#     group_cols = param_keys + ['threshold']
#     agg = val_df.groupby(group_cols, as_index=False).agg({
#         'n_preds_val': 'sum',
#         'tp_val': 'sum',
#         'fp_val': 'sum',
#         'fn_val': 'sum',
#         'val_precision': 'mean',
#         'val_accuracy': 'mean',
#         'val_recall': 'mean',
#         'val_auc': 'mean',
#         'val_logloss': 'mean',
#         'val_brier': 'mean',
#     })
#
#     # Wilson-LCB & pooled precision/recall
#     agg['val_precision_pooled'] = agg.apply(
#         lambda r: (r['tp_val'] / max(1, (r['tp_val'] + r['fp_val']))), axis=1
#     )
#     agg['val_precision_lcb'] = agg.apply(
#         lambda r: _wilson_lcb(int(r['tp_val']), int(r['fp_val']), conf=val_conf_level), axis=1
#     )
#     agg['val_recall_pooled'] = agg.apply(
#         lambda r: (r['tp_val'] / max(1, (r['tp_val'] + r['fn_val']))), axis=1
#     )
#
#     # ------------------ Validation gates (LCB + volume + optional recall) ------------------
#     qual = agg[
#         (agg['val_precision_lcb'] >= float(precision_test_threshold)) &
#         (agg['n_preds_val'] >= int(min_samples))
#     ].copy()
#
#     if min_val_recall > 0.0:
#         qual = qual[qual['val_recall'] >= float(min_val_recall)].copy()
#
#     if qual.empty:
#         # nothing qualifies even on validation → treat as failure early
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         sort_cols = ['val_logloss', 'val_brier', 'val_precision_lcb', 'n_preds_val']
#         sort_order = [True, True, False, False]
#         if save_diagnostics_on_fail:
#             fail_path = os.path.join(out_dir, f"model_metrics_{timestamp}_FAILED.csv")
#             (agg.sort_values(sort_cols, ascending=sort_order)
#              .assign(fail_reason="failed_validation_gate")
#              .to_csv(fail_path, index=False))
#         msg = (f"No strategy met validation gates (precision LCB ≥ {precision_test_threshold} "
#                f"and n_preds_val ≥ {min_samples}"
#                + (f" and val_recall ≥ {min_val_recall}" if min_val_recall > 0 else "")
#                + ").")
#         if on_fail == "raise":
#             raise RuntimeError(msg)
#         if on_fail == "warn":
#             print("[WARN]", msg)
#         return {
#             'status': 'failed_validation_gate',
#             'csv': fail_path if save_diagnostics_on_fail else None,
#             'model_pkl': None,
#             'summary_df': None,
#             'validation_table': agg.sort_values(sort_cols, ascending=sort_order).reset_index(drop=True)
#         }
#
#     # STRICT validation ordering (probability quality first, then conservative precision)
#     ranked = qual.sort_values(
#         by=['val_logloss', 'val_brier', 'val_precision_lcb', 'n_preds_val'],
#         ascending=[True, True, False, False]
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
#             'val_recall': float(row['val_recall']),
#             'val_recall_pooled': float(row['val_recall_pooled']),
#             'val_logloss': float(row['val_logloss']),
#             'val_brier': float(row['val_brier']),
#             'n_preds_val': int(row['n_preds_val']),
#         })
#
#     # Evaluate each Top-K candidate on TEST
#     records_all = []  # every candidate with test metrics + pass/fail reason
#     for cand in candidates:
#         best_params = cast_params(cand['params'])
#         pos = int(y_train_final.sum())
#         neg = len(y_train_final) - pos
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
#         # test recall
#         y_true_test = y_test.values.astype(np.uint8)
#         tp_test = int(((y_true_test == 1) & (y_pred == 1)).sum())
#         pos_test = int((y_true_test == 1).sum())
#         rec_test = float(tp_test / pos_test) if pos_test > 0 else 0.0
#
#         # test prob-based metrics
#         try:
#             test_auc = float(roc_auc_score(y_true_test, y_test_proba))
#         except Exception:
#             test_auc = np.nan
#         try:
#             test_ll = float(log_loss(y_true_test, y_test_proba, labels=[0, 1]))
#         except Exception:
#             test_ll = np.nan
#         try:
#             test_bri = float(brier_score_loss(y_true_test, y_test_proba))
#         except Exception:
#             test_bri = np.nan
#
#         # TEST GATE checks + reason (same logic as v1, but using updated cand metrics)
#         enough = n_preds_test >= int(min_test_samples)
#         not_collapsed = prc_test >= max(
#             float(precision_test_threshold),
#             float(cand['val_precision']) - float(max_precision_drop)
#         )
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
#             # validation metrics carried through
#             'val_precision_lcb': cand['val_precision_lcb'],
#             'val_precision': cand['val_precision'],
#             'val_accuracy': cand['val_accuracy'],
#             'val_recall': cand['val_recall'],
#             'val_recall_pooled': cand['val_recall_pooled'],
#             'val_logloss': cand['val_logloss'],
#             'val_brier': cand['val_brier'],
#             'n_preds_val': cand['n_preds_val'],
#             # test metrics
#             'n_preds_test': n_preds_test,
#             'test_precision': float(prc_test),
#             'test_accuracy': float(acc_test),
#             'test_recall': float(rec_test),
#             'test_auc': float(test_auc),
#             'test_logloss': float(test_ll),
#             'test_brier': float(test_bri),
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
#         # No survivor: write diagnostics, optionally return or warn instead of raising
#         sort_cols = ['val_logloss', 'val_brier', 'val_precision_lcb', 'n_preds_val']
#         sort_order = [True, True, False, False]
#         if save_diagnostics_on_fail:
#             diag = (survivors_df
#                     .drop(columns=['model_obj'])
#                     .sort_values(by=sort_cols, ascending=sort_order)
#                     )
#             fail_csv = os.path.join(out_dir, f"model_metrics_{timestamp}_FAILED.csv")
#             diag.to_csv(fail_csv, index=False)
#         else:
#             diag = survivors_df.drop(columns=['model_obj'])
#             fail_csv = None
#
#         msg = (f"All Top-{len(candidates)} failed the TEST gate "
#                f"(n_preds_test ≥ {min_test_samples} and precision not collapsing).")
#         if on_fail == "raise":
#             raise RuntimeError(msg)
#         if on_fail == "warn":
#             print("[WARN]", msg)
#         return {
#             'status': 'failed_test_gate',
#             'csv': fail_csv,
#             'model_pkl': None,
#             'summary_df': diag,
#             'validation_table': ranked,
#         }
#
#     # At least one survivor → choose best & save PKL + CSV (only passers)
#     passers_sorted = passers.sort_values(
#         by=[
#             'val_logloss',     # best probability quality on validation
#             'val_brier',
#             'test_logloss',    # then best probability quality on test
#             'test_brier',
#             'test_precision',  # subject to gate, but still useful as tie-breaker
#             'n_preds_test',
#             'val_precision_lcb',
#         ],
#         ascending=[True, True, True, True, False, False, False]
#     ).reset_index(drop=True)
#
#     # Save PKL next to CSVs
#     if market == 'OVER':
#         pkl_path = os.path.join(
#             r"C:\Users\leere\PycharmProjects\Football_ML3\Goals\Over_2_5\model_file",
#             f"best_model_{tag}_calibrated_{timestamp}.pkl"
#         )
#     else:
#         pkl_path = os.path.join(
#             r"C:\Users\leere\PycharmProjects\Football_ML3\Goals\Under_2_5\model_file",
#             f"best_model_{tag}_calibrated_{timestamp}.pkl"
#         )
#
#     # Prepare CSV (include the PKL path for the winning row only)
#     csv_df = passers_sorted.drop(columns=['model_obj']).copy()
#     csv_df['model_pkl'] = ""
#     csv_df.loc[0, 'model_pkl'] = pkl_path
#
#     csv_path = os.path.join(out_dir, f"model_metrics_{timestamp}.csv")
#     csv_df.to_csv(csv_path, index=False)
#
#     # Save PKL for the top row
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
#             'min_val_recall': float(min_val_recall),
#             'notes': (
#                 'CSV includes only candidates passing test gate; '
#                 'v2 ranks by val/test logloss & Brier, with Wilson-LCB precision and '
#                 'test precision/volume as constraints. Seeds are random each run.'
#             ),
#             'run_seed': int(RUN_SEED),  # for traceability
#         },
#         pkl_path
#     )
#
#     return {
#         'status': 'ok',
#         'csv': csv_path,
#         'model_pkl': pkl_path,
#         'summary_df': csv_df,        # passers only, with model_pkl set on row 0
#         'validation_table': ranked,  # full validation ranking (post-gates)
#     }

def run_models_25_v4_sensible_profit_selection(
        matches_filtered: "pd.DataFrame",
        features: list,

        # ── gates (validation) ───────────────────────────────────────────────
        min_samples_val: int = 200,
        precision_threshold: float = 0.80,
        val_conf_level: float = 0.99,
        min_val_recall: float = 0.0,

        # ── gates (selection) ────────────────────────────────────────────────
        min_bets_select: int = 100,
        min_roi_select: float | None = 0.0,

        # ── models to test ───────────────────────────────────────────────────
        models_to_test: tuple = ("xgb", "mlp"),

        # ── search ───────────────────────────────────────────────────────────
        search_mode: str = "random",
        n_random_param_sets: int = 10,
        cpu_jobs: int = 6,
        top_k: int = 10,
        thresholds: "np.ndarray | None" = None,

        out_dir: str | None = None,

        # ── market / odds ────────────────────────────────────────────────────
        market: str = "OVER",
        commission_rate: float = 0.0,

        # ── splits (time-ordered) ────────────────────────────────────────────
        train_frac: float = 0.70,
        select_frac: float = 0.15,

        # ── objective & ranking (SELECTION) ──────────────────────────────────
        selection_objective: str = "profit",   # "profit" | "roi" | "profit_dd"
        max_drawdown_floor: float | None = None,

        # ── staking simulation knobs ─────────────────────────────────────────
        stake_flat: float = 1.0,

        # Whitaker settings (reported)
        whitaker_bank_start: float = 100.0,
        whitaker_els_multiple: float = 4.0,
        whitaker_n_bets: int = 1000,
        whitaker_linked: bool = True,
        whitaker_divisor: float = 1.0,

        # ── persistence ──────────────────────────────────────────────────────
        save_bets_csv: bool = True,
        save_plot: bool = True,

        # ── failure handling ─────────────────────────────────────────────────
        on_fail: str = "return",
        save_diagnostics_on_fail: bool = True,
):
    import os, secrets, hashlib
    from datetime import datetime
    from math import sqrt

    import numpy as np
    import pandas as pd

    from sklearn.model_selection import ParameterSampler
    from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, log_loss, brier_score_loss
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier

    import joblib
    from joblib import Parallel, delayed, parallel_backend
    from tqdm import tqdm

    try:
        from tqdm_joblib import tqdm_joblib
        _HAS_TQDM_JOBLIB = True
    except Exception:
        _HAS_TQDM_JOBLIB = False

    # xgboost optional
    try:
        import xgboost as xgb
        _HAS_XGB_LOCAL = True
    except Exception:
        _HAS_XGB_LOCAL = False

    # scipy distributions fallback
    try:
        _randint  # noqa: F821
        _uniform  # noqa: F821
        _loguniform  # noqa: F821
    except NameError:
        from scipy.stats import randint as _randint
        from scipy.stats import uniform as _uniform
        from scipy.stats import loguniform as _loguniform

    # Z for Wilson
    try:
        from scipy.stats import norm
        _Z = lambda conf: float(norm.ppf(1 - (1 - conf) / 2))
    except Exception:
        _Z = lambda conf: 2.576 if conf >= 0.99 else 1.96

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

    # ------------------ defaults / dirs ------------------
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)

    out_dir = out_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    # ------------------ RNG seed ------------------
    RUN_SEED = secrets.randbits(32)

    def _seed_from(*vals) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(int(RUN_SEED).to_bytes(8, "little", signed=False))
        for v in vals:
            h.update(str(v).encode("utf-8"))
        return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF

    # ------------------ data prep ------------------
    req_cols = {"date", "target"}
    missing = req_cols - set(matches_filtered.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = matches_filtered.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    market_u = str(market).upper().strip()
    if market_u not in {"OVER", "UNDER"}:
        raise ValueError("market must be 'OVER' or 'UNDER'")

    odds_col = "over_25_odds" if market_u == "OVER" else "under_25_odds"
    if odds_col not in df.columns:
        raise ValueError(f"Missing required odds column: {odds_col}")
    df[odds_col] = pd.to_numeric(df[odds_col], errors="coerce")

    cols_needed = list(set(features) | {"target", odds_col, "date"})
    df = df.dropna(subset=cols_needed).reset_index(drop=True)

    X = df[features].copy()
    y = df["target"].astype(int).reset_index(drop=True)

    n = len(X)
    if n < max(int(min_samples_val) * 3, 500):
        raise RuntimeError(f"Not enough rows: {n}")

    # ------------------ chronological splits ------------------
    train_frac = float(train_frac)
    select_frac = float(select_frac)
    if train_frac + select_frac >= 0.95:
        raise ValueError("train_frac + select_frac must leave at least ~5% for final test.")

    train_end = int(train_frac * n)
    select_end = int((train_frac + select_frac) * n)
    test_start = select_end

    X_train = X.iloc[:train_end].reset_index(drop=True)
    y_train = y.iloc[:train_end].reset_index(drop=True)

    X_select = X.iloc[train_end:select_end].reset_index(drop=True)
    y_select = y.iloc[train_end:select_end].reset_index(drop=True)
    odds_select = df[odds_col].iloc[train_end:select_end].reset_index(drop=True).astype(float)

    X_test = X.iloc[test_start:].reset_index(drop=True)
    y_test = y.iloc[test_start:].reset_index(drop=True)
    odds_test = df[odds_col].iloc[test_start:].reset_index(drop=True).astype(float)
    dates_test = df["date"].iloc[test_start:].reset_index(drop=True)

    # ------------------ rolling folds inside TRAIN only ------------------
    N_FOLDS = 5
    n_train = len(X_train)

    total_val_len = max(1, int(0.15 * n_train))
    val_len = max(1, total_val_len // N_FOLDS)

    fold_val_ends = [n_train - total_val_len + (i + 1) * val_len for i in range(N_FOLDS)]
    fold_val_starts = [end - val_len for end in fold_val_ends]
    if fold_val_ends:
        fold_val_ends[-1] = min(fold_val_ends[-1], n_train)
        fold_val_starts[-1] = max(0, fold_val_ends[-1] - val_len)

    # calibration slice at end of TRAIN
    final_val_len = max(1, val_len)
    final_val_start = max(0, n_train - final_val_len)

    X_train_final = X_train.iloc[:final_val_start]
    y_train_final = y_train.iloc[:final_val_start]
    X_val_final = X_train.iloc[final_val_start:]
    y_val_final = y_train.iloc[final_val_start:]

    # ------------------ param spaces per model ------------------
    xgb_param_grid = {
        "n_estimators": [200],
        "max_depth": [5],
        "learning_rate": [0.1],
        "subsample": [0.7],
        "colsample_bytree": [1.0],
        "min_child_weight": [5],
        "reg_lambda": [1.0],
    }
    xgb_param_distributions = {
        "n_estimators": _randint(100, 1001),
        "max_depth": _randint(3, 8),
        "learning_rate": _loguniform(0.01, 0.2),
        "min_child_weight": _randint(3, 13),
        "subsample": _uniform(0.7, 0.3),
        "colsample_bytree": _uniform(0.6, 0.4),
        "reg_lambda": _loguniform(0.1, 10.0),
    }
    mlp_param_grid = {
        "hidden_layer_sizes": [(128,), (256,), (128, 64)],
        "alpha": [1e-4],
        "learning_rate_init": [1e-3],
        "batch_size": ["auto"],
        "max_iter": [200],
    }
    mlp_param_distributions = {
        "hidden_layer_sizes": [(64,), (128,), (256,), (128, 64), (256, 128)],
        "alpha": _loguniform(1e-5, 1e-2),
        "learning_rate_init": _loguniform(5e-4, 5e-2),
        "batch_size": _randint(32, 257),
        "max_iter": _randint(150, 401),
    }

    def _param_dicts_for(model_key: str) -> list[dict]:
        mk = str(model_key).lower().strip()
        if mk == "xgb":
            grid, dists = xgb_param_grid, xgb_param_distributions
        elif mk == "mlp":
            grid, dists = mlp_param_grid, mlp_param_distributions
        else:
            raise ValueError(f"Unsupported model '{model_key}' (use 'xgb' or 'mlp').")

        if str(search_mode).lower() == "grid":
            from itertools import product
            keys = list(grid.keys())
            vals = [grid[k] for k in keys]
            return [dict(zip(keys, combo)) for combo in product(*vals)]
        sampler_seed = _seed_from("sampler", mk)
        return list(ParameterSampler(dists, n_iter=int(n_random_param_sets), random_state=sampler_seed))

    def _cast_params(model_key: str, p: dict) -> dict:
        mk = str(model_key).lower().strip()
        q = dict(p)
        if mk == "xgb":
            for k in ["n_estimators", "max_depth", "min_child_weight"]:
                if k in q:
                    q[k] = int(q[k])
            for k in ["learning_rate", "subsample", "colsample_bytree", "reg_lambda"]:
                if k in q:
                    q[k] = float(q[k])
        else:
            if "max_iter" in q:
                q["max_iter"] = int(q["max_iter"])
            if "batch_size" in q and q["batch_size"] != "auto":
                q["batch_size"] = int(q["batch_size"])
            if "alpha" in q:
                q["alpha"] = float(q["alpha"])
            if "learning_rate_init" in q:
                q["learning_rate_init"] = float(q["learning_rate_init"])
            if "hidden_layer_sizes" in q:
                h = q["hidden_layer_sizes"]
                q["hidden_layer_sizes"] = tuple(h) if not isinstance(h, tuple) else h
        return q

    # ------------------ model helpers ------------------
    def _final_step_name(estimator):
        try:
            if isinstance(estimator, Pipeline):
                return estimator.steps[-1][0]
        except Exception:
            pass
        return None

    def _build_model(model_key: str, params: dict, spw: float):
        mk = str(model_key).lower().strip()
        model_seed = _seed_from("model", mk, tuple(sorted(params.items())))
        if mk == "xgb":
            if not _HAS_XGB_LOCAL:
                raise ImportError("XGBoost not available; remove 'xgb' or install xgboost.")
            return xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=model_seed,
                scale_pos_weight=spw,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
                **params,
            )
        mlp = MLPClassifier(
            random_state=model_seed,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            solver="adam",
            **params,
        )
        return make_pipeline(StandardScaler(with_mean=True, with_std=True), mlp)

    def _fit_model(model_key: str, model, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None):
        mk = str(model_key).lower().strip()
        if mk == "xgb":
            try:
                model.set_params(verbosity=0, early_stopping_rounds=50)
                if X_va is not None and y_va is not None:
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                else:
                    model.fit(X_tr, y_tr, verbose=False)
            except Exception:
                model.fit(X_tr, y_tr, verbose=False)
            return

        fit_kwargs = {}
        if sample_weight is not None:
            stepname = _final_step_name(model)
            if stepname is not None:
                fit_kwargs[f"{stepname}__sample_weight"] = sample_weight
        try:
            model.fit(X_tr, y_tr, **fit_kwargs)
        except TypeError:
            model.fit(X_tr, y_tr)

    def _fit_calibrator(fitted, X_va, y_va):
        try:
            from sklearn.calibration import FrozenEstimator
            frozen = FrozenEstimator(fitted)
            cal = CalibratedClassifierCV(frozen, method="sigmoid", cv=None)
            cal.fit(X_va, y_va)
            return cal
        except Exception:
            try:
                cal = CalibratedClassifierCV(fitted, method="sigmoid", cv="prefit")
                cal.fit(X_va, y_va)
                return cal
            except Exception:
                return fitted

    def _predict_proba_pos(model_or_cal, X_):
        proba = model_or_cal.predict_proba(X_)
        return proba[:, 1].astype(np.float32) if proba.ndim == 2 else np.asarray(proba, dtype=np.float32)

    # ------------------ profit sim helpers (flat) ------------------
    def _flat_profit_series(y_true_u8: np.ndarray, odds: np.ndarray, bet_mask: np.ndarray, stake: float, comm: float):
        y_true_u8 = np.asarray(y_true_u8, dtype=np.uint8)
        odds = np.asarray(odds, dtype=float)
        bet_mask = np.asarray(bet_mask, dtype=bool)

        prof = np.zeros_like(odds, dtype=np.float64)
        if stake <= 0:
            return prof

        wins = (y_true_u8 == 1) & bet_mask
        losses = (y_true_u8 == 0) & bet_mask

        prof[wins] = (np.maximum(odds[wins], 1.01) - 1.0) * stake * (1.0 - float(comm))
        prof[losses] = -stake
        return prof

    def _max_drawdown_from_profit(profit: np.ndarray, bank0: float):
        bank = bank0 + np.cumsum(profit)
        peak = np.maximum.accumulate(bank)
        dd = bank - peak
        return float(np.min(dd)), bank

    # ------------------ rolling CV evaluator (TRAIN only) ------------------
    def _evaluate_one(model_key: str, param_dict: dict):
        safe = _cast_params(model_key, param_dict)
        rows = []
        val_prob_all, val_true_all = [], []

        for vstart, vend in zip(fold_val_starts, fold_val_ends):
            if vstart is None or vend is None or vstart <= 0 or vend <= vstart:
                continue

            X_tr, y_tr = X_train.iloc[:vstart], y_train.iloc[:vstart]
            X_va, y_va = X_train.iloc[vstart:vend], y_train.iloc[vstart:vend]

            if y_tr.nunique() < 2:
                continue

            pos = int(y_tr.sum())
            neg = len(y_tr) - pos
            spw = (neg / pos) if pos > 0 else 1.0

            sample_weight = None
            if str(model_key).lower().strip() == "mlp":
                sample_weight = np.where(y_tr.values == 1, spw, 1.0).astype(np.float32)

            model = _build_model(model_key, safe, spw)
            _fit_model(model_key, model, X_tr, y_tr, X_va, y_va, sample_weight=sample_weight)
            cal = _fit_calibrator(model, X_va, y_va)

            proba_va = _predict_proba_pos(cal, X_va)
            val_prob_all.append(proba_va)
            y_true = y_va.values.astype(np.uint8)
            val_true_all.append(y_true)

            for thr in thresholds:
                y_pred = (proba_va >= thr).astype(np.uint8)
                n_preds = int(y_pred.sum())
                tp = int(((y_true == 1) & (y_pred == 1)).sum())
                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                pos_fold = int((y_true == 1).sum())
                fn = max(0, pos_fold - tp)

                rows.append({
                    "model_key": str(model_key).lower().strip(),
                    **safe,
                    "threshold": float(thr),
                    "n_preds_val": n_preds,
                    "tp_val": tp,
                    "fp_val": fp,
                    "fn_val": fn,
                    "val_precision": float(precision_score(y_va, y_pred, zero_division=0)),
                    "val_accuracy": float(accuracy_score(y_va, y_pred)),
                    "val_recall": float(tp / pos_fold) if pos_fold > 0 else 0.0,
                })

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
            r["val_auc"] = val_auc
            r["val_logloss"] = val_ll
            r["val_brier"] = val_bri

        return rows

    # ------------------ build tasks across models ------------------
    tasks = []
    for mk in models_to_test:
        mk = str(mk).lower().strip()
        if mk not in {"xgb", "mlp"}:
            raise ValueError(f"Unsupported model in models_to_test: {mk}")
        for pset in _param_dicts_for(mk):
            tasks.append((mk, pset))
    if not tasks:
        raise RuntimeError("No tasks generated for model search.")

    # ------------------ run search (parallel) ------------------
    eff_jobs = max(1, min(int(cpu_jobs), 4))
    with parallel_backend("loky", n_jobs=eff_jobs, inner_max_num_threads=1):
        if _HAS_TQDM_JOBLIB:
            with tqdm_joblib(tqdm(total=len(tasks), desc=f"Param search ({search_mode}, models={models_to_test})")):
                outs = Parallel(n_jobs=eff_jobs, batch_size=1)(
                    delayed(_evaluate_one)(mk, pset) for mk, pset in tasks
                )
        else:
            outs = []
            for mk, pset in tqdm(tasks, desc=f"Param search (serial, models={models_to_test})"):
                outs.append(_evaluate_one(mk, pset))

    val_rows = [r for sub in outs for r in sub]
    if not val_rows:
        raise RuntimeError("No validation rows produced (check folds and input data).")
    val_df = pd.DataFrame(val_rows)

    # ------------------ aggregate (candidate = model + params + threshold) ------------------
    metric_cols = {
        "threshold", "n_preds_val", "tp_val", "fp_val", "fn_val",
        "val_precision", "val_accuracy", "val_recall",
        "val_auc", "val_logloss", "val_brier",
    }
    base_cols = {"model_key"}
    param_cols = [c for c in val_df.columns if c not in metric_cols and c not in base_cols]
    group_cols = ["model_key"] + param_cols + ["threshold"]

    agg = val_df.groupby(group_cols, as_index=False).agg({
        "n_preds_val": "sum",
        "tp_val": "sum",
        "fp_val": "sum",
        "fn_val": "sum",
        "val_precision": "mean",
        "val_accuracy": "mean",
        "val_recall": "mean",
        "val_auc": "mean",
        "val_logloss": "mean",
        "val_brier": "mean",
    })

    # ── CRITICAL FIX: compute Wilson LCB without DataFrame.apply ────────────
    # Also defensively remove any existing duplicate val_precision_lcb columns.
    if "val_precision_lcb" in agg.columns:
        # If duplicates ever happened, pandas would keep only one name here,
        # but this keeps behaviour deterministic anyway.
        agg = agg.drop(columns=["val_precision_lcb"], errors="ignore")

    tp_arr = agg["tp_val"].to_numpy(dtype=np.int64)
    fp_arr = agg["fp_val"].to_numpy(dtype=np.int64)
    conf = float(val_conf_level)

    agg["val_precision_lcb"] = np.array(
        [_wilson_lcb(int(tp), int(fp), conf) for tp, fp in zip(tp_arr, fp_arr)],
        dtype=np.float64
    )

    # ------------------ validation gates ------------------
    qual = agg[
        (agg["val_precision_lcb"] >= float(precision_threshold)) &
        (agg["n_preds_val"] >= int(min_samples_val))
    ].copy()

    if float(min_val_recall) > 0.0:
        qual = qual[qual["val_recall"] >= float(min_val_recall)].copy()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if qual.empty:
        fail_path = None
        if save_diagnostics_on_fail:
            fail_path = os.path.join(out_dir, f"model_metrics_{market_u}_{timestamp}_FAILED_VAL.csv")
            agg.sort_values(
                ["val_logloss", "val_brier", "val_precision_lcb", "n_preds_val"],
                ascending=[True, True, False, False]
            ).to_csv(fail_path, index=False)
        msg = "No candidate met validation gates."
        if on_fail == "raise":
            raise RuntimeError(msg)
        if on_fail == "warn":
            print("[WARN]", msg)
        return {
            "status": "failed_validation_gate",
            "csv": fail_path,
            "model_pkl": None,
            "validation_table": agg.reset_index(drop=True),
            "selection_table": None,
            "final_test_table": None,
            "betting_summary": None,
            "betting_csv": None,
            "betting_plot": None,
        }

    # rank for Top-K carry-forward
    ranked_val = qual.sort_values(
        by=["val_logloss", "val_brier", "val_precision_lcb", "n_preds_val"],
        ascending=[True, True, False, False]
    ).reset_index(drop=True)
    topk_val = ranked_val.head(int(top_k)).reset_index(drop=True)

    # ------------------ SELECTION: per (model+params) optimise threshold for profit ------------------
    selection_rows = []
    thr_grid = np.round(np.arange(0.05, 0.96, 0.01), 2)

    def _candidate_key(row_):
        parts = [str(row_["model_key"])]
        for c in param_cols:
            parts.append(f"{c}={row_[c]}")
        return "|".join(parts)

    seen = set()
    for _, row in topk_val.iterrows():
        key = _candidate_key(row)
        if key in seen:
            continue
        seen.add(key)

        model_key = str(row["model_key"]).lower().strip()
        params = {k: row[k] for k in param_cols if k in row.index}
        params = _cast_params(model_key, params)

        pos = int(y_train_final.sum())
        neg = len(y_train_final) - pos
        spw = (neg / pos) if pos > 0 else 1.0

        sw = None
        if model_key == "mlp":
            sw = np.where(y_train_final.values == 1, spw, 1.0).astype(np.float32)

        model = _build_model(model_key, params, spw)
        _fit_model(model_key, model, X_train_final, y_train_final, X_val_final, y_val_final, sample_weight=sw)
        cal = _fit_calibrator(model, X_val_final, y_val_final)

        proba_sel = _predict_proba_pos(cal, X_select)
        y_true_sel = y_select.values.astype(np.uint8)
        odds_sel = odds_select.values.astype(float)

        best_thr = None
        best_score = -np.inf
        best_profit = -np.inf
        best_roi = -np.inf
        best_dd = -np.inf
        best_n = 0
        best_prec = 0.0

        obj = str(selection_objective).lower().strip()

        for thr in thr_grid:
            bet_mask = np.asarray(proba_sel >= thr, dtype=bool)
            n_bets = int(bet_mask.sum())
            if n_bets < int(min_bets_select):
                continue

            prof = _flat_profit_series(y_true_sel, odds_sel, bet_mask, float(stake_flat), float(commission_rate))
            total_profit = float(prof.sum())
            total_staked = float(n_bets * float(stake_flat))
            roi = (total_profit / total_staked) if total_staked > 0 else 0.0

            max_dd, _ = _max_drawdown_from_profit(prof, float(whitaker_bank_start))

            y_pred = bet_mask.astype(np.uint8)
            prec = float(precision_score(y_true_sel, y_pred, zero_division=0))

            if min_roi_select is not None and roi < float(min_roi_select):
                continue
            if max_drawdown_floor is not None and max_dd < float(max_drawdown_floor):
                continue

            if obj == "roi":
                score = roi
            elif obj == "profit_dd":
                score = total_profit + 0.05 * max_dd
            else:
                score = total_profit

            # tie-breakers
            if (score > best_score) or (
                score == best_score and roi > best_roi
            ) or (
                score == best_score and roi == best_roi and max_dd > best_dd
            ) or (
                score == best_score and roi == best_roi and max_dd == best_dd and n_bets > best_n
            ):
                best_thr = float(thr)
                best_score = float(score)
                best_profit = float(total_profit)
                best_roi = float(roi)
                best_dd = float(max_dd)
                best_n = int(n_bets)
                best_prec = float(prec)

        selection_rows.append({
            "model_key": model_key,
            **params,
            "val_precision_lcb": float(row["val_precision_lcb"]),
            "val_precision": float(row["val_precision"]),
            "val_logloss": float(row["val_logloss"]),
            "val_brier": float(row["val_brier"]),
            "n_preds_val": int(row["n_preds_val"]),
            "best_threshold_select": best_thr if best_thr is not None else np.nan,
            "n_bets_select": best_n,
            "select_profit_flat": best_profit,
            "select_roi_flat": best_roi,
            "select_max_drawdown_flat": best_dd,
            "select_precision": best_prec,
            "pass_select": bool(best_thr is not None),
            "fail_reason": "" if best_thr is not None else "no_threshold_met_min_bets_or_gates",
        })

    selection_df = pd.DataFrame(selection_rows)
    passers = selection_df[selection_df["pass_select"]].copy()

    if passers.empty:
        fail_path = None
        if save_diagnostics_on_fail:
            fail_path = os.path.join(out_dir, f"selection_metrics_{market_u}_{timestamp}_FAILED.csv")
            selection_df.to_csv(fail_path, index=False)
        msg = "No candidate produced a valid (min bets) threshold on SELECTION."
        if on_fail == "raise":
            raise RuntimeError(msg)
        if on_fail == "warn":
            print("[WARN]", msg)
        return {
            "status": "failed_selection_gate",
            "csv": fail_path,
            "model_pkl": None,
            "validation_table": ranked_val,
            "selection_table": selection_df,
            "final_test_table": None,
            "betting_summary": None,
            "betting_csv": None,
            "betting_plot": None,
        }

    passers_sorted = passers.sort_values(
        by=["select_profit_flat", "select_roi_flat", "select_max_drawdown_flat", "n_bets_select", "val_precision_lcb"],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)

    winner = passers_sorted.iloc[0]
    chosen_model_key = str(winner["model_key"]).lower().strip()
    chosen_threshold = float(winner["best_threshold_select"])
    chosen_params = {k: winner[k] for k in param_cols if k in winner.index}
    chosen_params = _cast_params(chosen_model_key, chosen_params)

    # ------------------ FINAL TEST: refit on pre-test, calibrate before test ------------------
    X_pretest = X.iloc[:test_start].reset_index(drop=True)
    y_pretest = y.iloc[:test_start].reset_index(drop=True)

    calib_len2 = max(1, val_len)
    calib_start2 = max(0, len(X_pretest) - calib_len2)

    X_train2 = X_pretest.iloc[:calib_start2]
    y_train2 = y_pretest.iloc[:calib_start2]
    X_cal2 = X_pretest.iloc[calib_start2:]
    y_cal2 = y_pretest.iloc[calib_start2:]

    pos2 = int(y_train2.sum())
    neg2 = len(y_train2) - pos2
    spw2 = (neg2 / pos2) if pos2 > 0 else 1.0

    sw2 = None
    if chosen_model_key == "mlp":
        sw2 = np.where(y_train2.values == 1, spw2, 1.0).astype(np.float32)

    final_model = _build_model(chosen_model_key, chosen_params, spw2)
    _fit_model(chosen_model_key, final_model, X_train2, y_train2, X_cal2, y_cal2, sample_weight=sw2)
    final_cal = _fit_calibrator(final_model, X_cal2, y_cal2)

    proba_test = _predict_proba_pos(final_cal, X_test)
    bet_mask_test = np.asarray(proba_test >= chosen_threshold, dtype=bool)
    y_true_test = y_test.values.astype(np.uint8)

    y_pred_test = bet_mask_test.astype(np.uint8)
    n_bets_test = int(bet_mask_test.sum())
    test_precision = float(precision_score(y_true_test, y_pred_test, zero_division=0))
    test_accuracy = float(accuracy_score(y_true_test, y_pred_test))
    try:
        test_auc = float(roc_auc_score(y_true_test, proba_test))
    except Exception:
        test_auc = np.nan
    try:
        test_ll = float(log_loss(y_true_test, proba_test, labels=[0, 1]))
    except Exception:
        test_ll = np.nan
    try:
        test_bri = float(brier_score_loss(y_true_test, proba_test))
    except Exception:
        test_bri = np.nan

    final_test_table = pd.DataFrame([{
        "market": market_u,
        "odds_col": odds_col,
        "model_key": chosen_model_key,
        **chosen_params,
        "threshold": chosen_threshold,
        "n_bets_test": n_bets_test,
        "test_precision": test_precision,
        "test_accuracy": test_accuracy,
        "test_auc": test_auc,
        "test_logloss": test_ll,
        "test_brier": test_bri,
        "train_frac": float(train_frac),
        "select_frac": float(select_frac),
        "commission_rate": float(commission_rate),
    }])

    # ------------------ simulate final test (flat + Whitaker) ------------------
    stake_f = float(stake_flat)
    bank0 = float(whitaker_bank_start)

    profit_flat = _flat_profit_series(y_true_test, odds_test.values, bet_mask_test, stake_f, float(commission_rate))
    stake_flat_arr = np.where(bet_mask_test, stake_f, 0.0).astype(np.float64)

    n_els = int(max(2, whitaker_n_bets))
    mult = float(max(0.1, whitaker_els_multiple))
    divisor = float(max(1e-9, whitaker_divisor))
    linked = bool(whitaker_linked)

    stake_w = np.zeros(len(y_true_test), dtype=np.float64)
    profit_w = np.zeros(len(y_true_test), dtype=np.float64)

    current_bank = bank0
    odds_arr = odds_test.values.astype(float)
    for i in range(len(y_true_test)):
        if (not bool(bet_mask_test[i])) or (current_bank <= 0.0):
            continue

        base_bank = current_bank if linked else bank0
        od = float(odds_arr[i])
        if not np.isfinite(od) or od <= 1.000001:
            od = 1.01

        wr = 1.0 / od
        wr = min(max(wr, 1e-9), 1.0 - 1e-9)

        els = np.log(n_els) / (-np.log(1.0 - wr))
        els = max(1e-9, float(els))

        st = (base_bank / (els * mult)) / divisor
        st = min(st, current_bank)

        stake_w[i] = st
        if y_true_test[i] == 1:
            profit_w[i] = (od - 1.0) * st * (1.0 - float(commission_rate))
        else:
            profit_w[i] = -st

        current_bank = current_bank + profit_w[i]

    bets_df = pd.DataFrame({
        "date": dates_test,
        "market": market_u,
        "odds": odds_arr,
        "proba": proba_test.astype(np.float32),
        "threshold": float(chosen_threshold),
        "bet": bet_mask_test.astype(np.uint8),
        "target": y_true_test.astype(np.uint8),
        "stake_flat": stake_flat_arr,
        "profit_flat": profit_flat,
        "stake_whitaker": stake_w,
        "profit_whitaker": profit_w,
    })

    # Fix: cum on date-order
    bets_df = bets_df.sort_values("date").reset_index(drop=True)

    bets_df["cum_profit_flat"] = bets_df["profit_flat"].cumsum()
    bets_df["bank_flat"] = bank0 + bets_df["cum_profit_flat"]

    bets_df["cum_profit_whitaker"] = bets_df["profit_whitaker"].cumsum()
    bets_df["bank_whitaker"] = bank0 + bets_df["cum_profit_whitaker"]

    bets_df["flat_peak"] = bets_df["bank_flat"].cummax()
    bets_df["flat_drawdown"] = bets_df["bank_flat"] - bets_df["flat_peak"]
    bets_df["whitaker_peak"] = bets_df["bank_whitaker"].cummax()
    bets_df["whitaker_drawdown"] = bets_df["bank_whitaker"] - bets_df["whitaker_peak"]

    def _summary(prefix: str):
        prof = float(bets_df[f"profit_{prefix}"].sum())
        stakes = float(bets_df[f"stake_{prefix}"].sum())
        roi = (prof / stakes) if stakes > 0 else 0.0
        max_dd = float(bets_df[f"{prefix}_drawdown"].min())
        end_bank = float(bets_df[f"bank_{prefix}"].iloc[-1]) if len(bets_df) else float("nan")
        return {
            "n_bets": int(bets_df["bet"].sum()),
            "total_staked": stakes,
            "total_profit": prof,
            "roi": roi,
            "max_drawdown": max_dd,
            "end_bank": end_bank,
        }

    betting_summary = {
        "market": market_u,
        "winner": {"model_key": chosen_model_key, "threshold": float(chosen_threshold), "params": chosen_params},
        "final_test_metrics": {
            "n_bets_test": n_bets_test,
            "test_precision": test_precision,
            "test_accuracy": test_accuracy,
            "test_auc": test_auc,
            "test_logloss": test_ll,
            "test_brier": test_bri,
        },
        "flat": _summary("flat"),
        "whitaker": _summary("whitaker"),
        "run_seed": int(RUN_SEED),
    }

    # ------------------ persist ------------------
    if market_u == "OVER":
        model_dir = r"C:\Users\leere\PycharmProjects\Football_ML3\Goals\Over_2_5\model_file"
    else:
        model_dir = r"C:\Users\leere\PycharmProjects\Football_ML3\Goals\Under_2_5\model_file"
    os.makedirs(model_dir, exist_ok=True)

    timestamp2 = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_path = os.path.join(model_dir, f"best_model_{chosen_model_key}_calibrated_{timestamp2}.pkl")
    joblib.dump(
        {
            "model": final_cal,
            "threshold": float(chosen_threshold),
            "features": features,
            "model_key": chosen_model_key,
            "best_params": chosen_params,
            "market": market_u,
            "odds_col": odds_col,
            "notes": "v4 robust: Wilson LCB computed without apply to avoid multi-column assignment bug.",
            "run_seed": int(RUN_SEED),
        },
        pkl_path
    )

    selection_csv = os.path.join(out_dir, f"selection_metrics_{market_u}_{timestamp2}.csv")
    passers_sorted.to_csv(selection_csv, index=False)

    bet_csv_path = None
    if save_bets_csv:
        bet_csv_path = os.path.join(out_dir, f"bets_FINALTEST_{market_u}_{timestamp2}.csv")
        bets_df.to_csv(bet_csv_path, index=False)

    plot_path = None
    if save_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_path = os.path.join(out_dir, f"equity_flat_vs_whitaker_FINALTEST_{market_u}_{timestamp2}.png")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(bets_df["date"], bets_df["bank_flat"])
        ax.plot(bets_df["date"], bets_df["bank_whitaker"])
        link_label = "LC" if linked else "NLC"
        ax.set_title(
            f"FINAL TEST Equity - {market_u} 2.5 | Flat vs Whitaker {link_label} "
            f"(ELSx{mult:g}, n={n_els}) | winner={chosen_model_key}"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Bank (units)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    return {
        "status": "ok",
        "selection_csv": selection_csv,
        "model_pkl": pkl_path,
        "validation_table": ranked_val,
        "selection_table": passers_sorted,
        "final_test_table": final_test_table,
        "betting_summary": betting_summary,
        "betting_csv": bet_csv_path,
        "betting_plot": plot_path,
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





# def run_models_outcome(
#     matches_filtered,                       # pd.DataFrame
#     features,                               # list[str]
#     # ── gates ──────────────────────────────────────────────────────────────
#     min_samples: int = 200,
#     min_test_samples: int = 100,
#     precision_test_threshold: float = 0.80,
#     # ── model/search ───────────────────────────────────────────────────────
#     base_model: str = "xgb",                # "xgb" | "mlp"
#     search_mode: str = "random",            # "random" | "grid"
#     n_random_param_sets: int = 10,
#     cpu_jobs: int = 6,
#     top_k: int = 10,
#     thresholds=None,                        # np.ndarray | None  (CLASSIFY only)
#     out_dir: str | None = None,
#     # ── anti-overfitting ──────────────────────────────────────────────────
#     val_conf_level: float = 0.99,
#     max_precision_drop: float = 1.0,
#     # ── failure handling ───────────────────────────────────────────────────
#     on_fail: str = "return",                # "return" | "warn" | "raise"
#     save_diagnostics_on_fail: bool = True,
#     # ── market ─────────────────────────────────────────────────────────────
#     market: str = "LAY_AWAY",               # LAY_* | BACK_* | OVER | UNDER (non-VALUE treated as CLASSIFY)
#
#     # ── VALUE controls ─────────────────────────────────────────────────────
#     use_value_for_lay: bool = True,
#     use_value_for_back: bool = True,
#     value_edge_grid_lay=None,               # np.ndarray | None  (edge >= 0): fair ≥ (1+edge)*market  (LAY)
#     value_edge_grid_back=None,              # np.ndarray | None  (edge >= 0): market ≥ (1+edge)*fair  (BACK)
#
#     # ── LAY minimum-odds sweep (applies to VALUE-LAY & CLASSIFY-LAY) ──────
#     lay_min_odds_grid=None,                 # np.ndarray | None (e.g. np.round(np.arange(1.05,3.51,0.05),2))
#
#     # ── VALUE staking options ──────────────────────────────────────────────
#     enable_staking_plan_search: bool = False,
#     staking_plan_lay_options=None,          # ["liability","flat_stake","edge_prop","kelly_approx"]
#     staking_plan_back_options=None,         # ["flat","edge_prop","kelly"]
#     staking_plan_lay: str = "liability",    # used if enable_staking_plan_search=False
#     staking_plan_back: str = "flat",
#
#     # BACK staking knobs
#     back_stake_test: float = 1.0,
#     back_edge_scale: float = 0.10,
#     kelly_fraction_back: float = 0.25,
#     bankroll_back: float = 100.0,
#     min_back_stake: float = 0.0,
#     max_back_stake: float = 10.0,
#
#     # ── CLASSIFY odds & side ───────────────────────────────────────────────
#     classify_stake: float = 1.0,             # used for BACK classify if odds column present
#     classify_odds_column: str | None = None, # e.g. 'away_odds'
#     classify_side: str = "back",             # "back" | "lay"
#     classify_odds_min_grid=None,             # np.ndarray | None
#     classify_odds_max_grid=None,             # np.ndarray | None
#
#     # ── CLASSIFY-LAY staking (fixed by spec) ───────────────────────────────
#     classify_lay_flat_stake_net_win: float = 1.0,  # target net win (+1 after commission)
#     classify_lay_liability_unit: float = 1.0,      # risk exactly 1 unit
#
#     # ── COMMISSION (applied only to winning returns) ───────────────────────
#     commission_rate: float = 0.02,
#
#     # ── OUTPUTS ────────────────────────────────────────────────────────────
#     save_bets_csv: bool = False,
#     bets_csv_dir: str | None = None,
#     plot_pl: bool = False,
#     plot_dir: str | None = None,
#     plot_title_suffix: str = "",
#     save_all_bets_csv: bool = False,
#     all_bets_dir: str | None = None,
#     all_bets_include_failed: bool = True,
# ):
#     """
#     Full pipeline with:
#       • Safe random hyperparameter sampler (no ParameterSampler int32 issues).
#       • VALUE vs CLASSIFY modes:
#            - VALUE uses fair-vs-market edge rules.
#            - CLASSIFY uses unified rule 'bet when P(success) >= threshold'.
#       • LAY minimum-odds sweep for VALUE-LAY and CLASSIFY-LAY; best min odds kept.
#       • LAY staking patch:
#            - flat_stake: set S = 1/(1-c) so net win = +1 after commission; liability L = S*(o-1).
#            - liability:  set L = 1; stake S = 1/(o-1). No clipping; always place if selected.
#       • Commission applied only on winning returns; losses are not charged commission.
#       • **All required directories are created if missing** (PKL, CSV, plots, bets, all_bets).
#     Returns dict with file paths and summary tables.
#     """
#     # ---------------- setup ----------------
#     import os, secrets, hashlib, json
#     from datetime import datetime
#     import numpy as np
#     import pandas as pd
#     from itertools import product
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
#     # ---------- path helpers (ensure dirs exist everywhere) ----------
#     def _normpath(p: str | None) -> str | None:
#         if p is None: return None
#         return os.path.normpath(p)
#
#     def _ensure_dir(path: str | None):
#         """Create directory if missing (no-op for None)."""
#         if path is None: return
#         os.makedirs(path, exist_ok=True)
#
#     def _ensure_parent(path_to_file: str | None):
#         """Create parent directory for a file path."""
#         if path_to_file is None: return
#         parent = os.path.dirname(path_to_file)
#         if parent:
#             os.makedirs(parent, exist_ok=True)
#
#     # ---------- optional XGBoost ----------
#     try:
#         import xgboost as xgb
#         _HAS_XGB_LOCAL = True
#     except Exception:
#         _HAS_XGB_LOCAL = False
#     _HAS_XGB = globals().get("_HAS_XGB", _HAS_XGB_LOCAL)
#     if base_model == "xgb" and not _HAS_XGB:
#         raise ImportError("XGBoost not available; set base_model='mlp' or install xgboost.")
#
#     # ---------- distributions for random search (SciPy) ----------
#     try:
#         from scipy.stats import randint as _randint
#         from scipy.stats import uniform as _uniform
#         from scipy.stats import loguniform as _loguniform
#     except Exception as _e:
#         raise ImportError("Please install SciPy: pip install scipy") from _e
#
#     # ---------- maths helpers ----------
#     def _Z(conf):
#         try:
#             from scipy.stats import norm
#             return float(norm.ppf(1 - (1 - conf) / 2))
#         except Exception:
#             return 2.576 if conf >= 0.99 else 1.96
#
#     def _Phi(z):
#         try:
#             from scipy.stats import norm
#             return float(norm.cdf(z))
#         except Exception:
#             import math
#             return 0.5 * (1.0 + math.erf(z / (2**0.5)))
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
#     # ---------- defaults ----------
#     if thresholds is None:
#         thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)
#     if value_edge_grid_lay is None:
#         value_edge_grid_lay = np.round(np.arange(0.00, 0.201, 0.01), 2)
#     if value_edge_grid_back is None:
#         value_edge_grid_back = np.round(np.arange(0.00, 0.201, 0.01), 2)
#     if lay_min_odds_grid is None:
#         lay_min_odds_grid = np.round(np.arange(1.05, 3.51, 0.05), 2)
#     if classify_odds_min_grid is None:
#         classify_odds_min_grid = np.round(np.arange(1.00, 10.01, 0.25), 2)
#     if classify_odds_max_grid is None:
#         classify_odds_max_grid = np.round(np.arange(1.00, 10.01, 0.25), 2)
#
#     classify_side = str(classify_side).lower().strip()
#     if classify_side not in ("back", "lay"):
#         raise ValueError("classify_side must be 'back' or 'lay'")
#
#     if staking_plan_lay_options is None:
#         staking_plan_lay_options = ["liability", "flat_stake", "edge_prop", "kelly_approx"]
#     if staking_plan_back_options is None:
#         staking_plan_back_options = ["flat", "edge_prop", "kelly"]
#     if not enable_staking_plan_search:
#         staking_plan_lay_options = [staking_plan_lay]
#         staking_plan_back_options = [staking_plan_back]
#
#     # ---------- dirs (create if missing) ----------
#     BASE = _normpath(r"C:\Users\leere\PycharmProjects\Football_ML3\FT Results")
#     _ensure_dir(BASE)
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
#     if market not in PKL_DIRS:
#         raise ValueError(f"Unsupported market '{market}'.")
#
#     # normalise dirs & ensure they exist
#     for d in PKL_DIRS.values(): _ensure_dir(_normpath(d))
#     for d in CSV_DIRS.values(): _ensure_dir(_normpath(d))
#
#     _IS_LAY  = market.startswith("LAY_")
#     _IS_BACK = market.startswith("BACK_")
#     _USE_VALUE_LAY  = bool(use_value_for_lay and _IS_LAY)
#     _USE_VALUE_BACK = bool(use_value_for_back and _IS_BACK)
#     _USE_VALUE = _USE_VALUE_LAY or _USE_VALUE_BACK
#     _IS_CLASSIFY = not _USE_VALUE
#
#     csv_save_dir = _normpath(out_dir) if (out_dir and len(str(out_dir)) > 0) else _normpath(CSV_DIRS[market])
#     _ensure_dir(csv_save_dir)
#
#     model_dir = _normpath(PKL_DIRS[market]); _ensure_dir(model_dir)
#
#     if bets_csv_dir is None: bets_csv_dir = csv_save_dir
#     if plot_dir is None: plot_dir = csv_save_dir
#     bets_csv_dir = _normpath(bets_csv_dir); _ensure_dir(bets_csv_dir)
#     plot_dir = _normpath(plot_dir); _ensure_dir(plot_dir)
#
#     if all_bets_dir is None:
#         all_bets_dir = os.path.join(os.path.dirname(CSV_DIRS[market]), "all_bets")
#     all_bets_dir = _normpath(all_bets_dir); _ensure_dir(all_bets_dir)
#
#     RUN_SEED = secrets.randbits(32)
#     def _seed_from(*vals) -> int:
#         h = hashlib.blake2b(digest_size=8)
#         h.update(int(RUN_SEED).to_bytes(8,'little',signed=False))
#         for v in vals: h.update(str(v).encode('utf-8'))
#         return int.from_bytes(h.digest(),'little') & 0x7FFFFFFF
#
#     # ---------------- data ----------------
#     df = matches_filtered.copy()
#     req_cols = {'date','target'}
#     if _USE_VALUE: req_cols |= {'home_odds','draw_odds','away_odds'}
#     missing = req_cols - set(df.columns)
#     if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
#
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
#     X_train_final = X.iloc[:max(0, test_start - val_len)]
#     y_train_final = y.iloc[:max(0, test_start - val_len)]
#     X_val_final   = X.iloc[max(0, test_start - val_len):test_start]
#     y_val_final   = y.iloc[max(0, test_start - val_len):test_start]
#
#     # ---------------- model param spaces ----------------
#     xgb_param_grid = {
#         'n_estimators': [200],
#         'max_depth': [5],
#         'learning_rate': [0.1],
#         'subsample': [0.7],
#         'colsample_bytree': [1.0],
#         'min_child_weight': [5],
#         'reg_lambda': [1.0],
#     }
#     xgb_param_distributions = {
#         'n_estimators':     _randint(100, 1001),
#         'max_depth':        _randint(3, 8),
#         'learning_rate':    _loguniform(0.01, 0.2),
#         'min_child_weight': _randint(3, 13),
#         'subsample':        _uniform(0.7, 0.3),
#         'colsample_bytree': _uniform(0.6, 0.4),
#         'reg_lambda':       _loguniform(0.1, 10.0),
#     }
#     mlp_param_grid = {
#         'hidden_layer_sizes': [(128,), (256,), (128, 64)],
#         'alpha': [1e-4],
#         'learning_rate_init': [1e-3],
#         'batch_size': ['auto'],
#         'max_iter': [200],
#     }
#     mlp_param_distributions = {
#         'hidden_layer_sizes': [(64,), (128,), (256,), (128, 64), (256, 128)],
#         'alpha':              _loguniform(1e-5, 1e-2),
#         'learning_rate_init': _loguniform(5e-4, 5e-2),
#         'batch_size':         _randint(32, 257),
#         'max_iter':           _randint(150, 401),
#     }
#
#     def _cast_params(p: dict) -> dict:
#         q = dict(p)
#         if base_model == "xgb":
#             for k in ['n_estimators','max_depth','min_child_weight']:
#                 if k in q: q[k] = int(round(float(q[k])))
#             for k in ['learning_rate','subsample','colsample_bytree','reg_lambda']:
#                 if k in q: q[k] = float(q[k])
#         else:
#             if 'max_iter' in q: q['max_iter'] = int(round(float(q['max_iter'])))
#             if 'batch_size' in q and q['batch_size'] != 'auto': q['batch_size'] = int(round(float(q['batch_size'])))
#             for k in ['alpha','learning_rate_init']:
#                 if k in q: q[k] = float(q[k])
#             if 'hidden_layer_sizes' in q and isinstance(q['hidden_layer_sizes'], (list, tuple)):
#                 q['hidden_layer_sizes'] = tuple(int(v) for v in q['hidden_layer_sizes'])
#         return q
#
#     # ---------- safe random sampler (avoids sklearn ParameterSampler bug) ----------
#     def _safe_random_param_sets(dists: dict, n_iter: int, seed: int) -> list[dict]:
#         import numpy as _np
#         rng = _np.random.RandomState(seed)
#         keys = list(dists.keys())
#         out = []
#         for _ in range(int(n_iter)):
#             params = {}
#             for k in keys:
#                 v = dists[k]
#                 if hasattr(v, "rvs"):
#                     params[k] = v.rvs(random_state=rng)
#                 elif isinstance(v, (list, tuple)):
#                     if len(v) == 0:
#                         raise ValueError(f"Empty choices for '{k}'")
#                     params[k] = v[rng.randint(0, len(v))]
#                 else:
#                     params[k] = v
#             out.append(params)
#         return out
#
#     # ---------------- model builders ----------------
#     def _final_step_name(estimator):
#         try:
#             if isinstance(estimator, Pipeline): return estimator.steps[-1][0]
#         except Exception:
#             pass
#         return None
#
#     def _build_model(params: dict, spw: float):
#         seed = _seed_from("model", base_model, *sorted(map(str, params.items())))
#         if base_model == "xgb":
#             return xgb.XGBClassifier(
#                 objective='binary:logistic',
#                 eval_metric='auc',
#                 random_state=seed,
#                 scale_pos_weight=spw,
#                 n_jobs=1,
#                 tree_method="hist",
#                 verbosity=0,
#                 **params
#             )
#         else:
#             mlp = MLPClassifier(
#                 random_state=seed,
#                 early_stopping=True,
#                 n_iter_no_change=20,
#                 validation_fraction=0.1,
#                 solver="adam",
#                 **params
#             )
#             return make_pipeline(StandardScaler(with_mean=True, with_std=True), mlp)
#
#     def _fit_model(model, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None):
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
#     def _fit_calibrator(fitted, X_va, y_va):
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
#     def _predict_proba_pos(model_or_cal, X_):
#         proba = model_or_cal.predict_proba(X_)
#         if proba.ndim == 2:
#             classes = getattr(model_or_cal, "classes_", None)
#             if classes is None:
#                 base = _unwrap_estimator(model_or_cal); classes = getattr(base, "classes_", None)
#             if classes is not None and len(classes) == proba.shape[1]:
#                 import numpy as _np
#                 try:
#                     idx = int(_np.where(_np.asarray(classes) == 1)[0][0])
#                     return proba[:, idx].astype(np.float32)
#                 except Exception:
#                     pass
#             if proba.shape[1] == 2: return proba[:, 1].astype(np.float32)
#             if proba.shape[1] == 1:
#                 only = getattr(model_or_cal, "classes_", [0])[0]
#                 return (np.ones_like(proba[:,0]) if only==1 else np.zeros_like(proba[:,0])).astype(np.float32)
#         return np.asarray(proba, dtype=np.float32)
#
#     # --- p-value helper (commission on wins only) ---
#     def _pvalue_break_even(bdf: pd.DataFrame, mode: str) -> dict:
#         if not isinstance(bdf, pd.DataFrame) or bdf.empty:
#             return {'z': 0.0, 'p_value': 1.0, 'var_sum': 0.0, 'n': 0, 'total_pl': 0.0}
#         o = np.asarray(bdf['market_odds'].values, dtype=float)
#         o = np.where(o <= 1.0, np.nan, o)
#         p_null_win = 1.0 / o
#         if mode == 'VALUE_BACK':
#             S = np.asarray(bdf['stake'].values, dtype=float)
#             win = (o - 1.0) * S * (1.0 - commission_rate)
#             lose = -S
#         else:  # VALUE_LAY
#             L = np.asarray(bdf.get('liability', np.nan*np.ones_like(o))).astype(float)
#             S = np.asarray(bdf['stake'].values, dtype=float)
#             win = S * (1.0 - commission_rate)    # selection loses
#             lose = -L                             # selection wins
#         var_i = p_null_win * (win ** 2) + (1.0 - p_null_win) * (lose ** 2)
#         var_i = np.where(np.isfinite(var_i), var_i, 0.0)
#         pl = np.asarray(bdf['pl'].values, dtype=float)
#         total_pl = float(np.nansum(np.where(np.isfinite(pl), pl, 0.0)))
#         var_sum = float(np.nansum(var_i))
#         z = total_pl / (np.sqrt(var_sum) + 1e-12)
#         p_val = max(0.0, 1.0 - _Phi(z))  # one-sided
#         return {'z': float(z), 'p_value': float(p_val), 'var_sum': var_sum, 'n': int(len(pl)), 'total_pl': total_pl}
#
#     # ---------------- VALUE staking calculators ----------------------------
#     def _lay_stakes_value(o: np.ndarray, edge_ratio_minus1: np.ndarray, plan: str):
#         # LAY (NO clipping; always place the bet once selected):
#         #   - flat_stake: S = 1/(1-c), L = S*(o-1)  (win +1 net after commission)
#         #   - liability:  L = 1, S = 1/(o-1)
#         o = np.asarray(o, dtype=float)
#         edge_plus = np.maximum(np.asarray(edge_ratio_minus1, dtype=float), 0.0)
#         denom = np.maximum(o - 1.0, 1e-12)
#
#         if plan == "liability":
#             L = np.ones_like(o, dtype=float)
#             S = L / denom
#             ok = np.ones_like(S, dtype=bool)
#             return S, L, ok
#
#         if plan == "flat_stake":
#             S = np.full_like(o, 1.0 / max(1e-12, 1.0 - commission_rate), dtype=float)
#             L = S * denom
#             ok = np.ones_like(S, dtype=bool)
#             return S, L, ok
#
#         if plan == "edge_prop":
#             scale = max(1e-12, float(back_edge_scale))
#             L = edge_plus / scale
#             S = L / denom
#             ok = np.ones_like(S, dtype=bool)
#             return S, L, ok
#
#         if plan == "kelly_approx":
#             L = 1.0 * edge_plus
#             S = L / denom
#             ok = np.ones_like(S, dtype=bool)
#             return S, L, ok
#
#         raise ValueError(f"Unknown staking_plan_lay: {plan}")
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
#     # ---------------- search grid / samples --------------------------------
#     if search_mode.lower() == "grid":
#         grid = xgb_param_grid if base_model == "xgb" else mlp_param_grid
#         keys = list(grid.keys()); vals = [grid[k] for k in keys]
#         all_param_dicts = [dict(zip(keys, combo)) for combo in product(*vals)]
#     else:
#         dists = xgb_param_distributions if base_model == "xgb" else mlp_param_distributions
#         sampler_seed = _seed_from("sampler")
#         all_param_dicts = _safe_random_param_sets(dists, n_random_param_sets, sampler_seed)
#
#     # ---------------- validation evaluation --------------------------------
#     def _evaluate_param_set(param_dict):
#         safe = _cast_params(param_dict)
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
#             model = _build_model(safe, spw)
#             _fit_model(model, X_tr, y_tr, X_va, y_va, sample_weight=sample_weight)
#             cal = _fit_calibrator(model, X_va, y_va)
#
#             p_pos = _predict_proba_pos(cal, X_va)
#             val_prob_all.append(p_pos)
#             y_true = y_va.values.astype(np.uint8); val_true_all.append(y_true)
#
#             if not _USE_VALUE:
#                 # CLASSIFY validation
#                 if classify_odds_column is None or classify_odds_column not in df_va.columns:
#                     for thr in thresholds:
#                         take = (p_pos >= float(thr))
#                         y_pred = take.astype(np.uint8)
#                         rows.append({
#                             **safe,'threshold':float(thr),'odds_min':np.nan,'odds_max':np.nan,
#                             'fold_vstart':int(vstart),'fold_vend':int(vend),
#                             'n_preds_val':int(y_pred.sum()),
#                             'tp_val':int(((y_true==1)&(y_pred==1)).sum()),
#                             'fp_val':int(((y_true==0)&(y_pred==1)).sum()),
#                             'val_precision':float(precision_score(y_va,y_pred,zero_division=0)),
#                             'val_accuracy':float(accuracy_score(y_va,y_pred)),
#                             'n_value_bets_val':int(y_pred.sum()),
#                             'val_edge_ratio_mean':np.nan,'val_edge_ratio_mean_back':np.nan
#                         })
#                 else:
#                     mkt = df_va[classify_odds_column].values.astype(float)
#                     valid = np.isfinite(mkt) & (mkt > 1.01)
#                     for thr in thresholds:
#                         pred_mask = (p_pos >= float(thr))
#                         for omin in classify_odds_min_grid:
#                             for omax in classify_odds_max_grid:
#                                 omin=float(omin); omax=float(omax)
#                                 if omin>omax: continue
#                                 odds_mask = valid & (mkt >= omin) & (mkt <= omax)
#                                 take = pred_mask & odds_mask
#                                 y_pred = take.astype(np.uint8)
#                                 rows.append({
#                                     **safe,'threshold':float(thr),'odds_min':omin,'odds_max':omax,
#                                     'fold_vstart':int(vstart),'fold_vend':int(vend),
#                                     'n_preds_val':int(y_pred.sum()),
#                                     'tp_val':int(((y_true==1)&(y_pred==1)).sum()),
#                                     'fp_val':int(((y_true==0)&(y_pred==1)).sum()),
#                                     'val_precision':float(precision_score(y_va,y_pred,zero_division=0)),
#                                     'val_accuracy':float(accuracy_score(y_va,y_pred)),
#                                     'n_value_bets_val':int(y_pred.sum()),
#                                     'val_edge_ratio_mean':np.nan,'val_edge_ratio_mean_back':np.nan
#                                 })
#             else:
#                 # VALUE validation (edge rule only; LAY min-odds is swept in TEST)
#                 if _IS_LAY:
#                     mkt = df_va['away_odds'].values if market=="LAY_AWAY" else (df_va['home_odds'].values if market=="LAY_HOME" else df_va['draw_odds'].values)
#                     fair = 1.0 / np.clip(1.0 - p_pos, 1e-9, 1.0)
#                     for edge_param in value_edge_grid_lay:
#                         edge_mask = (fair >= (1.0 + float(edge_param)) * mkt) & np.isfinite(mkt)
#                         with np.errstate(divide='ignore', invalid='ignore'):
#                             edge_ratio = fair / mkt
#                         y_pred = edge_mask.astype(np.uint8)
#                         rows.append({
#                             **safe,'threshold':np.nan,'odds_min':np.nan,'odds_max':np.nan,'edge_param':float(edge_param),
#                             'fold_vstart':int(vstart),'fold_vend':int(vend),
#                             'n_preds_val':int(y_pred.sum()),
#                             'tp_val':int(((y_true==1)&(y_pred==1)).sum()),
#                             'fp_val':int(((y_true==0)&(y_pred==1)).sum()),
#                             'val_precision':float(precision_score(y_true,y_pred,zero_division=0)),
#                             'val_accuracy':float(accuracy_score(y_true,y_pred)),
#                             'n_value_bets_val':int(y_pred.sum()),
#                             'val_edge_ratio_mean':float(np.nanmean(np.where(np.isfinite(edge_ratio), edge_ratio, np.nan))),
#                             'val_edge_ratio_mean_back':np.nan
#                         })
#                 else:
#                     mkt = df_va['away_odds'].values if market=="BACK_AWAY" else (df_va['home_odds'].values if market=="BACK_HOME" else df_va['draw_odds'].values)
#                     fair = 1.0 / np.clip(p_pos, 1e-9, 1.0)
#                     for edge_param in value_edge_grid_back:
#                         edge_mask = (mkt >= (1.0 + float(edge_param)) * fair) & np.isfinite(mkt)
#                         with np.errstate(divide='ignore', invalid='ignore'):
#                             edge_ratio = mkt / fair
#                         y_pred = edge_mask.astype(np.uint8)
#                         rows.append({
#                             **safe,'threshold':np.nan,'odds_min':np.nan,'odds_max':np.nan,'edge_param':float(edge_param),
#                             'fold_vstart':int(vstart),'fold_vend':int(vend),
#                             'n_preds_val':int(y_pred.sum()),
#                             'tp_val':int(((y_true==1)&(y_pred==1)).sum()),
#                             'fp_val':int(((y_true==0)&(y_pred==1)).sum()),
#                             'val_precision':float(precision_score(y_true,y_pred,zero_division=0)),
#                             'val_accuracy':float(accuracy_score(y_true,y_pred)),
#                             'n_value_bets_val':int(y_pred.sum()),
#                             'val_edge_ratio_mean':np.nan,
#                             'val_edge_ratio_mean_back':float(np.nanmean(np.where(np.isfinite(edge_ratio), edge_ratio, np.nan))),
#                         })
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
#     # ---------------- run search -------------------------------------------
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
#                     delayed(_evaluate_param_set)(_cast_params(pd_)) for pd_ in all_param_dicts
#                 )
#         except OSError as e:
#             print(f"[WARN] Parallel failed with {e}. Falling back to serial search...")
#             out = []
#             for pd_ in tqdm(all_param_dicts, desc=f"Param search (serial, {base_model})"):
#                 out.append(_evaluate_param_set(pd_))
#
#     val_rows = [r for sub in out for r in sub]
#     if not val_rows: raise RuntimeError("No validation rows produced (check folds and input data).")
#     val_df = pd.DataFrame(val_rows)
#
#     # ---------------- validation aggregate ---------------------------------
#     if base_model == "xgb":
#         param_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']
#     else:
#         param_keys = ['hidden_layer_sizes','alpha','learning_rate_init','batch_size','max_iter']
#
#     if _IS_CLASSIFY:
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
#         fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv") if save_diagnostics_on_fail else None
#         if save_diagnostics_on_fail:
#             diag = (agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                                     ascending=[False, False, False, False])
#                     .assign(fail_reason="failed_validation_gate", market=market))
#             _ensure_parent(fail_csv)
#             diag.to_csv(fail_csv, index=False)
#         if on_fail == "raise": raise RuntimeError("No strategy met validation gates.")
#         if on_fail == "warn": print("[WARN] No strategy met the validation gate.")
#         return {'status':'failed_validation_gate','csv':fail_csv,'model_pkl':None,
#                 'summary_df':None,'validation_table':agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                 ascending=[False,False,False,False]).reset_index(drop=True)}
#
#     ranked = qual.sort_values(by=['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
#                               ascending=[False, False, False, False]).reset_index(drop=True)
#     top_k = int(top_k)
#     topk_val = ranked.head(top_k).reset_index(drop=True)
#
#     def _extract_params_from_row(row):
#         return _cast_params({k: row[k] for k in param_keys if k in row.index})
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
#     # ---------------- TEST EVAL (with LAY min-odds sweep) ------------------
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
#     def _choose_best(df_):
#         if ('p_value' in df_.columns) and df_['p_value'].notna().any():
#             return df_.sort_values(['p_value','total_pl','avg_pl','val_precision_lcb','val_precision'],
#                                    ascending=[True, False, False, False, False]).iloc[0]
#         return df_.sort_values(['val_precision_lcb','val_precision','test_precision','n_preds_test','val_accuracy'],
#                                ascending=[False, False, False, False, False]).iloc[0]
#
#     for cand_id, cand in enumerate(candidates):
#         best_params = _cast_params(cand['params'])
#         pos = int(y_train_final.sum()); neg = len(y_train_final) - pos
#         spw_final = (neg/pos) if pos > 0 else 1.0
#
#         final_model = _build_model(best_params, spw_final)
#         final_sample_weight = None
#         if base_model == "mlp":
#             w_pos = spw_final
#             final_sample_weight = np.where(y_train_final.values==1, w_pos, 1.0).astype(np.float32)
#
#         _fit_model(final_model, X_train_final, y_train_final, X_val_final, y_val_final, sample_weight=final_sample_weight)
#         final_calibrator = _fit_calibrator(final_model, X_val_final, y_val_final)
#         p_pos_test = _predict_proba_pos(final_calibrator, X_test)
#
#         if _USE_VALUE:
#             # ===== VALUE modes =====
#             if _IS_LAY:
#                 # odds & fair
#                 if market == "LAY_AWAY":
#                     mkt_odds = df_test['away_odds'].values.astype(float); sel_name = 'AWAY'
#                 elif market == "LAY_HOME":
#                     mkt_odds = df_test['home_odds'].values.astype(float); sel_name = 'HOME'
#                 else:
#                     mkt_odds = df_test['draw_odds'].values.astype(float); sel_name = 'DRAW'
#                 fair_odds = 1.0 / np.clip(1.0 - p_pos_test, 1e-9, 1.0)
#                 valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01)
#
#                 edge = float(cand.get('edge_param', 0.0))
#                 with np.errstate(divide='ignore', invalid='ignore'):
#                     edge_ratio_minus1 = np.where(mkt_odds > 0, fair_odds / mkt_odds - 1.0, 0.0)
#                 base_value = valid & (fair_odds >= (1.0 + edge) * mkt_odds)
#
#                 per_min_records = []
#                 for min_o in lay_min_odds_grid:
#                     take_mask = base_value & (mkt_odds >= float(min_o))
#
#                     for plan in staking_plan_lay_options:
#                         S_plan, L_plan, _ = _lay_stakes_value(mkt_odds, edge_ratio_minus1, plan)
#                         S = np.where(take_mask, S_plan, 0.0)
#                         L = np.where(take_mask, L_plan, 0.0)
#
#                         sel_occurs = (y_test.values == 0)  # selection occurs => lay loses
#                         pl = np.where(~take_mask, 0.0,
#                                       np.where(sel_occurs, -L, S * (1.0 - commission_rate)))
#
#                         n_bets = int(np.count_nonzero(S > 0))
#                         y_pred = (S > 0).astype(np.uint8)
#                         prc_test = precision_score(y_test, y_pred, zero_division=0)
#                         acc_test = accuracy_score(y_test, y_pred)
#                         total_pl = float(pl.sum())
#                         avg_pl = float(total_pl / max(1, n_bets))
#
#                         if n_bets > 0:
#                             bet_idx = np.where(S > 0)[0]
#                             name_cols = _name_cols(df_test.iloc[bet_idx])
#                             bets_df = pd.DataFrame({
#                                 **name_cols,
#                                 'selection': sel_name,
#                                 'market_odds': mkt_odds[bet_idx],
#                                 'fair_odds': fair_odds[bet_idx],
#                                 'edge_ratio': np.where(mkt_odds[bet_idx] > 0, fair_odds[bet_idx] / mkt_odds[bet_idx], np.nan),
#                                 'liability': L[bet_idx],
#                                 'stake': S[bet_idx],
#                                 'commission_rate': float(commission_rate),
#                                 'selection_occurred': sel_occurs[bet_idx].astype(int),
#                                 'target': y_test.values[bet_idx],
#                                 'pl': pl[bet_idx],
#                             })
#                             if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
#                             bets_df['cum_pl'] = bets_df['pl'].cumsum()
#                             pv = _pvalue_break_even(bets_df, mode='VALUE_LAY')
#                         else:
#                             bets_df = None; pv = {'p_value': np.nan, 'z': np.nan}
#
#                         per_min_records.append(pd.Series({
#                             **best_params,
#                             'mode': 'VALUE_LAY',
#                             'edge_param': edge,
#                             'staking_plan_lay': plan,
#                             'lay_min_odds': float(min_o),
#                             'n_value_bets_test': int(n_bets),
#                             'test_precision_bets': float(prc_test),
#                             'test_accuracy_bets': float(acc_test),
#                             'total_pl': float(total_pl),
#                             'avg_pl': float(avg_pl),
#                             'p_value': float(pv['p_value']) if pd.notna(pv['p_value']) else np.nan,
#                             'zscore': float(pv['z']) if pd.notna(pv['z']) else np.nan,
#                             'val_precision_lcb': float(cand['val_precision_lcb']),
#                             'val_precision': float(cand['val_precision']),
#                             'val_accuracy': float(cand['val_accuracy']),
#                             'model_obj': final_calibrator if (n_bets >= int(min_test_samples) and prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop))) else None,
#                             'bets_obj': bets_df
#                         }))
#
#                 per_min_df = pd.DataFrame(per_min_records)
#                 if len(per_min_df):
#                     per_min_df['pass_test_gate'] = (
#                         (per_min_df['n_value_bets_test'] >= int(min_test_samples)) &
#                         (per_min_df['test_precision_bets'] >=
#                          np.maximum(float(precision_test_threshold),
#                                     float(cand['val_precision']) - float(max_precision_drop)))
#                     )
#                     passers_df = per_min_df[per_min_df['pass_test_gate']].copy()
#                     chosen = _choose_best(passers_df) if not passers_df.empty else _choose_best(per_min_df.fillna({'p_value': np.inf}))
#                     bets_df = chosen['bets_obj']
#                     rec = {
#                         **{k: chosen.get(k) for k in best_params.keys()},
#                         'mode': 'VALUE_LAY',
#                         'edge_param': float(chosen['edge_param']),
#                         'staking_plan_lay': chosen['staking_plan_lay'],
#                         'lay_min_odds': float(chosen['lay_min_odds']),
#                         'val_precision_lcb': float(chosen['val_precision_lcb']),
#                         'val_precision': float(chosen['val_precision']),
#                         'val_accuracy': float(chosen['val_accuracy']),
#                         'n_value_bets_test': int(chosen['n_value_bets_test']),
#                         'test_precision_bets': float(chosen['test_precision_bets']),
#                         'test_accuracy_bets': float(chosen['test_accuracy_bets']),
#                         'total_pl': float(chosen['total_pl']),
#                         'avg_pl': float(chosen['avg_pl']),
#                         'p_value': float(chosen['p_value']) if pd.notna(chosen['p_value']) else np.nan,
#                         'zscore': float(chosen['zscore']) if pd.notna(chosen['zscore']) else np.nan,
#                         'pass_test_gate': bool(chosen['pass_test_gate']),
#                         'fail_reason': "" if bool(chosen['pass_test_gate']) else "failed_min_odds_gate",
#                         'model_obj': chosen['model_obj'] if bool(chosen['pass_test_gate']) else None,
#                         'bets': bets_df if (bool(chosen['pass_test_gate']) and isinstance(bets_df, pd.DataFrame)) else None,
#                         'commission_rate': float(commission_rate),
#                     }
#                     records_all.append(rec)
#                     if isinstance(bets_df, pd.DataFrame) and len(bets_df):
#                         md = dict(candidate_id=cand_id, passed_test_gate=bool(chosen['pass_test_gate']),
#                                   mode='VALUE_LAY', market=market, edge_param=float(chosen['edge_param']),
#                                   lay_min_odds=float(chosen['lay_min_odds']),
#                                   commission_rate=float(commission_rate),
#                                   params_json=json.dumps(best_params, default=float))
#                         bdf = bets_df.copy()
#                         for k, v in md.items(): bdf[k] = v
#                         all_bets_collector.append(bdf)
#
#             else:
#                 # VALUE BACK
#                 if market == "BACK_AWAY":
#                     mkt_odds = df_test['away_odds'].values.astype(float); sel_name = 'AWAY'
#                 elif market == "BACK_HOME":
#                     mkt_odds = df_test['home_odds'].values.astype(float); sel_name = 'HOME'
#                 else:
#                     mkt_odds = df_test['draw_odds'].values.astype(float); sel_name = 'DRAW'
#                 p_sel_win = p_pos_test
#                 fair_odds = 1.0 / np.clip(p_sel_win, 1e-9, 1.0)
#                 valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01)
#
#                 edge = float(cand.get('edge_param', 0.0))
#                 with np.errstate(divide='ignore', invalid='ignore'):
#                     edge_ratio_minus1 = np.where(fair_odds > 0, mkt_odds / fair_odds - 1.0, 0.0)
#                 value_mask = valid & (mkt_odds >= (1.0 + edge) * fair_odds)
#
#                 for plan in staking_plan_back_options:
#                     stake = np.zeros_like(mkt_odds, dtype=float)
#                     s = _back_stakes(mkt_odds, edge_ratio_minus1, plan, p_sel_win)
#                     stake[value_mask] = s[value_mask]
#
#                     sel_occurs = (y_test.values == 1)
#                     pl = np.where((stake > 0) & sel_occurs,
#                                   (mkt_odds - 1.0) * stake * (1.0 - commission_rate),
#                                   0.0)
#                     pl = np.where((stake > 0) & (~sel_occurs), -stake, pl)
#
#                     n_bets = int(np.count_nonzero(stake > 0))
#                     total_pl = float(pl.sum()); avg_pl = float(total_pl / max(1, n_bets))
#                     y_pred = (stake > 0).astype(np.uint8)
#                     prc_test = precision_score(y_test, y_pred, zero_division=0)
#                     acc_test = accuracy_score(y_test, y_pred)
#
#                     bet_idx = np.where(stake > 0)[0]
#                     if len(bet_idx):
#                         name_cols = _name_cols(df_test.iloc[bet_idx])
#                         bets_df = pd.DataFrame({
#                             **name_cols,
#                             'selection': sel_name,
#                             'market_odds': mkt_odds[bet_idx],
#                             'fair_odds': fair_odds[bet_idx],
#                             'edge_ratio': np.where(fair_odds[bet_idx] > 0, mkt_odds[bet_idx] / fair_odds[bet_idx], np.nan),
#                             'stake': stake[bet_idx],
#                             'commission_rate': float(commission_rate),
#                             'selection_occurred': sel_occurs[bet_idx].astype(int),
#                             'target': y_test.values[bet_idx],
#                             'pl': pl[bet_idx],
#                         }).sort_values('date' if 'date' in df_test.columns else 'pl')
#                         bets_df['cum_pl'] = bets_df['pl'].cumsum()
#                         pv = _pvalue_break_even(bets_df, mode='VALUE_BACK')
#                     else:
#                         bets_df = None; pv = {'p_value': np.nan, 'z': np.nan}
#
#                     pass_gate = (n_bets >= int(min_test_samples)) and (prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop)))
#                     records_all.append({
#                         **best_params, 'mode': 'VALUE_BACK',
#                         'val_precision_lcb': cand['val_precision_lcb'], 'val_precision': cand['val_precision'],
#                         'val_accuracy': cand['val_accuracy'],
#                         'n_value_bets_test': n_bets, 'test_precision_bets': float(prc_test),
#                         'test_accuracy_bets': float(acc_test), 'total_pl': total_pl, 'avg_pl': avg_pl,
#                         'p_value': pv['p_value'], 'zscore': pv['z'],
#                         'pass_test_gate': bool(pass_gate), 'fail_reason': "" if pass_gate else "insufficient_or_collapse",
#                         'model_obj': final_calibrator if pass_gate else None,
#                         'bets': bets_df if pass_gate else None,
#                         'commission_rate': float(commission_rate),
#                     })
#
#         else:
#             # ===== CLASSIFY path =====
#             thr = float(cand['threshold'])
#             o = df_test[classify_odds_column].values.astype(float) if (classify_odds_column and classify_odds_column in df_test.columns) else None
#             if o is not None:
#                 valid = np.isfinite(o) & (o > 1.01)
#                 omin = cand.get('odds_min', np.nan); omax = cand.get('odds_max', np.nan)
#                 if np.isnan(omin) or np.isnan(omax):
#                     odds_mask = valid
#                 else:
#                     odds_mask = valid & (o >= float(omin)) & (o <= float(omax))
#             else:
#                 odds_mask = np.ones(len(X_test), dtype=bool)
#
#             pred_mask = (_predict_proba_pos(final_calibrator, X_test) >= thr)
#             base_take = pred_mask & odds_mask
#
#             if classify_side == "back":
#                 take = base_take
#                 bet_idx = np.where(take)[0]
#                 y_pred = take.astype(np.uint8)
#                 n_preds_test = int(y_pred.sum())
#                 prc_test = precision_score(y_test, y_pred, zero_division=0)
#                 acc_test = accuracy_score(y_test, y_pred)
#
#                 if len(bet_idx) and (o is not None):
#                     mkt_odds = o[bet_idx].astype(float)
#                     sel_occurs = (y_test.values == 1)[bet_idx]
#                     stake = np.full(len(bet_idx), float(classify_stake), dtype=float)
#                     pl = np.where(sel_occurs, (mkt_odds - 1.0) * stake * (1.0 - commission_rate), -stake)
#                     name_cols = _name_cols(df_test.iloc[bet_idx])
#                     bets_df = pd.DataFrame({**name_cols,'selection':'CLASSIFY_BACK','market_odds':mkt_odds,'stake':stake,'pl':pl})
#                     if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
#                     bets_df['cum_pl'] = bets_df['pl'].cumsum()
#                     pv = _pvalue_break_even(bets_df[['market_odds','stake','pl']], mode='VALUE_BACK')
#                     total_pl = float(pl.sum()); avg_pl = float(total_pl / max(1, len(bet_idx)))
#                 else:
#                     bets_df=None; pv={'p_value':np.nan,'z':np.nan}; total_pl=np.nan; avg_pl=np.nan
#
#                 pass_gate = (n_preds_test >= int(min_test_samples)) and (prc_test >= max(float(precision_test_threshold), float(cand['val_precision']) - float(max_precision_drop)))
#                 records_all.append({
#                     **best_params, 'mode': 'CLASSIFY_BACK', 'threshold': thr,
#                     'n_preds_test': n_preds_test, 'test_precision': float(prc_test), 'test_accuracy': float(acc_test),
#                     'total_pl': total_pl, 'avg_pl': avg_pl, 'p_value': pv['p_value'], 'zscore': pv['z'],
#                     'pass_test_gate': bool(pass_gate), 'fail_reason': "" if pass_gate else "insufficient_or_collapse",
#                     'model_obj': final_calibrator if pass_gate else None,
#                     'bets': bets_df if pass_gate else None,
#                     'val_precision_lcb': cand['val_precision_lcb'], 'val_precision': cand['val_precision'], 'val_accuracy': cand['val_accuracy'],
#                     'commission_rate': float(commission_rate),
#                 })
#
#             else:
#                 # CLASSIFY-LAY: sweep lay_min_odds and compare flat-stake vs flat-liability per min-odds
#                 if o is None:
#                     continue
#                 valid = np.isfinite(o) & (o > 1.01)
#                 pred = base_take & valid
#                 sel_occurs_full = (y_test.values == 0)
#
#                 per_min_records = []
#                 for min_o in lay_min_odds_grid:
#                     min_o = float(min_o)
#                     take = pred & (o >= min_o)
#                     bet_idx = np.where(take)[0]
#                     if len(bet_idx) == 0:
#                         per_min_records.append(pd.Series({
#                             **best_params, 'mode':'CLASSIFY_LAY','threshold':thr,
#                             'lay_min_odds': min_o, 'n_preds_test': 0,
#                             'test_precision': 0.0, 'test_accuracy': 0.0,
#                             'total_pl': 0.0, 'avg_pl': 0.0,
#                             'p_value': np.nan, 'zscore': np.nan,
#                             'val_precision_lcb': cand['val_precision_lcb'],
#                             'val_precision': cand['val_precision'],'val_accuracy': cand['val_accuracy'],
#                             'classify_lay_variant': 'none','bets_obj': None,'model_obj': None
#                         }))
#                         continue
#
#                     mkt_odds = o[bet_idx].astype(float)
#                     sel_occurs = sel_occurs_full[bet_idx]
#
#                     # Variant A: flat-stake LAY (win one net unit)
#                     stake_flat = np.full(len(bet_idx), float(classify_lay_flat_stake_net_win) / max(1e-12, 1.0 - commission_rate), dtype=float)
#                     liability_flat = stake_flat * (mkt_odds - 1.0)
#                     pl_flat = np.where(sel_occurs, -liability_flat, stake_flat * (1.0 - commission_rate))
#                     pv_flat = _pvalue_break_even(pd.DataFrame({'market_odds':mkt_odds,'stake':stake_flat,'liability':liability_flat,'pl':pl_flat}), mode='VALUE_LAY')
#
#                     # Variant B: flat-liability LAY (risk one unit)
#                     liability_const = np.full(len(bet_idx), float(classify_lay_liability_unit), dtype=float)
#                     stake_liab = liability_const / np.maximum(mkt_odds - 1.0, 1e-12)
#                     pl_liab = np.where(sel_occurs, -liability_const, stake_liab * (1.0 - commission_rate))
#                     pv_liab = _pvalue_break_even(pd.DataFrame({'market_odds':mkt_odds,'stake':stake_liab,'liability':liability_const,'pl':pl_liab}), mode='VALUE_LAY')
#
#                     choose_liab = (pv_liab['p_value'] < pv_flat['p_value'])
#                     if choose_liab:
#                         total_pl = float(np.sum(pl_liab))
#                         avg_pl = float(np.mean(pl_liab))
#                         p_value = float(pv_liab['p_value']); zscore = float(pv_liab['z'])
#                         variant = 'lay_liability'
#                     else:
#                         total_pl = float(np.sum(pl_flat))
#                         avg_pl = float(np.mean(pl_flat))
#                         p_value = float(pv_flat['p_value']); zscore = float(pv_flat['z'])
#                         variant = 'lay_flat'
#
#                     name_cols = _name_cols(df_test.iloc[bet_idx])
#                     bets_df = pd.DataFrame({
#                         **name_cols, 'selection':'CLASSIFY_LAY',
#                         'market_odds': mkt_odds, 'threshold': thr,
#                         'pl_flat': pl_flat, 'pl_liability': pl_liab,
#                         'chosen_variant': variant
#                     })
#                     if 'date' in bets_df.columns: bets_df = bets_df.sort_values('date').reset_index(drop=True)
#                     bets_df['pl'] = bets_df['pl_liability'] if variant=='lay_liability' else bets_df['pl_flat']
#                     bets_df['cum_pl'] = bets_df['pl'].cumsum()
#
#                     n_preds_test = int(len(bet_idx))
#                     prc_test = precision_score(y_test, (take).astype(np.uint8), zero_division=0)
#                     acc_test = accuracy_score(y_test, (take).astype(np.uint8))
#
#                     per_min_records.append(pd.Series({
#                         **best_params, 'mode':'CLASSIFY_LAY','threshold':thr,
#                         'lay_min_odds': min_o,
#                         'n_preds_test': n_preds_test, 'test_precision': float(prc_test), 'test_accuracy': float(acc_test),
#                         'total_pl': total_pl, 'avg_pl': avg_pl,
#                         'p_value': p_value, 'zscore': zscore,
#                         'val_precision_lcb': cand['val_precision_lcb'],
#                         'val_precision': cand['val_precision'], 'val_accuracy': cand['val_accuracy'],
#                         'classify_lay_variant': variant, 'bets_obj': bets_df, 'model_obj': final_calibrator
#                     }))
#
#                 per_min_df = pd.DataFrame(per_min_records)
#                 if len(per_min_df):
#                     per_min_df['pass_test_gate'] = (
#                         (per_min_df['n_preds_test'] >= int(min_test_samples)) &
#                         (per_min_df['test_precision'] >=
#                          np.maximum(float(precision_test_threshold),
#                                     float(cand['val_precision']) - float(max_precision_drop)))
#                     )
#                     passers_df = per_min_df[per_min_df['pass_test_gate']].copy()
#                     chosen = _choose_best(passers_df) if not passers_df.empty else _choose_best(per_min_df.fillna({'p_value': np.inf}))
#                     bets_df = chosen['bets_obj']
#                     records_all.append({
#                         **{k: chosen.get(k) for k in best_params.keys()},
#                         'mode': 'CLASSIFY_LAY', 'threshold': float(chosen['threshold']),
#                         'lay_min_odds': float(chosen['lay_min_odds']),
#                         'classify_lay_variant': chosen['classify_lay_variant'],
#                         'n_preds_test': int(chosen['n_preds_test']),
#                         'test_precision': float(chosen['test_precision']),
#                         'test_accuracy': float(chosen['test_accuracy']),
#                         'total_pl': float(chosen['total_pl']),
#                         'avg_pl': float(chosen['avg_pl']),
#                         'p_value': float(chosen['p_value']) if pd.notna(chosen['p_value']) else np.nan,
#                         'zscore': float(chosen['zscore']) if pd.notna(chosen['zscore']) else np.nan,
#                         'pass_test_gate': bool(chosen['pass_test_gate']),
#                         'fail_reason': "" if bool(chosen['pass_test_gate']) else "failed_min_odds_gate",
#                         'model_obj': chosen['model_obj'] if bool(chosen['pass_test_gate']) else None,
#                         'bets': bets_df if (bool(chosen['pass_test_gate']) and isinstance(bets_df, pd.DataFrame)) else None,
#                         'val_precision_lcb': float(cand['val_precision_lcb']),
#                         'val_precision': float(cand['val_precision']),
#                         'val_accuracy': float(cand['val_accuracy']),
#                         'commission_rate': float(commission_rate),
#                     })
#                     if isinstance(bets_df, pd.DataFrame) and len(bets_df):
#                         md = dict(candidate_id=cand_id, passed_test_gate=bool(chosen['pass_test_gate']),
#                                   mode='CLASSIFY_LAY', market=market, threshold=float(chosen['threshold']),
#                                   lay_min_odds=float(chosen['lay_min_odds']),
#                                   commission_rate=float(commission_rate),
#                                   params_json=json.dumps(best_params, default=float))
#                         bdf = bets_df.copy()
#                         for k, v in md.items(): bdf[k] = v
#                         all_bets_collector.append(bdf)
#
#     survivors_df = pd.DataFrame(records_all)
#     passers = survivors_df[survivors_df['pass_test_gate']].copy()
#
#     # ---------------- save / rank ------------------------------------------
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
#             _ensure_parent(fail_csv)
#             diag.to_csv(fail_csv, index=False); summary_df = diag
#         else:
#             summary_df = survivors_df.drop(columns=['model_obj','bets'], errors='ignore')
#
#         all_bets_csv_path = None
#         if save_all_bets_csv and ((_USE_VALUE) or (_IS_CLASSIFY and classify_odds_column is not None)) and all_bets_collector:
#             all_bets_df = pd.concat(all_bets_collector, ignore_index=True)
#             if not all_bets_include_failed:
#                 all_bets_df = all_bets_df[all_bets_df['passed_test_gate'] == True]
#             preferred = [c for c in [
#                 'date','league','country','home_team','away_team','match_id','event_name','selection',
#                 'market_odds','fair_odds','edge_ratio','stake','liability','commission_rate',
#                 'selection_occurred','target','pl','cum_pl',
#                 'stake_flat','liability_flat','pl_flat',
#                 'stake_liability','liability_liability','pl_liability',
#                 'candidate_id','passed_test_gate','mode','market','threshold',
#                 'odds_min','odds_max','edge_param','chosen_variant','lay_min_odds',
#                 'staking_plan_lay','staking_plan_back',
#                 'val_precision','val_precision_lcb','n_value_bets_test','total_pl','avg_pl','p_value','zscore','params_json'
#             ] if c in all_bets_df.columns]
#             all_bets_df = all_bets_df[preferred + [c for c in all_bets_df.columns if c not in preferred]]
#             all_bets_csv_path = os.path.join(all_bets_dir, f"all_bets_{market}_{timestamp}.csv")
#             _ensure_parent(all_bets_csv_path)
#             all_bets_df.to_csv(all_bets_csv_path, index=False)
#
#         if on_fail == "raise": raise RuntimeError("All Top-K failed the TEST gate.")
#         if on_fail == "warn": print("[WARN] All Top-K failed the TEST gate.")
#         return {'status':'failed_test_gate','csv':fail_csv,'model_pkl':None,
#                 'summary_df':summary_df,'validation_table':ranked,
#                 'bets_csv':None,'pl_plot':None,'all_bets_csv':all_bets_csv_path}
#
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
#     pkl_path = os.path.join(model_dir, f"best_model_{market}_{tag}_calibrated_{timestamp}.pkl")
#     _ensure_parent(pkl_path)
#
#     csv_df = passers_sorted.drop(columns=['model_obj','bets'], errors='ignore').copy()
#     csv_df['model_pkl'] = ""; csv_df.loc[0, 'model_pkl'] = pkl_path
#     csv_df['market'] = market
#     csv_path = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}.csv")
#     _ensure_parent(csv_path)
#     csv_df.to_csv(csv_path, index=False)
#
#     # Save top model (+ chosen min-odds if LAY)
#     top = passers_sorted.iloc[0]
#     if base_model == "xgb":
#         param_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']
#     else:
#         param_keys = ['hidden_layer_sizes','alpha','learning_rate_init','batch_size','max_iter']
#     chosen_params = {k: top[k] for k in param_keys if k in passers_sorted.columns}
#
#     joblib.dump(
#         {
#             'model': top['model_obj'],
#             'features': features,
#             'base_model': base_model,
#             'best_params': chosen_params,
#             'precision_test_threshold': float(precision_test_threshold),
#             'min_samples': int(min_samples),
#             'min_test_samples': int(min_test_samples),
#             'val_conf_level': float(val_conf_level),
#             'max_precision_drop': float(max_precision_drop),
#             'market': market,
#             'mode': top['mode'],
#             # VALUE bits
#             'edge_param': float(top.get('edge_param', np.nan)),
#             'staking_plan_lay': top.get('staking_plan_lay', None) if _IS_LAY else None,
#             'staking_plan_back': top.get('staking_plan_back', None) if _IS_BACK else None,
#             # CLASSIFY bits
#             'threshold': float(top.get('threshold', np.nan)) if _IS_CLASSIFY else np.nan,
#             'classify_side': classify_side if _IS_CLASSIFY else None,
#             'classify_odds_column': classify_odds_column if _IS_CLASSIFY else None,
#             'classify_lay_variant': top.get('classify_lay_variant', None) if (_IS_CLASSIFY and classify_side=='lay') else None,
#             # chosen min lay odds
#             'lay_min_odds': float(top.get('lay_min_odds', np.nan)) if (_IS_LAY or classify_side=='lay') else np.nan,
#             # commission
#             'commission_rate': float(commission_rate),
#             'notes': ('Commission applied only on winning returns; '
#                       'LAY flat-stake stakes S=1/(1-c) (win +1 net), LAY liability risks 1 (L=1, S=1/(o-1)); '
#                       'LAY min-odds swept and saved as lay_min_odds.'),
#             'run_seed': int(RUN_SEED),
#         },
#         pkl_path
#     )
#
#     # chosen bets CSV / plot
#     bets_path = None
#     plot_path = None
#     bets_df = passers_sorted.iloc[0].get('bets', None)
#     if (save_bets_csv or plot_pl) and isinstance(bets_df, pd.DataFrame) and len(bets_df):
#         if save_bets_csv:
#             bets_name = f"bets_{market}_{timestamp}.csv"
#             bets_path = os.path.join(bets_csv_dir, bets_name)
#             _ensure_parent(bets_path)
#             bets_df.to_csv(bets_path, index=False)
#         if plot_pl:
#             try:
#                 import matplotlib.pyplot as plt
#                 fig = plt.figure()
#                 x = bets_df['date'] if 'date' in bets_df.columns else np.arange(len(bets_df))
#                 y = bets_df['pl'].cumsum() if 'pl' in bets_df.columns else bets_df.iloc[:, -1]
#                 plt.plot(x, y)
#                 title = f"{market} cumulative P/L ({passers_sorted.iloc[0]['mode']})"
#                 if plot_title_suffix: title += f" — {plot_title_suffix}"
#                 plt.title(title)
#                 plt.xlabel('Date' if 'date' in bets_df.columns else 'Bet #')
#                 plt.ylabel('Cumulative P/L')
#                 plt.tight_layout()
#                 plot_name = f"cum_pl_{market}_{timestamp}.png"
#                 plot_path = os.path.join(plot_dir, plot_name)
#                 _ensure_parent(plot_path)
#                 plt.savefig(plot_path, dpi=160); plt.close(fig)
#             except Exception as e:
#                 print(f"[WARN] Failed to create plot: {e}")
#
#     # ALL bets export
#     all_bets_csv_path = None
#     if save_all_bets_csv and ((_USE_VALUE) or (_IS_CLASSIFY and classify_odds_column is not None)) and all_bets_collector:
#         all_bets_df = pd.concat(all_bets_collector, ignore_index=True)
#         if not all_bets_include_failed:
#             all_bets_df = all_bets_df[all_bets_df['passed_test_gate'] == True]
#         preferred = [c for c in [
#             'date','league','country','home_team','away_team','match_id','event_name','selection',
#             'market_odds','fair_odds','edge_ratio','stake','liability','commission_rate',
#             'selection_occurred','target','pl','cum_pl',
#             'stake_flat','liability_flat','pl_flat',
#             'stake_liability','liability_liability','pl_liability',
#             'candidate_id','passed_test_gate','mode','market','threshold',
#             'odds_min','odds_max','edge_param','chosen_variant','lay_min_odds',
#             'staking_plan_lay','staking_plan_back',
#             'val_precision','val_precision_lcb','n_value_bets_test','total_pl','avg_pl','p_value','zscore','params_json'
#         ] if c in all_bets_df.columns]
#         all_bets_df = all_bets_df[preferred + [c for c in all_bets_df.columns if c not in preferred]]
#         all_bets_csv_path = os.path.join(all_bets_dir, f"all_bets_{market}_{timestamp}.csv")
#         _ensure_parent(all_bets_csv_path)
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

def run_models_outcome_v2(
    matches_filtered,                       # pd.DataFrame
    features,                               # list[str]

    # ── gates ──────────────────────────────────────────────────────────────
    min_samples: int = 200,
    min_test_samples: int = 100,
    precision_test_threshold: float = 0.80,
    min_val_auc: float = 0.55,              # NEW: AUC gate on validation
    max_val_brier: float | None = None,     # optional Brier gate (if not None)

    # ── model/search ───────────────────────────────────────────────────────
    base_model: str = "xgb",                # "xgb" | "mlp"
    search_mode: str = "random",            # "random" | "grid"
    n_random_param_sets: int = 10,
    cpu_jobs: int = 6,
    top_k: int = 10,
    thresholds=None,                        # np.ndarray | None (NOT used in VALUE mode but kept for API symmetry)
    out_dir: str | None = None,

    # ── anti-overfitting ───────────────────────────────────────────────────
    val_conf_level: float = 0.99,
    max_precision_drop: float = 1.0,

    # ── failure handling ───────────────────────────────────────────────────
    on_fail: str = "return",                # "return" | "warn" | "raise"
    save_diagnostics_on_fail: bool = True,

    # ── market & VALUE controls ────────────────────────────────────────────
    market: str = "LAY_AWAY",               # LAY_* | BACK_*
    use_value_for_lay: bool = True,
    use_value_for_back: bool = True,
    value_edge_grid_lay=None,               # np.ndarray | None (edge >= 0): fair ≥ (1+edge)*market  (LAY)
    value_edge_grid_back=None,              # np.ndarray | None (edge >= 0): market ≥ (1+edge)*fair  (BACK)

    # ── LAY minimum odds (FIXED hyperparam in v2) ─────────────────────────
    lay_min_odds: float = 1.50,             # single min-odds used in both val & test (no grid search)

    # ── VALUE staking options (FIXED per run in v2) ───────────────────────
    staking_plan_lay: str = "liability",    # "liability" | "flat_stake" | "edge_prop" | "kelly_approx"
    staking_plan_back: str = "flat",        # "flat" | "edge_prop" | "kelly"

    # BACK staking knobs
    back_stake_test: float = 1.0,
    back_edge_scale: float = 0.10,
    kelly_fraction_back: float = 0.25,
    bankroll_back: float = 100.0,
    min_back_stake: float = 0.0,
    max_back_stake: float = 10.0,

    # ── COMMISSION (applied only to winning returns) ───────────────────────
    commission_rate: float = 0.02,

    # ── OUTPUTS ────────────────────────────────────────────────────────────
    save_bets_csv: bool = False,
    bets_csv_dir: str | None = None,
    plot_pl: bool = False,
    plot_dir: str | None = None,
    plot_title_suffix: str = "",
):
    """
    VALUE-only FT outcome pipeline with:
      • Temporal split into SEARCH (0–80%), CALIB (80–85%), TEST (85–100%).
      • Rolling CV on SEARCH region to tune:
           - model hyperparameters
           - VALUE edge parameter (lay/back)
      • No strategy hyperparameter tuning on the TEST set:
           - lay_min_odds and staking plans are fixed inputs for the whole run.
      • Calibration performed on a dedicated CALIB slice (not used for model selection).
      • Validation gate uses:
           - precision
           - AUC
           - (optional) Brier score
           - min bet counts.
      • Commission applied only to winning returns; losses not charged commission.
      • Returns dict with file paths and summary tables.
    """
    # ---------------- setup ----------------
    import os, secrets, hashlib, json
    from datetime import datetime
    import numpy as np
    import pandas as pd
    from itertools import product
    from sklearn.metrics import (
        precision_score, accuracy_score, roc_auc_score, log_loss, brier_score_loss
    )
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from joblib import Parallel, delayed, parallel_backend
    from tqdm import tqdm
    from tqdm_joblib import tqdm_joblib
    import joblib

    # ---------- path helpers ----------
    def _normpath(p: str | None) -> str | None:
        if p is None:
            return None
        return os.path.normpath(p)

    def _ensure_dir(path: str | None):
        if path is None:
            return
        os.makedirs(path, exist_ok=True)

    def _ensure_parent(path_to_file: str | None):
        if path_to_file is None:
            return
        parent = os.path.dirname(path_to_file)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ---------- optional XGBoost ----------
    try:
        import xgboost as xgb
        _HAS_XGB_LOCAL = True
    except Exception:
        _HAS_XGB_LOCAL = False
    _HAS_XGB = globals().get("_HAS_XGB", _HAS_XGB_LOCAL)
    if base_model == "xgb" and not _HAS_XGB:
        raise ImportError("XGBoost not available; set base_model='mlp' or install xgboost.")

    # ---------- SciPy distributions ----------
    try:
        from scipy.stats import randint as _randint
        from scipy.stats import uniform as _uniform
        from scipy.stats import loguniform as _loguniform
    except Exception as _e:
        raise ImportError("Please install SciPy: pip install scipy") from _e

    # ---------- maths helpers ----------
    def _Z(conf):
        try:
            from scipy.stats import norm
            return float(norm.ppf(1 - (1 - conf) / 2))
        except Exception:
            return 2.576 if conf >= 0.99 else 1.96

    def _Phi(z):
        try:
            from scipy.stats import norm
            return float(norm.cdf(z))
        except Exception:
            import maths
            return 0.5 * (1.0 + maths.erf(z / (2**0.5)))

    def _wilson_lcb(tp: int, fp: int, conf: float) -> float:
        n = tp + fp
        if n <= 0:
            return 0.0
        p = tp / n
        z = _Z(conf)
        denom = 1.0 + (z * z) / n
        centre = p + (z * z) / (2 * n)
        rad = z * np.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
        return max(0.0, (centre - rad) / denom)

    # ---------- defaults ----------
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)
    if value_edge_grid_lay is None:
        value_edge_grid_lay = np.round(np.arange(0.00, 0.201, 0.01), 2)
    if value_edge_grid_back is None:
        value_edge_grid_back = np.round(np.arange(0.00, 0.201, 0.01), 2)

    market = str(market).upper().strip()
    _IS_LAY  = market.startswith("LAY_")
    _IS_BACK = market.startswith("BACK_")
    _USE_VALUE_LAY  = bool(use_value_for_lay and _IS_LAY)
    _USE_VALUE_BACK = bool(use_value_for_back and _IS_BACK)
    _USE_VALUE = _USE_VALUE_LAY or _USE_VALUE_BACK

    if not _USE_VALUE:
        raise NotImplementedError(
            "run_models_outcome_v2 currently supports VALUE modes only "
            "(LAY_* / BACK_* with use_value_for_lay/use_value_for_back=True). "
            "Use your original run_models_outcome for pure CLASSIFY modes."
        )

    # ---------- dirs (reuse your existing layout for compatibility) ----------
    BASE = _normpath(r"C:\Users\leere\PycharmProjects\Football_ML3\FT Results")
    _ensure_dir(BASE)

    PKL_DIRS = {
        "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "model_file"),
        "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "model_file"),
        "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "model_file"),
        "BACK_HOME": os.path.join(BASE, "Back_Home", "model_file"),
        "BACK_AWAY": os.path.join(BASE, "Back_Away", "model_file"),
        "BACK_DRAW": os.path.join(BASE, "Back_Draw", "model_file"),
    }
    CSV_DIRS = {
        "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "best_model_metrics"),
        "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "best_model_metrics"),
        "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "best_model_metrics"),
        "BACK_HOME": os.path.join(BASE, "Back_Home", "best_model_metrics"),
        "BACK_AWAY": os.path.join(BASE, "Back_Away", "best_model_metrics"),
        "BACK_DRAW": os.path.join(BASE, "Back_Draw", "best_model_metrics"),
    }

    if market not in PKL_DIRS:
        raise ValueError(f"Unsupported market '{market}' for v2.")

    # normalise dirs & ensure they exist
    for d in PKL_DIRS.values():
        _ensure_dir(_normpath(d))
    for d in CSV_DIRS.values():
        _ensure_dir(_normpath(d))

    csv_save_dir = _normpath(out_dir) if (out_dir and len(str(out_dir)) > 0) else _normpath(CSV_DIRS[market])
    _ensure_dir(csv_save_dir)

    model_dir = _normpath(PKL_DIRS[market])
    _ensure_dir(model_dir)

    if bets_csv_dir is None:
        bets_csv_dir = csv_save_dir
    if plot_dir is None:
        plot_dir = csv_save_dir
    bets_csv_dir = _normpath(bets_csv_dir); _ensure_dir(bets_csv_dir)
    plot_dir = _normpath(plot_dir); _ensure_dir(plot_dir)

    RUN_SEED = secrets.randbits(32)

    def _seed_from(*vals) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(int(RUN_SEED).to_bytes(8, "little", signed=False))
        for v in vals:
            h.update(str(v).encode("utf-8"))
        return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF

    # ---------------- data ----------------
    df = matches_filtered.copy()
    req_cols = {"date", "target", "home_odds", "draw_odds", "away_odds"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    cols_needed = list(set(features) | {"target", "home_odds", "draw_odds", "away_odds"})
    df = df.dropna(subset=cols_needed).reset_index(drop=True)

    X = df[features].copy()
    y = df["target"].astype(int).reset_index(drop=True)

    n = len(X)
    if n < max(min_samples * 3, 500):
        raise RuntimeError(f"Not enough rows for v2: {n}")

    # ---------------- temporal split: SEARCH / CALIB / TEST -----------------
    # 0–80%: SEARCH (train + rolling val)
    # 80–85%: CALIB (calibration only, no hyperparameter search)
    # 85–100%: TEST (single final evaluation)
    search_end = int(0.80 * n)
    calib_end = int(0.85 * n)
    test_start = calib_end

    if search_end <= 0 or test_start <= search_end:
        raise RuntimeError("Data too short for SEARCH/CALIB/TEST split.")

    X_search = X.iloc[:search_end].reset_index(drop=True)
    y_search = y.iloc[:search_end].reset_index(drop=True)
    df_search = df.iloc[:search_end].reset_index(drop=True)

    X_calib = X.iloc[search_end:calib_end].reset_index(drop=True)
    y_calib = y.iloc[search_end:calib_end].reset_index(drop=True)

    X_test = X.iloc[test_start:].reset_index(drop=True)
    y_test = y.iloc[test_start:].reset_index(drop=True)
    df_test = df.iloc[test_start:].reset_index(drop=True)

    # rolling validation folds inside SEARCH region
    N_FOLDS = 5
    pretest_end = len(X_search)
    total_val_len = max(N_FOLDS, int(0.25 * pretest_end))
    total_val_len = min(total_val_len, pretest_end)
    val_len = max(1, total_val_len // N_FOLDS)

    fold_val_ends = [pretest_end - total_val_len + (i + 1) * val_len for i in range(N_FOLDS)]
    fold_val_starts = [end - val_len for end in fold_val_ends]
    if fold_val_ends:
        fold_val_ends[-1] = min(fold_val_ends[-1], pretest_end)
        fold_val_starts[-1] = max(0, fold_val_ends[-1] - val_len)

    # ---------------- model param spaces ----------------
    xgb_param_grid = {
        "n_estimators": [200],
        "max_depth": [5],
        "learning_rate": [0.1],
        "subsample": [0.7],
        "colsample_bytree": [1.0],
        "min_child_weight": [5],
        "reg_lambda": [1.0],
    }
    xgb_param_distributions = {
        "n_estimators":     _randint(100, 1001),
        "max_depth":        _randint(3, 8),
        "learning_rate":    _loguniform(0.01, 0.2),
        "min_child_weight": _randint(3, 13),
        "subsample":        _uniform(0.7, 0.3),
        "colsample_bytree": _uniform(0.6, 0.4),
        "reg_lambda":       _loguniform(0.1, 10.0),
    }
    mlp_param_grid = {
        "hidden_layer_sizes": [(128,), (256,), (128, 64)],
        "alpha": [1e-4],
        "learning_rate_init": [1e-3],
        "batch_size": ["auto"],
        "max_iter": [200],
    }
    mlp_param_distributions = {
        "hidden_layer_sizes": [(64,), (128,), (256,), (128, 64), (256, 128)],
        "alpha":              _loguniform(1e-5, 1e-2),
        "learning_rate_init": _loguniform(5e-4, 5e-2),
        "batch_size":         _randint(32, 257),
        "max_iter":           _randint(150, 401),
    }

    def _cast_params(p: dict) -> dict:
        q = dict(p)
        if base_model == "xgb":
            for k in ["n_estimators", "max_depth", "min_child_weight"]:
                if k in q:
                    q[k] = int(round(float(q[k])))
            for k in ["learning_rate", "subsample", "colsample_bytree", "reg_lambda"]:
                if k in q:
                    q[k] = float(q[k])
        else:
            if "max_iter" in q:
                q["max_iter"] = int(round(float(q["max_iter"])))
            if "batch_size" in q and q["batch_size"] != "auto":
                q["batch_size"] = int(round(float(q["batch_size"])))
            for k in ["alpha", "learning_rate_init"]:
                if k in q:
                    q[k] = float(q[k])
            if "hidden_layer_sizes" in q and isinstance(q["hidden_layer_sizes"], (list, tuple)):
                q["hidden_layer_sizes"] = tuple(int(v) for v in q["hidden_layer_sizes"])
        return q

    # ---------- safe random sampler ----------
    def _safe_random_param_sets(dists: dict, n_iter: int, seed: int) -> list[dict]:
        import numpy as _np
        rng = _np.random.RandomState(seed)
        keys = list(dists.keys())
        out = []
        for _ in range(int(n_iter)):
            params = {}
            for k in keys:
                v = dists[k]
                if hasattr(v, "rvs"):
                    params[k] = v.rvs(random_state=rng)
                elif isinstance(v, (list, tuple)):
                    if len(v) == 0:
                        raise ValueError(f"Empty choices for '{k}'")
                    params[k] = v[rng.randint(0, len(v))]
                else:
                    params[k] = v
            out.append(params)
        return out

    # ---------------- model builders ----------------
    def _final_step_name(estimator):
        try:
            if isinstance(estimator, Pipeline):
                return estimator.steps[-1][0]
        except Exception:
            pass
        return None

    def _build_model(params: dict, spw: float):
        seed = _seed_from("model", base_model, *sorted(map(str, params.items())))
        if base_model == "xgb":
            return xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=seed,
                scale_pos_weight=spw,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
                **params,
            )
        else:
            mlp = MLPClassifier(
                random_state=seed,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
                solver="adam",
                **params,
            )
            return make_pipeline(StandardScaler(with_mean=True, with_std=True), mlp)

    def _fit_model(model, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None):
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

    def _fit_calibrator(fitted, X_va, y_va):
        try:
            from sklearn.calibration import FrozenEstimator
            frozen = FrozenEstimator(fitted)
            cal = CalibratedClassifierCV(frozen, method="sigmoid", cv=None)
            cal.fit(X_va, y_va)
            return cal
        except Exception:
            try:
                cal = CalibratedClassifierCV(fitted, method="sigmoid", cv="prefit")
                cal.fit(X_va, y_va)
                return cal
            except Exception:
                return fitted

    def _unwrap_estimator(est):
        if isinstance(est, Pipeline):
            return est.steps[-1][1]
        return est

    def _predict_proba_pos(model_or_cal, X_):
        proba = model_or_cal.predict_proba(X_)
        if proba.ndim == 2:
            classes = getattr(model_or_cal, "classes_", None)
            if classes is None:
                base = _unwrap_estimator(model_or_cal)
                classes = getattr(base, "classes_", None)
            if classes is not None and len(classes) == proba.shape[1]:
                import numpy as _np
                try:
                    idx = int(_np.where(_np.asarray(classes) == 1)[0][0])
                    return proba[:, idx].astype(np.float32)
                except Exception:
                    pass
            if proba.shape[1] == 2:
                return proba[:, 1].astype(np.float32)
            if proba.shape[1] == 1:
                only = getattr(model_or_cal, "classes_", [0])[0]
                return (np.ones_like(proba[:, 0]) if only == 1 else np.zeros_like(proba[:, 0])).astype(np.float32)
        return np.asarray(proba, dtype=np.float32)

    # --- p-value helper (commission on wins only) ---
    def _pvalue_break_even(bdf: pd.DataFrame, mode: str) -> dict:
        if not isinstance(bdf, pd.DataFrame) or bdf.empty:
            return {"z": 0.0, "p_value": 1.0, "var_sum": 0.0, "n": 0, "total_pl": 0.0}
        o = np.asarray(bdf["market_odds"].values, dtype=float)
        o = np.where(o <= 1.0, np.nan, o)
        p_null_win = 1.0 / o
        if mode == "VALUE_BACK":
            S = np.asarray(bdf["stake"].values, dtype=float)
            win = (o - 1.0) * S * (1.0 - commission_rate)
            lose = -S
        else:  # VALUE_LAY
            L = np.asarray(bdf.get("liability", np.nan * np.ones_like(o))).astype(float)
            S = np.asarray(bdf["stake"].values, dtype=float)
            win = S * (1.0 - commission_rate)    # selection loses
            lose = -L                             # selection wins
        var_i = p_null_win * (win ** 2) + (1.0 - p_null_win) * (lose ** 2)
        var_i = np.where(np.isfinite(var_i), var_i, 0.0)
        pl = np.asarray(bdf["pl"].values, dtype=float)
        total_pl = float(np.nansum(np.where(np.isfinite(pl), pl, 0.0)))
        var_sum = float(np.nansum(var_i))
        z = total_pl / (np.sqrt(var_sum) + 1e-12)
        p_val = max(0.0, 1.0 - _Phi(z))  # one-sided
        return {"z": float(z), "p_value": float(p_val), "var_sum": var_sum, "n": int(len(pl)), "total_pl": total_pl}

    # ---------------- VALUE staking calculators ----------------------------
    def _lay_stakes_value(o: np.ndarray, edge_ratio_minus1: np.ndarray, plan: str):
        # LAY (NO clipping; always place the bet once selected):
        #   - liability:  L = 1, S = 1/(o-1)
        #   - flat_stake: S = 1/(1-c), L = S*(o-1)
        #   - edge_prop/kelly_approx: simple edge-based scaling of liability.
        o = np.asarray(o, dtype=float)
        edge_plus = np.maximum(np.asarray(edge_ratio_minus1, dtype=float), 0.0)
        denom = np.maximum(o - 1.0, 1e-12)

        if plan == "liability":
            L = np.ones_like(o, dtype=float)
            S = L / denom
            ok = np.ones_like(S, dtype=bool)
            return S, L, ok

        if plan == "flat_stake":
            S = np.full_like(o, 1.0 / max(1e-12, 1.0 - commission_rate), dtype=float)
            L = S * denom
            ok = np.ones_like(S, dtype=bool)
            return S, L, ok

        if plan == "edge_prop":
            scale = max(1e-12, float(back_edge_scale))
            L = edge_plus / scale
            S = L / denom
            ok = np.ones_like(S, dtype=bool)
            return S, L, ok

        if plan == "kelly_approx":
            L = 1.0 * edge_plus
            S = L / denom
            ok = np.ones_like(S, dtype=bool)
            return S, L, ok

        raise ValueError(f"Unknown staking_plan_lay: {plan}")

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

    # ---------------- search grid / samples --------------------------------
    if search_mode.lower() == "grid":
        grid = xgb_param_grid if base_model == "xgb" else mlp_param_grid
        keys = list(grid.keys()); vals = [grid[k] for k in keys]
        all_param_dicts = [dict(zip(keys, combo)) for combo in product(*vals)]
    else:
        dists = xgb_param_distributions if base_model == "xgb" else mlp_param_distributions
        sampler_seed = _seed_from("sampler")
        all_param_dicts = _safe_random_param_sets(dists, n_random_param_sets, sampler_seed)

    # ---------------- validation evaluation --------------------------------
    def _evaluate_param_set(param_dict):
        safe = _cast_params(param_dict)
        rows = []
        val_prob_all = []
        val_true_all = []

        for vstart, vend in zip(fold_val_starts, fold_val_ends):
            if vstart is None or vend is None or vstart <= 0 or vend <= vstart:
                continue
            X_tr, y_tr = X_search.iloc[:vstart], y_search.iloc[:vstart]
            X_va, y_va = X_search.iloc[vstart:vend], y_search.iloc[vstart:vend]
            df_va = df_search.iloc[vstart:vend]

            if y_tr.nunique() < 2:
                continue

            pos = int(y_tr.sum()); neg = len(y_tr) - pos
            spw = (neg / pos) if pos > 0 else 1.0

            sample_weight = None
            if base_model == "mlp":
                w_pos = spw
                sample_weight = np.where(y_tr.values == 1, w_pos, 1.0).astype(np.float32)

            model = _build_model(safe, spw)
            _fit_model(model, X_tr, y_tr, X_va, y_va, sample_weight=sample_weight)
            cal = _fit_calibrator(model, X_va, y_va)

            p_pos = _predict_proba_pos(cal, X_va)
            val_prob_all.append(p_pos)
            y_true = y_va.values.astype(np.uint8)
            val_true_all.append(y_true)

            # VALUE validation (edge rule only; lay_min_odds & staking fixed per run)
            if _IS_LAY:
                if market == "LAY_AWAY":
                    mkt = df_va["away_odds"].values
                elif market == "LAY_HOME":
                    mkt = df_va["home_odds"].values
                else:
                    mkt = df_va["draw_odds"].values
                mkt = mkt.astype(float)
                fair = 1.0 / np.clip(1.0 - p_pos, 1e-9, 1.0)
                valid = np.isfinite(mkt) & (mkt > 1.01) & (mkt >= float(lay_min_odds))

                for edge_param in value_edge_grid_lay:
                    edge_param = float(edge_param)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        edge_ratio = np.where(mkt > 0, fair / mkt, np.nan)
                    edge_mask = valid & (fair >= (1.0 + edge_param) * mkt)
                    y_pred = edge_mask.astype(np.uint8)

                    rows.append({
                        **safe,
                        "edge_param": edge_param,
                        "fold_vstart": int(vstart),
                        "fold_vend": int(vend),
                        "n_preds_val": int(y_pred.sum()),
                        "tp_val": int(((y_true == 1) & (y_pred == 1)).sum()),
                        "fp_val": int(((y_true == 0) & (y_pred == 1)).sum()),
                        "val_precision": float(precision_score(y_true, y_pred, zero_division=0)),
                        "val_accuracy": float(accuracy_score(y_true, y_pred)),
                        "n_value_bets_val": int(y_pred.sum()),
                        "val_edge_ratio_mean": float(
                            np.nanmean(np.where(np.isfinite(edge_ratio), edge_ratio, np.nan))
                        ),
                    })
            else:
                # BACK VALUE
                if market == "BACK_AWAY":
                    mkt = df_va["away_odds"].values
                elif market == "BACK_HOME":
                    mkt = df_va["home_odds"].values
                else:
                    mkt = df_va["draw_odds"].values
                mkt = mkt.astype(float)
                fair = 1.0 / np.clip(p_pos, 1e-9, 1.0)
                valid = np.isfinite(mkt) & (mkt > 1.01)

                with np.errstate(divide="ignore", invalid="ignore"):
                    edge_ratio_back = np.where(fair > 0, mkt / fair, np.nan)

                for edge_param in value_edge_grid_back:
                    edge_param = float(edge_param)
                    edge_mask = valid & (mkt >= (1.0 + edge_param) * fair)
                    y_pred = edge_mask.astype(np.uint8)

                    rows.append({
                        **safe,
                        "edge_param": edge_param,
                        "fold_vstart": int(vstart),
                        "fold_vend": int(vend),
                        "n_preds_val": int(y_pred.sum()),
                        "tp_val": int(((y_true == 1) & (y_pred == 1)).sum()),
                        "fp_val": int(((y_true == 0) & (y_pred == 1)).sum()),
                        "val_precision": float(precision_score(y_true, y_pred, zero_division=0)),
                        "val_accuracy": float(accuracy_score(y_true, y_pred)),
                        "n_value_bets_val": int(y_pred.sum()),
                        "val_edge_ratio_mean_back": float(
                            np.nanmean(np.where(np.isfinite(edge_ratio_back), edge_ratio_back, np.nan))
                        ),
                    })

        # pooled diagnostics
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
            r["val_auc"] = val_auc
            r["val_logloss"] = val_ll
            r["val_brier"] = val_bri

        return rows

    # ---------------- run search -------------------------------------------
    if base_model == "mlp":
        eff_jobs = min(max(1, cpu_jobs), 4); prefer = "threads"; backend = "threading"; pre_dispatch = eff_jobs
        ctx = parallel_backend(backend, n_jobs=eff_jobs)
    else:
        eff_jobs = max(1, min(cpu_jobs, 4)) if cpu_jobs != -1 else 4
        prefer = "processes"; backend = "loky"; pre_dispatch = f"{2 * eff_jobs}"
        ctx = parallel_backend(backend, n_jobs=eff_jobs, inner_max_num_threads=1)

    with ctx:
        try:
            with tqdm_joblib(
                tqdm(total=len(all_param_dicts), desc=f"Param search ({search_mode}, {base_model})")
            ) as _:
                out = Parallel(
                    n_jobs=eff_jobs, batch_size=1, prefer=prefer, pre_dispatch=pre_dispatch
                )(
                    delayed(_evaluate_param_set)(_cast_params(pd_)) for pd_ in all_param_dicts
                )
        except OSError as e:
            print(f"[WARN] Parallel failed with {e}. Falling back to serial search...")
            out = []
            for pd_ in tqdm(all_param_dicts, desc=f"Param search (serial, {base_model})"):
                out.append(_evaluate_param_set(pd_))

    val_rows = [r for sub in out for r in sub]
    if not val_rows:
        raise RuntimeError("No validation rows produced (check folds and input data).")
    val_df = pd.DataFrame(val_rows)

    # ---------------- validation aggregate ---------------------------------
    if base_model == "xgb":
        param_keys = ["n_estimators", "max_depth", "learning_rate", "subsample",
                      "colsample_bytree", "min_child_weight", "reg_lambda"]
    else:
        param_keys = ["hidden_layer_sizes", "alpha", "learning_rate_init", "batch_size", "max_iter"]

    group_cols = param_keys + ["edge_param"]

    agg_dict = {
        "n_preds_val": "sum",
        "tp_val": "sum",
        "fp_val": "sum",
        "val_precision": "mean",
        "val_accuracy": "mean",
        "val_auc": "mean",
        "val_logloss": "mean",
        "val_brier": "mean",
        "n_value_bets_val": "sum",
    }
    if "val_edge_ratio_mean" in val_df.columns:
        agg_dict["val_edge_ratio_mean"] = "mean"
    if "val_edge_ratio_mean_back" in val_df.columns:
        agg_dict["val_edge_ratio_mean_back"] = "mean"

    agg = val_df.groupby(group_cols, as_index=False).agg(agg_dict)

    def _scalar(x):
        # If duplicate column names exist, r["tp_val"] can be a Series.
        # Convert that safely into a single scalar.
        if hasattr(x, "iloc"):
            return x.iloc[0]
        return x

    agg["val_precision_pooled"] = agg.apply(
        lambda r: float(_scalar(r["tp_val"])) / max(1.0, float(_scalar(r["tp_val"])) + float(_scalar(r["fp_val"]))),
        axis=1
    )

    agg["val_precision_lcb"] = agg.apply(
        lambda r: _wilson_lcb(int(_scalar(r["tp_val"])), int(_scalar(r["fp_val"])), conf=val_conf_level),
        axis=1
    )

    # multi-metric quality mask
    qual_mask = (
        (agg["val_precision"] >= float(precision_test_threshold)) &
        (agg["n_value_bets_val"] >= int(min_samples))
    )

    if not np.isnan(agg["val_auc"]).all():
        qual_mask &= (agg["val_auc"] >= float(min_val_auc))

    if max_val_brier is not None and not np.isnan(agg["val_brier"]).all():
        qual_mask &= (agg["val_brier"] <= float(max_val_brier))

    qual = agg[qual_mask].copy()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if qual.empty:
        fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv") \
            if save_diagnostics_on_fail else None
        if save_diagnostics_on_fail:
            diag = (
                agg.sort_values(
                    ["val_precision_lcb", "val_precision", "n_value_bets_val", "val_accuracy"],
                    ascending=[False, False, False, False]
                ).assign(fail_reason="failed_validation_gate", market=market)
            )
            _ensure_parent(fail_csv)
            diag.to_csv(fail_csv, index=False)
        if on_fail == "raise":
            raise RuntimeError("No strategy met validation gates.")
        if on_fail == "warn":
            print("[WARN] No strategy met the validation gate.")
        return {
            "status": "failed_validation_gate",
            "csv": fail_csv,
            "model_pkl": None,
            "summary_df": None,
            "validation_table": agg.sort_values(
                ["val_precision_lcb", "val_precision", "n_value_bets_val", "val_accuracy"],
                ascending=[False, False, False, False]
            ).reset_index(drop=True),
            "bets_csv": None,
            "pl_plot": None,
        }

    ranked = qual.sort_values(
        by=["val_precision_lcb", "val_precision", "n_value_bets_val", "val_accuracy"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)
    top_k = int(top_k)
    topk_val = ranked.head(top_k).reset_index(drop=True)

    def _extract_params_from_row(row):
        return _cast_params({k: row[k] for k in param_keys if k in row.index})

    candidates = []
    for _, row in topk_val.iterrows():
        c = {
            "params": _extract_params_from_row(row),
            "edge_param": float(row["edge_param"]),
            "val_precision": float(row["val_precision"]),
            "val_precision_lcb": float(row["val_precision_lcb"]),
            "val_accuracy": float(row["val_accuracy"]),
            "val_auc": float(row["val_auc"]),
            "val_brier": float(row["val_brier"]),
            "n_value_bets_val": int(row["n_value_bets_val"]),
        }
        candidates.append(c)

    # ---------------- TEST EVAL (no strategy search on test) ----------------
    records_all = []

    def _name_cols(subdf):
        cols = {}
        for c in ["date", "league", "country", "home_team", "away_team", "match_id"]:
            if c in subdf.columns:
                cols[c] = subdf[c].values
        if {"home_team", "away_team"}.issubset(subdf.columns):
            cols["event_name"] = (subdf["home_team"] + " v " + subdf["away_team"]).values
        return cols

    for cand in candidates:
        best_params = _cast_params(cand["params"])
        edge_param = float(cand["edge_param"])

        # Training data for final fit = whole SEARCH region
        X_train_final = X_search
        y_train_final = y_search

        pos = int(y_train_final.sum()); neg = len(y_train_final) - pos
        spw_final = (neg / pos) if pos > 0 else 1.0

        final_model = _build_model(best_params, spw_final)
        final_sample_weight = None
        if base_model == "mlp":
            w_pos = spw_final
            final_sample_weight = np.where(y_train_final.values == 1, w_pos, 1.0).astype(np.float32)

        # Fit on SEARCH; calibrate on CALIB slice only (no reuse of validation folds)
        _fit_model(final_model, X_train_final, y_train_final, X_calib, y_calib, sample_weight=final_sample_weight)
        final_calibrator = _fit_calibrator(final_model, X_calib, y_calib)

        p_pos_test = _predict_proba_pos(final_calibrator, X_test)

        if _IS_LAY:
            # ===== VALUE_LAY TEST =====
            if market == "LAY_AWAY":
                mkt_odds = df_test["away_odds"].values.astype(float)
                sel_name = "AWAY"
            elif market == "LAY_HOME":
                mkt_odds = df_test["home_odds"].values.astype(float)
                sel_name = "HOME"
            else:
                mkt_odds = df_test["draw_odds"].values.astype(float)
                sel_name = "DRAW"

            fair_odds = 1.0 / np.clip(1.0 - p_pos_test, 1e-9, 1.0)
            valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01) & (mkt_odds >= float(lay_min_odds))

            with np.errstate(divide="ignore", invalid="ignore"):
                edge_ratio_minus1 = np.where(mkt_odds > 0, fair_odds / mkt_odds - 1.0, 0.0)

            base_value = valid & (fair_odds >= (1.0 + edge_param) * mkt_odds)

            S_plan, L_plan, _ = _lay_stakes_value(mkt_odds, edge_ratio_minus1, staking_plan_lay)
            S = np.where(base_value, S_plan, 0.0)
            L = np.where(base_value, L_plan, 0.0)

            sel_occurs = (y_test.values == 0)  # selection occurs => lay loses
            pl = np.where(~base_value, 0.0,
                          np.where(sel_occurs, -L, S * (1.0 - commission_rate)))

            n_bets = int(np.count_nonzero(S > 0))
            y_pred = (S > 0).astype(np.uint8)
            prc_test = precision_score(y_test, y_pred, zero_division=0)
            acc_test = accuracy_score(y_test, y_pred)
            total_pl = float(pl.sum())
            avg_pl = float(total_pl / max(1, n_bets))

            if n_bets > 0:
                bet_idx = np.where(S > 0)[0]
                name_cols = _name_cols(df_test.iloc[bet_idx])
                bets_df = pd.DataFrame({
                    **name_cols,
                    "selection": sel_name,
                    "market_odds": mkt_odds[bet_idx],
                    "fair_odds": fair_odds[bet_idx],
                    "edge_ratio": np.where(mkt_odds[bet_idx] > 0, fair_odds[bet_idx] / mkt_odds[bet_idx], np.nan),
                    "liability": L[bet_idx],
                    "stake": S[bet_idx],
                    "commission_rate": float(commission_rate),
                    "selection_occurred": sel_occurs[bet_idx].astype(int),
                    "target": y_test.values[bet_idx],
                    "pl": pl[bet_idx],
                })
                if "date" in bets_df.columns:
                    bets_df = bets_df.sort_values("date").reset_index(drop=True)
                bets_df["cum_pl"] = bets_df["pl"].cumsum()
                pv = _pvalue_break_even(bets_df, mode="VALUE_LAY")
            else:
                bets_df = None
                pv = {"p_value": np.nan, "z": np.nan}

            pass_gate = (
                (n_bets >= int(min_test_samples)) and
                (prc_test >= max(float(precision_test_threshold),
                                 float(cand["val_precision"]) - float(max_precision_drop)))
            )

            records_all.append({
                **best_params,
                "mode": "VALUE_LAY",
                "edge_param": edge_param,
                "lay_min_odds": float(lay_min_odds),
                "staking_plan_lay": staking_plan_lay,
                "val_precision_lcb": cand["val_precision_lcb"],
                "val_precision": cand["val_precision"],
                "val_accuracy": cand["val_accuracy"],
                "val_auc": cand["val_auc"],
                "val_brier": cand["val_brier"],
                "n_value_bets_test": int(n_bets),
                "test_precision_bets": float(prc_test),
                "test_accuracy_bets": float(acc_test),
                "total_pl": float(total_pl),
                "avg_pl": float(avg_pl),
                "p_value": float(pv["p_value"]) if pd.notna(pv["p_value"]) else np.nan,
                "zscore": float(pv["z"]) if pd.notna(pv["z"]) else np.nan,
                "pass_test_gate": bool(pass_gate),
                "fail_reason": "" if pass_gate else "insufficient_or_collapse",
                "model_obj": final_calibrator if pass_gate else None,
                "bets": bets_df if pass_gate else None,
                "commission_rate": float(commission_rate),
            })

        else:
            # ===== VALUE_BACK TEST =====
            if market == "BACK_AWAY":
                mkt_odds = df_test["away_odds"].values.astype(float)
                sel_name = "AWAY"
            elif market == "BACK_HOME":
                mkt_odds = df_test["home_odds"].values.astype(float)
                sel_name = "HOME"
            else:
                mkt_odds = df_test["draw_odds"].values.astype(float)
                sel_name = "DRAW"

            p_sel_win = p_pos_test
            fair_odds = 1.0 / np.clip(p_sel_win, 1e-9, 1.0)
            valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01)

            with np.errstate(divide="ignore", invalid="ignore"):
                edge_ratio_minus1 = np.where(fair_odds > 0, mkt_odds / fair_odds - 1.0, 0.0)

            value_mask = valid & (mkt_odds >= (1.0 + edge_param) * fair_odds)

            stake = np.zeros_like(mkt_odds, dtype=float)
            s = _back_stakes(mkt_odds, edge_ratio_minus1, staking_plan_back, p_sel_win)
            stake[value_mask] = s[value_mask]

            sel_occurs = (y_test.values == 1)
            pl = np.where(
                (stake > 0) & sel_occurs,
                (mkt_odds - 1.0) * stake * (1.0 - commission_rate),
                0.0
            )
            pl = np.where(
                (stake > 0) & (~sel_occurs),
                -stake,
                pl
            )

            n_bets = int(np.count_nonzero(stake > 0))
            total_pl = float(pl.sum())
            avg_pl = float(total_pl / max(1, n_bets))
            y_pred = (stake > 0).astype(np.uint8)
            prc_test = precision_score(y_test, y_pred, zero_division=0)
            acc_test = accuracy_score(y_test, y_pred)

            bet_idx = np.where(stake > 0)[0]
            if len(bet_idx):
                name_cols = _name_cols(df_test.iloc[bet_idx])
                bets_df = pd.DataFrame({
                    **name_cols,
                    "selection": sel_name,
                    "market_odds": mkt_odds[bet_idx],
                    "fair_odds": fair_odds[bet_idx],
                    "edge_ratio": np.where(fair_odds[bet_idx] > 0, mkt_odds[bet_idx] / fair_odds[bet_idx], np.nan),
                    "stake": stake[bet_idx],
                    "commission_rate": float(commission_rate),
                    "selection_occurred": sel_occurs[bet_idx].astype(int),
                    "target": y_test.values[bet_idx],
                    "pl": pl[bet_idx],
                }).sort_values("date" if "date" in df_test.columns else "pl")
                bets_df["cum_pl"] = bets_df["pl"].cumsum()
                pv = _pvalue_break_even(bets_df[["market_odds", "stake", "pl"]], mode="VALUE_BACK")
            else:
                bets_df = None
                pv = {"p_value": np.nan, "z": np.nan}

            pass_gate = (
                (n_bets >= int(min_test_samples)) and
                (prc_test >= max(float(precision_test_threshold),
                                 float(cand["val_precision"]) - float(max_precision_drop)))
            )

            records_all.append({
                **best_params,
                "mode": "VALUE_BACK",
                "edge_param": edge_param,
                "staking_plan_back": staking_plan_back,
                "val_precision_lcb": cand["val_precision_lcb"],
                "val_precision": cand["val_precision"],
                "val_accuracy": cand["val_accuracy"],
                "val_auc": cand["val_auc"],
                "val_brier": cand["val_brier"],
                "n_value_bets_test": int(n_bets),
                "test_precision_bets": float(prc_test),
                "test_accuracy_bets": float(acc_test),
                "total_pl": float(total_pl),
                "avg_pl": float(avg_pl),
                "p_value": float(pv["p_value"]) if pd.notna(pv["p_value"]) else np.nan,
                "zscore": float(pv["z"]) if pd.notna(pv["z"]) else np.nan,
                "pass_test_gate": bool(pass_gate),
                "fail_reason": "" if pass_gate else "insufficient_or_collapse",
                "model_obj": final_calibrator if pass_gate else None,
                "bets": bets_df if pass_gate else None,
                "commission_rate": float(commission_rate),
            })

    survivors_df = pd.DataFrame(records_all)
    passers = survivors_df[survivors_df["pass_test_gate"]].copy()

    # ---------------- save / rank ------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "xgb" if base_model == "xgb" else "mlp"

    if passers.empty:
        fail_csv = None
        if save_diagnostics_on_fail:
            if "p_value" in survivors_df.columns and survivors_df["p_value"].notna().any():
                sort_cols = ["p_value", "total_pl", "val_precision_lcb"]
                asc = [True, False, False]
            else:
                sort_cols = ["val_precision_lcb", "val_precision", "n_value_bets_test", "val_accuracy"]
                asc = [False, False, False, False]
            diag = (
                survivors_df.drop(columns=["model_obj", "bets"], errors="ignore")
                .sort_values(by=sort_cols, ascending=asc)
                .assign(market=market)
            )
            fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv")
            _ensure_parent(fail_csv)
            diag.to_csv(fail_csv, index=False)
            summary_df = diag
        else:
            summary_df = survivors_df.drop(columns=["model_obj", "bets"], errors="ignore")

        if on_fail == "raise":
            raise RuntimeError("All Top-K failed the TEST gate.")
        if on_fail == "warn":
            print("[WARN] All Top-K failed the TEST gate.")
        return {
            "status": "failed_test_gate",
            "csv": fail_csv,
            "model_pkl": None,
            "summary_df": summary_df,
            "validation_table": ranked,
            "bets_csv": None,
            "pl_plot": None,
        }

    if ("p_value" in passers.columns) and passers["p_value"].notna().any():
        passers_sorted = passers.sort_values(
            by=["p_value", "total_pl", "avg_pl", "val_precision_lcb", "val_precision"],
            ascending=[True, False, False, False, False]
        ).reset_index(drop=True)
    else:
        passers_sorted = passers.sort_values(
            by=["val_precision_lcb", "val_precision", "n_value_bets_test", "val_accuracy"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)

    pkl_path = os.path.join(model_dir, f"best_model_{market}_{tag}_calibrated_{timestamp}.pkl")
    _ensure_parent(pkl_path)

    csv_df = passers_sorted.drop(columns=["model_obj", "bets"], errors="ignore").copy()
    csv_df["model_pkl"] = ""
    csv_df.loc[0, "model_pkl"] = pkl_path
    csv_df["market"] = market
    csv_path = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}.csv")
    _ensure_parent(csv_path)
    csv_df.to_csv(csv_path, index=False)

    # Save top model
    top = passers_sorted.iloc[0]
    chosen_params = {k: top[k] for k in param_keys if k in passers_sorted.columns}

    joblib.dump(
        {
            "model": top["model_obj"],
            "features": features,
            "base_model": base_model,
            "best_params": chosen_params,
            "precision_test_threshold": float(precision_test_threshold),
            "min_samples": int(min_samples),
            "min_test_samples": int(min_test_samples),
            "val_conf_level": float(val_conf_level),
            "max_precision_drop": float(max_precision_drop),
            "market": market,
            "mode": top["mode"],
            # VALUE bits
            "edge_param": float(top.get("edge_param", np.nan)),
            "staking_plan_lay": top.get("staking_plan_lay", None) if _IS_LAY else None,
            "staking_plan_back": top.get("staking_plan_back", None) if _IS_BACK else None,
            # chosen lay min odds
            "lay_min_odds": float(top.get("lay_min_odds", np.nan)) if _IS_LAY else np.nan,
            # commission
            "commission_rate": float(commission_rate),
            "notes": (
                "v2: SEARCH/CALIB/TEST split (0–80/80–85/85–100). "
                "VALUE-only; no strategy tuning on TEST. "
                "Commission applied only on winning returns."
            ),
            "run_seed": int(RUN_SEED),
        },
        pkl_path,
    )

    # chosen bets CSV / plot
    bets_path = None
    plot_path = None
    bets_df = passers_sorted.iloc[0].get("bets", None)
    if (save_bets_csv or plot_pl) and isinstance(bets_df, pd.DataFrame) and len(bets_df):
        if save_bets_csv:
            bets_name = f"bets_{market}_{timestamp}.csv"
            bets_path = os.path.join(bets_csv_dir, bets_name)
            _ensure_parent(bets_path)
            bets_df.to_csv(bets_path, index=False)
        if plot_pl:
            try:
                import matplotlib.pyplot as plt
                fig = plt.figure()
                x = bets_df["date"] if "date" in bets_df.columns else np.arange(len(bets_df))
                y = bets_df["pl"].cumsum() if "pl" in bets_df.columns else bets_df.iloc[:, -1]
                plt.plot(x, y)
                title = f"{market} cumulative P/L ({passers_sorted.iloc[0]['mode']})"
                if plot_title_suffix:
                    title += f" — {plot_title_suffix}"
                plt.title(title)
                plt.xlabel("Date" if "date" in bets_df.columns else "Bet #")
                plt.ylabel("Cumulative P/L")
                plt.tight_layout()
                plot_name = f"cum_pl_{market}_{timestamp}.png"
                plot_path = os.path.join(plot_dir, plot_name)
                _ensure_parent(plot_path)
                plt.savefig(plot_path, dpi=160)
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] Failed to create plot: {e}")

    return {
        "status": "ok",
        "csv": csv_path,
        "model_pkl": pkl_path,
        "summary_df": csv_df,
        "validation_table": ranked,
        "bets_csv": bets_path,
        "pl_plot": plot_path,
    }

def run_models_outcome_v3_default_models(
    matches_filtered,                       # pd.DataFrame
    features,                               # list[str]

    # ── gates ──────────────────────────────────────────────────────────────
    min_samples: int = 200,
    min_test_samples: int = 100,
    precision_test_threshold: float = 0.80,
    min_val_auc: float = 0.55,
    max_val_brier: float | None = None,

    # ── models to test ─────────────────────────────────────────────────────
    models_to_test: tuple[str, ...] = ("xgb", "mlp", "lr", "rf"),

    # ── light param search per model ───────────────────────────────────────
    search_mode: str = "random",            # "random" | "grid"
    n_random_param_sets: int = 6,           # light search per model
    cpu_jobs: int = 6,
    top_k: int = 10,

    # ── anti-overfitting ───────────────────────────────────────────────────
    val_conf_level: float = 0.99,
    max_precision_drop: float = 1.0,

    # ── failure handling ───────────────────────────────────────────────────
    on_fail: str = "return",                # "return" | "warn" | "raise"
    save_diagnostics_on_fail: bool = True,

    # ── market & VALUE controls ────────────────────────────────────────────
    market: str = "LAY_AWAY",               # LAY_* | BACK_*
    use_value_for_lay: bool = True,
    use_value_for_back: bool = True,
    value_edge_grid_lay=None,               # np.ndarray | None
    value_edge_grid_back=None,              # np.ndarray | None

    # ── LAY minimum odds (single OR list) ──────────────────────────────────
    lay_min_odds: float = 1.50,             # used if lay_min_odds_grid is None
    lay_min_odds_grid=None,                 # e.g. [1.5, 1.8, 2.0, 2.2]

    # ── LAY staking to test ────────────────────────────────────────────────
    lay_staking_plans_to_test: tuple[str, ...] = ("liability", "flat_stake"),

    # BACK staking
    staking_plan_back: str = "flat",

    # BACK staking knobs
    back_stake_test: float = 1.0,
    back_edge_scale: float = 0.10,
    kelly_fraction_back: float = 0.25,
    bankroll_back: float = 100.0,
    min_back_stake: float = 0.0,
    max_back_stake: float = 10.0,

    # ── COMMISSION ─────────────────────────────────────────────────────────
    commission_rate: float = 0.02,

    # ── OUTPUTS ────────────────────────────────────────────────────────────
    out_dir: str | None = None,
    plot_title_suffix: str = "",
):
    """
    VALUE-only FT outcome pipeline (SEARCH/CALIB/TEST) that:
      • tests MULTIPLE models
      • does LIGHT param search per model (random/grid)
      • for LAY: tests staking plans + lay_min_odds_grid + value edge grid
      • tqdm progress bar (incl. parallel)
      • calibrates on CALIB slice, evaluates once on TEST
    """

    # ---------------- setup ----------------
    import os, secrets, hashlib
    from datetime import datetime
    import numpy as np
    import pandas as pd

    from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, log_loss, brier_score_loss
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    import joblib
    from joblib import Parallel, delayed, parallel_backend

    from tqdm import tqdm
    try:
        from tqdm_joblib import tqdm_joblib
        _HAS_TQDM_JOBLIB = True
    except Exception:
        _HAS_TQDM_JOBLIB = False

    # ---------- optional XGBoost ----------
    try:
        import xgboost as xgb
        _HAS_XGB_LOCAL = True
    except Exception:
        _HAS_XGB_LOCAL = False

    # ---------- defaults ----------
    if value_edge_grid_lay is None:
        value_edge_grid_lay = np.round(np.arange(0.00, 0.201, 0.01), 2)
    if value_edge_grid_back is None:
        value_edge_grid_back = np.round(np.arange(0.00, 0.201, 0.01), 2)

    market = str(market).upper().strip()
    _IS_LAY  = market.startswith("LAY_")
    _IS_BACK = market.startswith("BACK_")

    _USE_VALUE_LAY  = bool(use_value_for_lay and _IS_LAY)
    _USE_VALUE_BACK = bool(use_value_for_back and _IS_BACK)
    _USE_VALUE = _USE_VALUE_LAY or _USE_VALUE_BACK
    if not _USE_VALUE:
        raise NotImplementedError("This function is VALUE-only (LAY_* / BACK_* with use_value_* = True).")

    # ---------- lay_min_odds grid normalisation ----------
    if lay_min_odds_grid is None:
        lay_min_odds_grid = [float(lay_min_odds)]
    else:
        if isinstance(lay_min_odds_grid, (float, int, np.floating, np.integer)):
            lay_min_odds_grid = [float(lay_min_odds_grid)]
        else:
            lay_min_odds_grid = [float(x) for x in lay_min_odds_grid]
        lay_min_odds_grid = [x for x in lay_min_odds_grid if np.isfinite(x)]
        if len(lay_min_odds_grid) == 0:
            lay_min_odds_grid = [float(lay_min_odds)]
    lay_min_odds_grid = sorted(set([float(np.round(x, 4)) for x in lay_min_odds_grid]))

    if _IS_LAY:
        for p in lay_staking_plans_to_test:
            if p not in ("liability", "flat_stake"):
                raise ValueError(f"Unsupported LAY staking plan: {p}")

    # ---------- path helpers ----------
    def _normpath(p: str | None) -> str | None:
        if p is None:
            return None
        return os.path.normpath(p)

    def _ensure_dir(path: str | None):
        if path is None:
            return
        os.makedirs(path, exist_ok=True)

    def _ensure_parent(path_to_file: str | None):
        if path_to_file is None:
            return
        parent = os.path.dirname(path_to_file)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ---------- maths helpers ----------
    def _Z(conf):
        try:
            from scipy.stats import norm
            return float(norm.ppf(1 - (1 - conf) / 2))
        except Exception:
            return 2.576 if conf >= 0.99 else 1.96

    def _wilson_lcb(tp: int, fp: int, conf: float) -> float:
        n = tp + fp
        if n <= 0:
            return 0.0
        p = tp / n
        z = _Z(conf)
        denom = 1.0 + (z * z) / n
        centre = p + (z * z) / (2 * n)
        rad = z * np.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
        return max(0.0, (centre - rad) / denom)

    # ---------- dirs (same layout as your v2 for compatibility) ----------
    BASE = _normpath(r"C:\Users\leere\PycharmProjects\Football_ML3\FT Results")
    _ensure_dir(BASE)

    PKL_DIRS = {
        "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "model_file"),
        "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "model_file"),
        "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "model_file"),
        "BACK_HOME": os.path.join(BASE, "Back_Home", "model_file"),
        "BACK_AWAY": os.path.join(BASE, "Back_Away", "model_file"),
        "BACK_DRAW": os.path.join(BASE, "Back_Draw", "model_file"),
    }
    CSV_DIRS = {
        "LAY_HOME":  os.path.join(BASE, "Lay_Home",  "best_model_metrics"),
        "LAY_AWAY":  os.path.join(BASE, "Lay_Away",  "best_model_metrics"),
        "LAY_DRAW":  os.path.join(BASE, "Lay_Draw",  "best_model_metrics"),
        "BACK_HOME": os.path.join(BASE, "Back_Home", "best_model_metrics"),
        "BACK_AWAY": os.path.join(BASE, "Back_Away", "best_model_metrics"),
        "BACK_DRAW": os.path.join(BASE, "Back_Draw", "best_model_metrics"),
    }
    if market not in PKL_DIRS:
        raise ValueError(f"Unsupported market '{market}'.")

    for d in PKL_DIRS.values():
        _ensure_dir(_normpath(d))
    for d in CSV_DIRS.values():
        _ensure_dir(_normpath(d))

    csv_save_dir = _normpath(out_dir) if (out_dir and len(str(out_dir)) > 0) else _normpath(CSV_DIRS[market])
    _ensure_dir(csv_save_dir)

    model_dir = _normpath(PKL_DIRS[market])
    _ensure_dir(model_dir)

    # ---------------- RNG / seeds ----------------
    RUN_SEED = secrets.randbits(32)

    def _seed_from(*vals) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(int(RUN_SEED).to_bytes(8, "little", signed=False))
        for v in vals:
            h.update(str(v).encode("utf-8"))
        return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF

    # ---------------- data ----------------
    df = matches_filtered.copy()
    req_cols = {"date", "target", "home_odds", "draw_odds", "away_odds"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    cols_needed = list(set(features) | {"target", "home_odds", "draw_odds", "away_odds"})
    df = df.dropna(subset=cols_needed).reset_index(drop=True)

    X = df[features].copy()
    y = df["target"].astype(int).reset_index(drop=True)

    n = len(X)
    if n < max(min_samples * 3, 500):
        raise RuntimeError(f"Not enough rows: {n}")

    # ---------------- temporal split: SEARCH / CALIB / TEST -----------------
    search_end = int(0.80 * n)
    calib_end = int(0.85 * n)
    test_start = calib_end
    if search_end <= 0 or test_start <= search_end:
        raise RuntimeError("Data too short for SEARCH/CALIB/TEST split.")

    X_search = X.iloc[:search_end].reset_index(drop=True)
    y_search = y.iloc[:search_end].reset_index(drop=True)
    df_search = df.iloc[:search_end].reset_index(drop=True)

    X_calib = X.iloc[search_end:calib_end].reset_index(drop=True)
    y_calib = y.iloc[search_end:calib_end].reset_index(drop=True)

    X_test = X.iloc[test_start:].reset_index(drop=True)
    y_test = y.iloc[test_start:].reset_index(drop=True)
    df_test = df.iloc[test_start:].reset_index(drop=True)

    # rolling validation folds inside SEARCH region
    N_FOLDS = 5
    pretest_end = len(X_search)
    total_val_len = max(N_FOLDS, int(0.25 * pretest_end))
    total_val_len = min(total_val_len, pretest_end)
    val_len = max(1, total_val_len // N_FOLDS)

    fold_val_ends = [pretest_end - total_val_len + (i + 1) * val_len for i in range(N_FOLDS)]
    fold_val_starts = [end - val_len for end in fold_val_ends]
    if fold_val_ends:
        fold_val_ends[-1] = min(fold_val_ends[-1], pretest_end)
        fold_val_starts[-1] = max(0, fold_val_ends[-1] - val_len)

    # ---------------- param spaces (LIGHT) --------------------------------
    def _param_space(model_key: str):
        mk = str(model_key).lower().strip()

        if mk == "xgb":
            # small, sensible ranges
            grid = {
                "n_estimators": [200, 400],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "min_child_weight": [1, 5],
                "reg_lambda": [1.0, 3.0],
            }
            dists = {
                "n_estimators": [150, 250, 400, 600],
                "max_depth": [2, 3, 4, 5, 6],
                "learning_rate": [0.03, 0.05, 0.07, 0.1, 0.15],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5, 8],
                "reg_lambda": [0.5, 1.0, 2.0, 4.0],
            }
            return grid, dists

        if mk == "mlp":
            grid = {
                "hidden_layer_sizes": [(128,), (256,), (128, 64)],
                "alpha": [1e-4, 5e-4],
                "learning_rate_init": [1e-3, 5e-3],
                "max_iter": [200],
            }
            dists = {
                "hidden_layer_sizes": [(64,), (128,), (256,), (128, 64), (256, 128)],
                "alpha": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                "learning_rate_init": [5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
                "max_iter": [200, 250, 300],
            }
            return grid, dists

        if mk == "lr":
            grid = {"C": [0.3, 1.0, 3.0]}
            dists = {"C": [0.1, 0.3, 1.0, 3.0, 10.0]}
            return grid, dists

        if mk == "rf":
            grid = {
                "n_estimators": [300, 600],
                "max_depth": [None, 8, 14],
                "min_samples_leaf": [1, 2, 4],
            }
            dists = {
                "n_estimators": [200, 300, 500, 800],
                "max_depth": [None, 6, 8, 12, 16],
                "min_samples_leaf": [1, 2, 4, 6],
                "min_samples_split": [2, 4, 8],
                "max_features": ["sqrt", "log2", None],
            }
            return grid, dists

        raise ValueError(f"Unknown model key: {model_key}")

    def _iter_grid(grid: dict):
        from itertools import product
        keys = list(grid.keys())
        vals = [grid[k] for k in keys]
        for combo in product(*vals):
            yield dict(zip(keys, combo))

    def _sample_random(dists: dict, n_iter: int, seed: int):
        rng = np.random.RandomState(seed)
        keys = list(dists.keys())
        out = []
        for _ in range(int(n_iter)):
            p = {}
            for k in keys:
                choices = dists[k]
                p[k] = choices[rng.randint(0, len(choices))]
            out.append(p)
        return out

    # ---------------- model builders --------------------------------------
    def _build_model(model_key: str, params: dict, spw: float):
        mk = str(model_key).lower().strip()
        seed = _seed_from("model", mk, *sorted(map(str, params.items())))

        if mk == "xgb":
            if not _HAS_XGB_LOCAL:
                raise ImportError("XGBoost not available, remove 'xgb' or install xgboost.")
            return xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=seed,
                scale_pos_weight=spw,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
                **params,
            )

        if mk == "mlp":
            mlp = MLPClassifier(
                random_state=seed,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
                solver="adam",
                **params,
            )
            return make_pipeline(StandardScaler(with_mean=True, with_std=True), mlp)

        if mk == "lr":
            lr = LogisticRegression(
                random_state=seed,
                max_iter=2000,
                n_jobs=1,
                **params,
            )
            return make_pipeline(StandardScaler(with_mean=True, with_std=True), lr)

        if mk == "rf":
            # keep class_weight balanced; allow params override
            base = dict(class_weight="balanced", n_jobs=1, random_state=seed)
            base.update(params)
            return RandomForestClassifier(**base)

        raise ValueError(f"Unknown model key: {model_key}")

    def _unwrap_estimator(est):
        if isinstance(est, Pipeline):
            return est.steps[-1][1]
        return est

    def _predict_proba_pos(model_or_cal, X_):
        proba = model_or_cal.predict_proba(X_)
        if proba.ndim == 2:
            classes = getattr(model_or_cal, "classes_", None)
            if classes is None:
                base = _unwrap_estimator(model_or_cal)
                classes = getattr(base, "classes_", None)
            if classes is not None and len(classes) == proba.shape[1]:
                try:
                    idx = int(np.where(np.asarray(classes) == 1)[0][0])
                    return proba[:, idx].astype(np.float32)
                except Exception:
                    pass
            if proba.shape[1] == 2:
                return proba[:, 1].astype(np.float32)
        return np.asarray(proba, dtype=np.float32)

    def _fit_model(model, X_tr, y_tr, X_va=None, y_va=None):
        # early stopping only for raw XGB
        if _HAS_XGB_LOCAL and "xgboost" in str(type(model)).lower():
            try:
                model.set_params(verbosity=0, early_stopping_rounds=50)
                if X_va is not None and y_va is not None:
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                else:
                    model.fit(X_tr, y_tr, verbose=False)
                return
            except Exception:
                pass
        model.fit(X_tr, y_tr)

    def _fit_calibrator(fitted, X_va, y_va):
        try:
            from sklearn.calibration import FrozenEstimator
            frozen = FrozenEstimator(fitted)
            cal = CalibratedClassifierCV(frozen, method="sigmoid", cv=None)
            cal.fit(X_va, y_va)
            return cal
        except Exception:
            try:
                cal = CalibratedClassifierCV(fitted, method="sigmoid", cv="prefit")
                cal.fit(X_va, y_va)
                return cal
            except Exception:
                return fitted

    # ---------------- staking (LAY) ----------------------------
    def _lay_stakes_value(odds: np.ndarray, plan: str):
        o = np.asarray(odds, dtype=float)
        denom = np.maximum(o - 1.0, 1e-12)

        if plan == "liability":
            L = np.ones_like(o, dtype=float)
            S = L / denom
            return S, L

        if plan == "flat_stake":
            S = np.full_like(o, 1.0 / max(1e-12, 1.0 - commission_rate), dtype=float)
            L = S * denom
            return S, L

        raise ValueError(f"Unknown LAY staking plan: {plan}")

    # ---------------- BACK staking (unchanged) ----------------------------
    def _back_stakes(odds: np.ndarray, edge_plus: np.ndarray, plan: str, p_win: np.ndarray):
        o = np.asarray(odds, dtype=float)
        p = np.clip(np.asarray(p_win, dtype=float), 0.0, 1.0)
        edge_plus = np.maximum(edge_plus, 0.0)

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

        return np.clip(stake, float(min_back_stake), float(max_back_stake))

    # ---------------- validation eval -------------------------------------
    def _evaluate_one(model_key: str, params: dict, lay_plan: str | None):
        rows = []
        val_prob_all = []
        val_true_all = []

        for vstart, vend in zip(fold_val_starts, fold_val_ends):
            if vstart is None or vend is None or vstart <= 0 or vend <= vstart:
                continue

            X_tr, y_tr = X_search.iloc[:vstart], y_search.iloc[:vstart]
            X_va, y_va = X_search.iloc[vstart:vend], y_search.iloc[vstart:vend]
            df_va = df_search.iloc[vstart:vend]

            if y_tr.nunique() < 2:
                continue

            pos = int(y_tr.sum()); neg = len(y_tr) - pos
            spw = (neg / pos) if pos > 0 else 1.0

            model = _build_model(model_key, params, spw)
            _fit_model(model, X_tr, y_tr, X_va, y_va)
            cal = _fit_calibrator(model, X_va, y_va)

            p_pos = _predict_proba_pos(cal, X_va)
            val_prob_all.append(p_pos)
            y_true = y_va.values.astype(np.uint8)
            val_true_all.append(y_true)

            if _IS_LAY:
                if market == "LAY_AWAY":
                    mkt = df_va["away_odds"].values.astype(float)
                elif market == "LAY_HOME":
                    mkt = df_va["home_odds"].values.astype(float)
                else:
                    mkt = df_va["draw_odds"].values.astype(float)

                fair = 1.0 / np.clip(1.0 - p_pos, 1e-9, 1.0)

                for min_odds in lay_min_odds_grid:
                    valid = np.isfinite(mkt) & (mkt > 1.01) & (mkt >= float(min_odds))

                    for edge_param in value_edge_grid_lay:
                        edge_param = float(edge_param)
                        mask = valid & (fair >= (1.0 + edge_param) * mkt)
                        y_pred = mask.astype(np.uint8)

                        tp = int(((y_true == 1) & (y_pred == 1)).sum())
                        fp = int(((y_true == 0) & (y_pred == 1)).sum())

                        rows.append({
                            "model_key": model_key,
                            "params": params,
                            "lay_plan": lay_plan,
                            "lay_min_odds": float(min_odds),
                            "edge_param": edge_param,
                            "tp_val": tp,
                            "fp_val": fp,
                            "n_value_bets_val": int(y_pred.sum()),
                            "val_precision": float(precision_score(y_true, y_pred, zero_division=0)),
                            "val_accuracy": float(accuracy_score(y_true, y_pred)),
                        })
            else:
                if market == "BACK_AWAY":
                    mkt = df_va["away_odds"].values.astype(float)
                elif market == "BACK_HOME":
                    mkt = df_va["home_odds"].values.astype(float)
                else:
                    mkt = df_va["draw_odds"].values.astype(float)

                fair = 1.0 / np.clip(p_pos, 1e-9, 1.0)
                valid = np.isfinite(mkt) & (mkt > 1.01)

                for edge_param in value_edge_grid_back:
                    edge_param = float(edge_param)
                    mask = valid & (mkt >= (1.0 + edge_param) * fair)
                    y_pred = mask.astype(np.uint8)

                    tp = int(((y_true == 1) & (y_pred == 1)).sum())
                    fp = int(((y_true == 0) & (y_pred == 1)).sum())

                    rows.append({
                        "model_key": model_key,
                        "params": params,
                        "lay_plan": None,
                        "lay_min_odds": np.nan,
                        "edge_param": edge_param,
                        "tp_val": tp,
                        "fp_val": fp,
                        "n_value_bets_val": int(y_pred.sum()),
                        "val_precision": float(precision_score(y_true, y_pred, zero_division=0)),
                        "val_accuracy": float(accuracy_score(y_true, y_pred)),
                    })

        # pooled metrics across folds
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
            r["val_auc"] = val_auc
            r["val_logloss"] = val_ll
            r["val_brier"] = val_bri

        return rows

    # ---------------- build param sets per model ---------------------------
    def _param_sets_for_model(model_key: str):
        grid, dists = _param_space(model_key)
        if str(search_mode).lower() == "grid":
            ps = list(_iter_grid(grid))
        else:
            ps = _sample_random(dists, int(n_random_param_sets), seed=_seed_from("sampler", model_key))
        # always ensure at least 1
        if not ps:
            ps = [dict()]
        return ps

    # ---------------- run validation search (tqdm + parallel) --------------
    tasks = []
    for mk in models_to_test:
        mk = str(mk).lower().strip()
        param_sets = _param_sets_for_model(mk)
        if _IS_LAY:
            for lp in lay_staking_plans_to_test:
                for pset in param_sets:
                    tasks.append((mk, pset, lp))
        else:
            for pset in param_sets:
                tasks.append((mk, pset, None))

    eff_jobs = max(1, min(int(cpu_jobs), 8))
    total_tasks = len(tasks)

    with parallel_backend("loky", n_jobs=eff_jobs, inner_max_num_threads=1):
        if _HAS_TQDM_JOBLIB:
            with tqdm_joblib(tqdm(total=total_tasks, desc="Model search (val)")):
                outs = Parallel(n_jobs=eff_jobs, batch_size=1)(
                    delayed(_evaluate_one)(mk, pset, lp) for mk, pset, lp in tasks
                )
        else:
            outs = []
            for mk, pset, lp in tqdm(tasks, total=total_tasks, desc="Model search (val)"):
                outs.append(_evaluate_one(mk, pset, lp))

    val_rows = [r for sub in outs for r in sub]
    if not val_rows:
        raise RuntimeError("No validation rows produced (check folds and input data).")

    val_df = pd.DataFrame(val_rows)

    # ---------------- aggregate across folds/params ------------------------
    # We'll serialise params to a stable string key for grouping.
    def _params_key(d: dict) -> str:
        if not isinstance(d, dict):
            return "{}"
        items = sorted((str(k), str(v)) for k, v in d.items())
        return "|".join([f"{k}={v}" for k, v in items])

    val_df["params_key"] = val_df["params"].apply(_params_key)

    group_cols = ["model_key", "params_key", "lay_plan", "edge_param"]
    if _IS_LAY:
        group_cols.insert(3, "lay_min_odds")  # model, params, plan, min_odds, edge
    else:
        if "lay_min_odds" not in val_df.columns:
            val_df["lay_min_odds"] = np.nan

    agg = (
        val_df.groupby(group_cols, as_index=False)
        .agg({
            "tp_val": "sum",
            "fp_val": "sum",
            "n_value_bets_val": "sum",
            "val_precision": "mean",
            "val_accuracy": "mean",
            "val_auc": "mean",
            "val_logloss": "mean",
            "val_brier": "mean",
        })
    )

    agg["val_precision_pooled"] = agg["tp_val"] / np.maximum(1, agg["tp_val"] + agg["fp_val"])
    agg["val_precision_lcb"] = agg.apply(
        lambda r: _wilson_lcb(int(r["tp_val"]), int(r["fp_val"]), conf=val_conf_level),
        axis=1
    )

    qual_mask = (
        (agg["val_precision"] >= float(precision_test_threshold)) &
        (agg["n_value_bets_val"] >= int(min_samples))
    )
    if not np.isnan(agg["val_auc"]).all():
        qual_mask &= (agg["val_auc"] >= float(min_val_auc))
    if max_val_brier is not None and not np.isnan(agg["val_brier"]).all():
        qual_mask &= (agg["val_brier"] <= float(max_val_brier))

    qual = agg[qual_mask].copy()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if qual.empty:
        fail_csv = None
        if save_diagnostics_on_fail:
            fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED.csv")
            _ensure_parent(fail_csv)
            agg.sort_values(
                ["val_precision_lcb", "val_precision", "n_value_bets_val", "val_accuracy"],
                ascending=[False, False, False, False]
            ).to_csv(fail_csv, index=False)

        if on_fail == "raise":
            raise RuntimeError("No (model × params × staking × min_odds × edge) met validation gates.")
        if on_fail == "warn":
            print("[WARN] No (model × params × staking × min_odds × edge) met validation gates.")
        return {
            "status": "failed_validation_gate",
            "csv": fail_csv,
            "model_pkl": None,
            "summary_df": None,
            "validation_table": agg.sort_values(
                ["val_precision_lcb", "val_precision", "n_value_bets_val", "val_accuracy"],
                ascending=[False, False, False, False]
            ).reset_index(drop=True),
        }

    ranked = qual.sort_values(
        ["val_precision_lcb", "val_precision", "n_value_bets_val", "val_accuracy"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    topk_val = ranked.head(int(top_k)).reset_index(drop=True)

    # map params_key -> real dict (from first occurrence)
    params_lookup = (
        val_df[["params_key", "params"]]
        .drop_duplicates("params_key")
        .set_index("params_key")["params"]
        .to_dict()
    )

    # ---------------- FINAL FIT + TEST eval (no further search) ------------
    def _get_market_odds(df_):
        if market.endswith("AWAY"):
            return df_["away_odds"].values.astype(float), "AWAY"
        if market.endswith("HOME"):
            return df_["home_odds"].values.astype(float), "HOME"
        return df_["draw_odds"].values.astype(float), "DRAW"

    results = []
    for _, row in tqdm(topk_val.iterrows(), total=len(topk_val), desc="Final fit + test"):
        model_key = row["model_key"]
        params_key = row["params_key"]
        params = params_lookup.get(params_key, {}) or {}
        lay_plan = row["lay_plan"] if _IS_LAY else None
        edge_param = float(row["edge_param"])
        chosen_min_odds = float(row["lay_min_odds"]) if _IS_LAY else np.nan

        pos = int(y_search.sum()); neg = len(y_search) - pos
        spw = (neg / pos) if pos > 0 else 1.0

        model = _build_model(model_key, params, spw)
        _fit_model(model, X_search, y_search, X_calib, y_calib)
        cal = _fit_calibrator(model, X_calib, y_calib)

        p_pos_test = _predict_proba_pos(cal, X_test)
        mkt_odds, sel_name = _get_market_odds(df_test)

        if _IS_LAY:
            fair = 1.0 / np.clip(1.0 - p_pos_test, 1e-9, 1.0)
            valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01) & (mkt_odds >= chosen_min_odds)
            value_mask = valid & (fair >= (1.0 + edge_param) * mkt_odds)

            S_plan, L_plan = _lay_stakes_value(mkt_odds, plan=str(lay_plan))
            S = np.where(value_mask, S_plan, 0.0)
            L = np.where(value_mask, L_plan, 0.0)

            sel_occurs = (y_test.values == 0)  # your convention
            pl = np.where(~value_mask, 0.0, np.where(sel_occurs, -L, S * (1.0 - commission_rate)))

            n_bets = int(np.count_nonzero(S > 0))
            y_pred = (S > 0).astype(np.uint8)
            prc_test = float(precision_score(y_test, y_pred, zero_division=0))
            acc_test = float(accuracy_score(y_test, y_pred))
            total_pl = float(pl.sum())
            avg_pl = float(total_pl / max(1, n_bets))

            pass_gate = (
                (n_bets >= int(min_test_samples)) and
                (prc_test >= max(float(precision_test_threshold), float(row["val_precision"]) - float(max_precision_drop)))
            )

            results.append({
                "model_key": model_key,
                "params_key": params_key,
                "lay_plan": lay_plan,
                "lay_min_odds": chosen_min_odds,
                "edge_param": edge_param,
                "val_precision_lcb": float(row["val_precision_lcb"]),
                "val_precision": float(row["val_precision"]),
                "val_auc": float(row["val_auc"]),
                "val_brier": float(row["val_brier"]),
                "n_value_bets_test": n_bets,
                "test_precision_bets": prc_test,
                "test_accuracy_bets": acc_test,
                "total_pl": total_pl,
                "avg_pl": avg_pl,
                "pass_test_gate": bool(pass_gate),
                "model_obj": cal if pass_gate else None,
                "commission_rate": float(commission_rate),
            })

        else:
            p_sel_win = p_pos_test
            fair = 1.0 / np.clip(p_sel_win, 1e-9, 1.0)
            valid = np.isfinite(mkt_odds) & (mkt_odds > 1.01)

            edge_plus = np.maximum(mkt_odds / np.maximum(fair, 1e-12) - 1.0, 0.0)
            value_mask = valid & (mkt_odds >= (1.0 + edge_param) * fair)

            stake = np.zeros_like(mkt_odds, dtype=float)
            s = _back_stakes(mkt_odds, edge_plus, staking_plan_back, p_sel_win)
            stake[value_mask] = s[value_mask]

            sel_occurs = (y_test.values == 1)
            pl = np.where((stake > 0) & sel_occurs, (mkt_odds - 1.0) * stake * (1.0 - commission_rate), 0.0)
            pl = np.where((stake > 0) & (~sel_occurs), -stake, pl)

            n_bets = int(np.count_nonzero(stake > 0))
            y_pred = (stake > 0).astype(np.uint8)
            prc_test = float(precision_score(y_test, y_pred, zero_division=0))
            acc_test = float(accuracy_score(y_test, y_pred))
            total_pl = float(pl.sum())
            avg_pl = float(total_pl / max(1, n_bets))

            pass_gate = (
                (n_bets >= int(min_test_samples)) and
                (prc_test >= max(float(precision_test_threshold), float(row["val_precision"]) - float(max_precision_drop)))
            )

            results.append({
                "model_key": model_key,
                "params_key": params_key,
                "lay_plan": None,
                "lay_min_odds": np.nan,
                "edge_param": edge_param,
                "val_precision_lcb": float(row["val_precision_lcb"]),
                "val_precision": float(row["val_precision"]),
                "val_auc": float(row["val_auc"]),
                "val_brier": float(row["val_brier"]),
                "n_value_bets_test": n_bets,
                "test_precision_bets": prc_test,
                "test_accuracy_bets": acc_test,
                "total_pl": total_pl,
                "avg_pl": avg_pl,
                "pass_test_gate": bool(pass_gate),
                "model_obj": cal if pass_gate else None,
                "commission_rate": float(commission_rate),
            })

    survivors_df = pd.DataFrame(results)
    passers = survivors_df[survivors_df["pass_test_gate"]].copy()

    if passers.empty:
        fail_csv = None
        if save_diagnostics_on_fail:
            fail_csv = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}_FAILED_TEST.csv")
            _ensure_parent(fail_csv)
            survivors_df.drop(columns=["model_obj"], errors="ignore").to_csv(fail_csv, index=False)

        if on_fail == "raise":
            raise RuntimeError("All Top-K failed the TEST gate.")
        if on_fail == "warn":
            print("[WARN] All Top-K failed the TEST gate.")
        return {
            "status": "failed_test_gate",
            "csv": fail_csv,
            "model_pkl": None,
            "summary_df": survivors_df.drop(columns=["model_obj"], errors="ignore"),
            "validation_table": ranked,
        }

    passers_sorted = passers.sort_values(
        by=["total_pl", "avg_pl", "val_precision_lcb", "val_precision"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    # save best model
    best = passers_sorted.iloc[0]
    best_params = params_lookup.get(best["params_key"], {}) or {}

    pkl_path = os.path.join(model_dir, f"best_model_{market}_{best['model_key']}_{timestamp}.pkl")
    _ensure_parent(pkl_path)

    joblib.dump(
        {
            "model": best["model_obj"],
            "features": features,
            "model_key": best["model_key"],
            "best_params": best_params,
            "market": market,
            "edge_param": float(best["edge_param"]),
            "lay_plan": best.get("lay_plan", None) if _IS_LAY else None,
            "lay_min_odds": float(best.get("lay_min_odds", np.nan)) if _IS_LAY else np.nan,
            "commission_rate": float(commission_rate),
            "notes": "v3: model sweep + light param search + staking sweep + min_odds_grid.",
            "run_seed": int(RUN_SEED),
        },
        pkl_path,
    )

    # save metrics CSV
    csv_path = os.path.join(csv_save_dir, f"model_metrics_{market}_{timestamp}.csv")
    _ensure_parent(csv_path)
    passers_sorted.drop(columns=["model_obj"], errors="ignore").to_csv(csv_path, index=False)

    return {
        "status": "ok",
        "csv": csv_path,
        "model_pkl": pkl_path,
        "summary_df": passers_sorted.drop(columns=["model_obj"], errors="ignore"),
        "validation_table": ranked,
        "lay_min_odds_grid_used": lay_min_odds_grid if _IS_LAY else None,
        "plot_title_suffix": plot_title_suffix,
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
