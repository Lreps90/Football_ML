# import datetime
import random as rd
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


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


import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("XGBOOST_VERBOSITY", "0")

from datetime import datetime
from itertools import product
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    precision_score, accuracy_score
)
from sklearn.model_selection import ParameterSampler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

def run_models_v2(
    matches_filtered: pd.DataFrame,
    features: list,
    ht_score: tuple | str,
    min_samples: int = 200,            # validation gate: min predicted positives across folds
    min_test_samples: int = 100,       # test gate: min predicted positives on hold-out test
    precision_test_threshold: float = 0.80,
    base_model: str = "xgb",           # "xgb" or "mlp"
    search_mode: str = "random",       # "random" or "grid"
    n_random_param_sets: int = 10,
    cpu_jobs: int = 6,
    top_k: int = 10,
    thresholds: np.ndarray | None = None,
    out_dir: str | None = None,
    # --- anti-overfitting knobs ---
    val_conf_level: float = 0.95,      # Wilson-LCB confidence for validation precision
    max_precision_drop: float = 0.10,  # allow at most 10pp drop val → test
    # --- failure handling ---
    on_fail: str = "return",           # "return" | "warn" | "raise"
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
        denom = 1.0 + (z*z)/n
        centre = p + (z*z)/(2*n)
        rad = z * sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
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
    X_val_final   = X.iloc[final_val_start:test_start]
    y_val_final   = y.iloc[final_val_start:test_start]

    # ------------------ Hyper-parameter spaces ------------------
    xgb_param_grid = {
        'n_estimators':      [200],
        'max_depth':         [5],
        'learning_rate':     [0.1],
        'subsample':         [0.7],
        'colsample_bytree':  [1.0],
        'min_child_weight':  [5],
        'reg_lambda':        [1.0],
    }
    xgb_param_distributions = {
        'n_estimators':      _randint(100, 1001),
        'max_depth':         _randint(3, 8),
        'learning_rate':     _loguniform(0.01, 0.2),
        'min_child_weight':  _randint(3, 13),
        'subsample':         _uniform(0.7, 0.3),
        'colsample_bytree':  _uniform(0.6, 0.4),
        'reg_lambda':        _loguniform(0.1, 10.0),
    }
    mlp_param_grid = {
        'hidden_layer_sizes': [(128,), (256,), (128, 64)],
        'alpha':              [1e-4],
        'learning_rate_init': [1e-3],
        'batch_size':         ['auto'],
        'max_iter':           [200],
    }
    mlp_param_distributions = {
        'hidden_layer_sizes': [(64,), (128,), (256,), (128, 64), (256, 128)],
        'alpha':              _loguniform(1e-5, 1e-2),
        'learning_rate_init': _loguniform(5e-4, 5e-2),
        'batch_size':         _randint(32, 257),
        'max_iter':           _randint(150, 401),
    }

    if search_mode.lower() == "grid":
        grid, dists = (xgb_param_grid, None) if base_model == "xgb" else (mlp_param_grid, None)
        all_param_dicts = [dict(zip(grid.keys(), combo)) for combo in product(*grid.values())]
    else:
        grid, dists = (xgb_param_grid, xgb_param_distributions) if base_model == "xgb" else (mlp_param_grid, mlp_param_distributions)
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

            pos = int(y_tr.sum()); neg = len(y_tr) - pos
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
            try: val_auc = float(roc_auc_score(vt, vp))
            except Exception: val_auc = np.nan
            try: val_ll  = float(log_loss(vt, vp, labels=[0, 1]))
            except Exception: val_ll  = np.nan
            try: val_bri = float(brier_score_loss(vt, vp))
            except Exception: val_bri = np.nan
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
        prefer = "threads"; backend = "threading"; pre_dispatch = eff_jobs
        ctx = parallel_backend(backend, n_jobs=eff_jobs)
    else:
        eff_jobs = max(1, min(cpu_jobs, 4)) if cpu_jobs != -1 else 4
        prefer = "processes"; backend = "loky"; pre_dispatch = f"{2*eff_jobs}"
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
            (agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
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
            'validation_table': agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
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
    records_all = []   # every candidate with test metrics + pass/fail reason
    for cand in candidates:
        best_params = cast_params(cand['params'])
        pos = int(y_train_final.sum()); neg = len(y_train_final) - pos
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
                    .sort_values(by=['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
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
        'summary_df': csv_df,           # passers only, with model_pkl set on row 0
        'validation_table': ranked,     # full validation ranking (post-gates)
    }

def run_models_o25(
    matches_filtered: pd.DataFrame,
    features: list,
    min_samples: int = 200,            # validation gate: min predicted positives across folds
    min_test_samples: int = 100,       # test gate: min predicted positives on hold-out test
    precision_test_threshold: float = 0.80,
    base_model: str = "xgb",           # "xgb" or "mlp"
    search_mode: str = "random",       # "random" or "grid"
    n_random_param_sets: int = 10,
    cpu_jobs: int = 6,
    top_k: int = 10,
    thresholds: np.ndarray | None = None,
    out_dir: str | None = None,
    # --- anti-overfitting knobs ---
    val_conf_level: float = 0.99,      # Wilson-LCB confidence for validation precision
    max_precision_drop: float = 0.02,  # allow at most 10pp drop val → test
    # --- failure handling ---
    on_fail: str = "return",           # "return" | "warn" | "raise"
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
        denom = 1.0 + (z*z)/n
        centre = p + (z*z)/(2*n)
        rad = z * sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
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
    X_val_final   = X.iloc[final_val_start:test_start]
    y_val_final   = y.iloc[final_val_start:test_start]

    # ------------------ Hyper-parameter spaces ------------------
    xgb_param_grid = {
        'n_estimators':      [200],
        'max_depth':         [5],
        'learning_rate':     [0.1],
        'subsample':         [0.7],
        'colsample_bytree':  [1.0],
        'min_child_weight':  [5],
        'reg_lambda':        [1.0],
    }
    xgb_param_distributions = {
        'n_estimators':      _randint(100, 1001),
        'max_depth':         _randint(3, 8),
        'learning_rate':     _loguniform(0.01, 0.2),
        'min_child_weight':  _randint(3, 13),
        'subsample':         _uniform(0.7, 0.3),
        'colsample_bytree':  _uniform(0.6, 0.4),
        'reg_lambda':        _loguniform(0.1, 10.0),
    }
    mlp_param_grid = {
        'hidden_layer_sizes': [(128,), (256,), (128, 64)],
        'alpha':              [1e-4],
        'learning_rate_init': [1e-3],
        'batch_size':         ['auto'],
        'max_iter':           [200],
    }
    mlp_param_distributions = {
        'hidden_layer_sizes': [(64,), (128,), (256,), (128, 64), (256, 128)],
        'alpha':              _loguniform(1e-5, 1e-2),
        'learning_rate_init': _loguniform(5e-4, 5e-2),
        'batch_size':         _randint(32, 257),
        'max_iter':           _randint(150, 401),
    }

    if search_mode.lower() == "grid":
        grid, dists = (xgb_param_grid, None) if base_model == "xgb" else (mlp_param_grid, None)
        all_param_dicts = [dict(zip(grid.keys(), combo)) for combo in product(*grid.values())]
    else:
        grid, dists = (xgb_param_grid, xgb_param_distributions) if base_model == "xgb" else (mlp_param_grid, mlp_param_distributions)
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

            pos = int(y_tr.sum()); neg = len(y_tr) - pos
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
            try: val_auc = float(roc_auc_score(vt, vp))
            except Exception: val_auc = np.nan
            try: val_ll  = float(log_loss(vt, vp, labels=[0, 1]))
            except Exception: val_ll  = np.nan
            try: val_bri = float(brier_score_loss(vt, vp))
            except Exception: val_bri = np.nan
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
        prefer = "threads"; backend = "threading"; pre_dispatch = eff_jobs
        ctx = parallel_backend(backend, n_jobs=eff_jobs)
    else:
        eff_jobs = max(1, min(cpu_jobs, 4)) if cpu_jobs != -1 else 4
        prefer = "processes"; backend = "loky"; pre_dispatch = f"{2*eff_jobs}"
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
            (agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
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
            'validation_table': agg.sort_values(['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
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
    records_all = []   # every candidate with test metrics + pass/fail reason
    for cand in candidates:
        best_params = cast_params(cand['params'])
        pos = int(y_train_final.sum()); neg = len(y_train_final) - pos
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
                    .sort_values(by=['val_precision_lcb','val_precision','n_preds_val','val_accuracy'],
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
        pkl_path = os.path.join(r"C:\Users\leere\PycharmProjects\Football_ML3\Goals\Over_2_5\model_file", f"best_model_{tag}_calibrated_{timestamp}.pkl")
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
        'summary_df': csv_df,           # passers only, with model_pkl set on row 0
        'validation_table': ranked,     # full validation ranking (post-gates)
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
    return np.random.default_rng(abs(hash(random_state)) % (2**32))

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



def run_models_with_probs(data, features, filename_feature, min_samples=100, apply_calibration=True):
    """
    Run grid-search experiments over different models.
    If apply_pca is True, PCA is applied using various variance thresholds.
    If apply_calibration is True, probability calibration is applied to each fitted model
    using the sigmoid method (Platt scaling).
    The metrics computed include standard evaluation metrics plus summary statistics
    of the calibrated probabilities.
    """
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

    counts = Counter(y_train_full)
    minority_class = min(counts, key=counts.get)
    majority_class = max(counts, key=counts.get)
    current_ratio = counts[minority_class] / counts[majority_class]
    print("Current minority/majority ratio:", current_ratio)

    # Define SMOTE strategies and probability thresholds.
    upper_bound = 0.95  # adjust as needed
    smote_strategies = [round(x, 2) for x in np.arange(current_ratio + 0.01, upper_bound, 0.05)] + [None]
    probability_thresholds = [round(x, 2) for x in np.arange(0.2, 0.81, 0.01)]

    metrics_list = []

    for apply_pca in [True, False]:
        if apply_pca:
            var_thresholds = [0.9, 0.92, 0.94, 0.96, 0.98]
        else:
            var_thresholds = [1]

        # Compute master total tests.
        temp_pipelines, temp_param_grids = build_pipelines(apply_pca=apply_pca)
        master_total_tests = 0
        for model in temp_param_grids:
            num_param = len(list(ParameterGrid(temp_param_grids[model])))
            master_total_tests += len(var_thresholds) * len(smote_strategies) * num_param * len(probability_thresholds)
        print(f"Total tests to be performed: {master_total_tests}")

        if apply_pca:
            total_outer_runs = len(var_thresholds) * len(smote_strategies) * len(temp_pipelines)
        else:
            total_outer_runs = len(smote_strategies) * len(temp_pipelines)
        outer_run_counter = 0
        print(f"Total grid search outer runs: {total_outer_runs}")

        master_test_counter = 0

        # Loop over whether to use PCA or not.
        for pca_flag in [apply_pca]:
            if pca_flag:
                for var_threshold in var_thresholds:
                    optimal_n_components = select_optimal_pca_components(X_train_full, variance_threshold=var_threshold)
                    print(f"PCA: {optimal_n_components} components for {var_threshold * 100:.0f}% variance")
                    pipelines, param_grids = build_pipelines(apply_pca=True)
                    for model_name, pipeline in pipelines.items():
                        pipeline.set_params(pca__n_components=optimal_n_components)
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
                                f"Run {outer_run_counter}/{total_outer_runs} - Model: {model_name}, SMOTE: {smote_label}, Params: {num_params}")
                            for params in ParameterGrid(param_grids[model_name]):
                                pipeline = pipelines[model_name]
                                pipeline.set_params(**params)
                                pipeline.fit(X_train_res, y_train_res)

                                # Calibrate probabilities if requested.
                                if apply_calibration:
                                    calibrated_pipeline = CalibratedClassifierCV(estimator=pipeline,
                                                                                 method='sigmoid',
                                                                                 cv='prefit')
                                    calibrated_pipeline.fit(X_train_res, y_train_res)
                                    current_pipeline = calibrated_pipeline
                                else:
                                    current_pipeline = pipeline

                                train_probs = current_pipeline.predict_proba(X_train_res)[:, 1]
                                test_probs = current_pipeline.predict_proba(X_test)[:, 1]
                                cal_train_mean = np.mean(train_probs)
                                cal_test_mean = np.mean(test_probs)

                                for thresh in probability_thresholds:
                                    master_test_counter += 1
                                    train_pred = (train_probs >= thresh).astype(int)
                                    test_pred = (test_probs >= thresh).astype(int)
                                    if np.sum(test_pred) < min_samples:
                                        continue
                                    train_sample_size = np.sum(train_pred)
                                    test_sample_size = np.sum(test_pred)

                                    train_mcc = matthews_corrcoef(y_train_res, train_pred)
                                    test_mcc = matthews_corrcoef(y_test, test_pred)
                                    mcc_ratio = test_mcc / (train_mcc + 1e-10)
                                    if mcc_ratio > 1:
                                        mcc_ratio = train_mcc / (test_mcc + 1e-10)

                                    train_acc = accuracy_score(y_train_res, train_pred)
                                    test_acc = accuracy_score(y_test, test_pred)
                                    acc_ratio = test_acc / (train_acc + 1e-10)
                                    if acc_ratio > 1:
                                        acc_ratio = train_acc / (test_acc + 1e-10)

                                    train_f1 = f1_score(y_train_res, train_pred)
                                    test_f1 = f1_score(y_test, test_pred)
                                    f1_ratio = test_f1 / (train_f1 + 1e-10)
                                    if f1_ratio > 1:
                                        f1_ratio = train_f1 / (test_f1 + 1e-10)

                                    train_auc = roc_auc_score(y_train_res, train_probs)
                                    test_auc = roc_auc_score(y_test, test_probs)
                                    auc_ratio = test_auc / (train_auc + 1e-10)
                                    if auc_ratio > 1:
                                        auc_ratio = train_auc / (test_auc + 1e-10)

                                    train_precision = precision_score(y_train_res, train_pred, zero_division=0)
                                    test_precision = precision_score(y_test, test_pred, zero_division=0)
                                    precision_ratio = test_precision / (train_precision + 1e-10)
                                    if precision_ratio > 1:
                                        precision_ratio = train_precision / (test_precision + 1e-10)

                                    train_recall = recall_score(y_train_res, train_pred, zero_division=0)
                                    test_recall = recall_score(y_test, test_pred, zero_division=0)
                                    recall_ratio = test_recall / (train_recall + 1e-10)
                                    if recall_ratio > 1:
                                        recall_ratio = train_recall / (test_recall + 1e-10)

                                    if (auc_ratio > 0.8 and precision_ratio > 0.9):
                                        metrics_list.append({
                                            'Model': model_name,
                                            'SMOTE': smote_label,
                                            'Probability_Threshold': thresh,
                                            'AUC_Train': round(train_auc, 4),
                                            'AUC_Test': round(test_auc, 4),
                                            'AUC_Ratio': round(auc_ratio, 4),
                                            'Precision_Train': round(train_precision, 4),
                                            'Precision_Test': round(test_precision, 4),
                                            'Precision_Ratio': round(precision_ratio, 4),
                                            'MCC_Train': round(train_mcc, 4),
                                            'MCC_Test': round(test_mcc, 4),
                                            'MCC_Ratio': round(mcc_ratio, 4),
                                            'ACC_Train': round(train_acc, 4),
                                            'ACC_Test': round(test_acc, 4),
                                            'ACC_Ratio': round(acc_ratio, 4),
                                            'F1_Train': round(train_f1, 4),
                                            'F1_Test': round(test_f1, 4),
                                            'F1_Ratio': round(f1_ratio, 4),
                                            'Recall_Train': round(train_recall, 4),
                                            'Recall_Test': round(test_recall, 4),
                                            'Recall_Ratio': round(recall_ratio, 4),
                                            'Train_Sample_Size': train_sample_size,
                                            'Test_Sample_Size': test_sample_size,
                                            'Calibrated_Train_Prob_Mean': round(cal_train_mean, 4),
                                            'Calibrated_Test_Prob_Mean': round(cal_test_mean, 4),
                                            'Var_Threshold': var_threshold,
                                            'Params': params
                                        })
            else:
                # Running models without PCA (similar logic; calibration can be applied similarly)
                print("Running models without PCA not shown here for brevity.")

    print(f"Total tests performed: {master_test_counter}")
    filename_out = f"model_metrics_{filename_feature}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.sort_values(by='Precision_Test', ascending=False, inplace=True)
    metrics_df.to_csv(filename_out, index=False)
    print(f"Model metrics saved to {filename_out}")


def running_ttest_p_profit(x):
    """
    One-sided (greater) t-test on sample x against popmean=0.
    Returns p-value for H1: mean > 0.
    """
    x = np.asarray(x)
    if len(x) < 2:
        return np.nan
    t_stat, p_two = stats.ttest_1samp(x, popmean=0, nan_policy='omit')
    return (p_two / 2) if (t_stat > 0) else (1 - p_two / 2)

models_params = [
    (
        'XGBoost',
        XGBClassifier,
        {
            # Trees / learning‐rate trade‐off
            'n_estimators':   [100, 300],
            'learning_rate':  [0.05, 0.1],
            # Shallow vs moderate depth
            'max_depth':      [3, 5],
            # Row & column sampling
            'subsample':      [0.8, 1.0],
            'colsample_bytree': [0.8],
            # Light regularisation
            'gamma':          [0.0, 0.1],
            'reg_alpha':      [0.0, 0.1],
            'reg_lambda':     [1.0],
            # for imbalanced data you can compute:
            # 'scale_pos_weight': [sum(neg)/sum(pos)]
        }
    ),
    (
        'MLP',
        MLPClassifier,
        {
            # small vs medium single‐layer nets
            'hidden_layer_sizes': [(50,), (100,)],
            # L2 penalty
            'alpha':           [1e-4, 1e-3, 1e-2],
            # fixed learning rate
            'learning_rate_init': [1e-3],
            'max_iter':        [1000],
            'early_stopping':  [True],
            # you get a 10%‐validation split for the stopping criterion by default
        }
    )
]

# Main function

def run_value_betting(data,
                      features,
                      target_col='target',
                      odds_feature='over_25_odds',
                      models_params=models_params,
                      cv_splits=5,
                      calibrate_cv=3,
                      test_size=0.2,
                      random_state=42,
                      min_bets=50,
                      filename_feature='value_bet'):
    """
    Uses time-series cross-validation on the training split, then evaluates on a final test set,
    printing metrics for both train (CV) and test for each hyperparameter run.
    P-values use a one-sided test: H0 mean <= 0 vs H1 mean > 0.
    """
    if models_params is None:
        models_params = [
            (
                'XGBoost',
                XGBClassifier,
                {
                    'n_estimators': [100, 500],
                    'max_depth': [3, 7],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8],
                    'gamma': [0, 0.2],
                    'min_child_weight': [1, 5],
                    'reg_alpha': [0, 0.1],
                    'reg_lambda': [1, 2],
                    'scale_pos_weight': [1]
                }
            ),
            (
                'MLP',
                MLPClassifier,
                {
                    'hidden_layer_sizes': [(100,), (100, 50)],
                    'activation': ['relu'],
                    'solver': ['adam'],
                    'alpha': [1e-3, 1e-2, 1e-1, 1],
                    'learning_rate': ['constant'],
                    'learning_rate_init': [1e-3],
                    'max_iter': [1000],
                    'early_stopping': [True]
                }
            )
        ]

    # split data
    train_df, test_df = train_test_split(data, test_size=test_size, shuffle=False)
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    all_results = []
    total_runs = sum(len(list(ParameterGrid(grid))) for _, _, grid in models_params)
    run_counter = 0

    for name, ModelClass, grid in models_params:
        for params in ParameterGrid(grid):
            run_counter += 1
            print(f"=== Run {run_counter}/{total_runs}: {name}, params={params} ===")

            # TRAIN CV METRICS
            cv_profits = []
            for train_idx, val_idx in tscv.split(train_df):
                tr = train_df.iloc[train_idx]
                val = train_df.iloc[val_idx]

                model = ModelClass(**params)
                if 'random_state' in model.get_params():
                    model.set_params(random_state=random_state)
                pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
                pipe.fit(tr[features], tr[target_col])

                calib = CalibratedClassifierCV(pipe, method='sigmoid', cv=calibrate_cv)
                calib.fit(tr[features], tr[target_col])

                probs = calib.predict_proba(val[features])[:, 1]
                val = val.copy()
                val['model_odds'] = 1.0 / (probs + 1e-12)
                bets_val = val.loc[val['model_odds'] < val[odds_feature]].copy()
                if not bets_val.empty:
                    bets_val['profit'] = np.where(
                        bets_val[target_col] == 1,
                        bets_val[odds_feature] - 1,
                        -1
                    )
                    cv_profits.append(bets_val['profit'])

            # aggregate train CV
            if cv_profits:
                train_profits = pd.concat(cv_profits, ignore_index=True)
                n_train = len(train_profits)
                train_profit = train_profits.sum()
                train_roi = train_profit / n_train
                train_sr = (train_profits > 0).sum() / n_train
                train_p = running_ttest_p_profit(train_profits)
            else:
                n_train = train_profit = train_roi = train_sr = train_p = None

            # TEST METRICS
            model_f = ModelClass(**params)
            if 'random_state' in model_f.get_params():
                model_f.set_params(random_state=random_state)
            pipe_f = Pipeline([('scaler', StandardScaler()), ('model', model_f)])
            pipe_f.fit(train_df[features], train_df[target_col])

            calib_f = CalibratedClassifierCV(pipe_f, method='sigmoid', cv=calibrate_cv)
            calib_f.fit(train_df[features], train_df[target_col])

            probs_t = calib_f.predict_proba(test_df[features])[:, 1]
            test = test_df.copy()
            test['model_odds'] = 1.0 / (probs_t + 1e-12)
            bets_test = test.loc[test['model_odds'] < test[odds_feature]].copy()
            if not bets_test.empty:
                bets_test['profit'] = np.where(
                    bets_test[target_col] == 1,
                    bets_test[odds_feature] - 1,
                    -1
                )
                n_test = len(bets_test)
                test_profit = bets_test['profit'].sum()
                test_roi = test_profit / n_test
                test_sr = bets_test[target_col].sum() / n_test
                test_p = running_ttest_p_profit(bets_test['profit'])
            else:
                n_test = test_profit = test_roi = test_sr = test_p = None

            # print metrics
            print(f"Train CV: bets={n_train}, profit={train_profit}, roi={train_roi:.3f}, "
                  f"sr={train_sr:.3f}, pval={train_p}")
            print(f"Test Set: bets={n_test}, profit={test_profit}, roi={test_roi:.3f}, "
                  f"sr={test_sr:.3f}, pval={test_p}")

            # collect if valid
            if all(v is not None for v in [n_train, n_test, train_profit, test_profit]) \
               and n_train >= min_bets and n_test >= min_bets and test_profit > 0:
                all_results.append({
                    'model_name': name,
                    'num_bets_train': n_train,
                    'total_profit_train': round(train_profit, 4),
                    'roi_train': round(train_roi, 4),
                    'strike_rate_train': round(train_sr, 4),
                    'pvalue_train': round(train_p, 4) if train_p is not None else None,
                    'num_bets_test': n_test,
                    'total_profit_test': round(test_profit, 4),
                    'roi_test': round(test_roi, 4),
                    'strike_rate_test': round(test_sr, 4),
                    'pvalue_test': round(test_p, 4) if test_p is not None else None
                })

    df = pd.DataFrame(all_results)
    if df.empty:
        print("No profitable runs found.")
        return df

    df.sort_values('total_profit_test', ascending=False, inplace=True)
    top100 = df.head(100)

    csv = f"value_betting_top100_{filename_feature}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
    top100.to_csv(csv, index=False)
    print(f"Saved top-100 to {csv}")
    return top100

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_score

# Assumes running_ttest_p_profit is defined elsewhere
def run_models_with_probs_v2(
    data,
    features,
    models_params=models_params,
    thresholds=np.arange(0.1, 0.9, 0.1),
    test_size=0.2,
    random_state=42,
    apply_calibration=True,
    min_samples=25,
    filename_feature='model_probs'
):
    """
    1. Splits `data` into train/test by `test_size` (default 80/20).
    2. For each model/parameter combination:
       - Trains model (with optional calibration) on train set.
       - Predicts probabilities on both train and test sets.
       - For each threshold, computes:
         * Number of bets
         * AUC, Precision
         * Profit/Loss using `over_25_odds`
         * ROI, P-value
         * Precision ratio (test/train)
       - If BOTH train and test P/L > 0, runs model on full data and computes overall Bets, P/L, ROI, P-value.
         * Prints overall metrics with threshold included.
    3. Aggregates runs into DataFrame, ensures full-data columns exist.
    4. Saves CSV only if at least one run has both `pl_train>0` and `pl_test>0`.
    """
    # Split data
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    X_tr, y_tr = train_data[features], train_data['target']
    X_te, y_te = test_data[features], test_data['target']

    odds_tr = train_data['over_25_odds'].values
    odds_te = test_data['over_25_odds'].values
    odds_full = data['over_25_odds'].values

    all_metrics = []
    total_runs = sum(len(list(ParameterGrid(grid))) for _, _, grid in models_params)
    run_count = 0

    for name, ModelClass, grid in models_params:
        for params in ParameterGrid(grid):
            run_count += 1
            print(f"\n=== Run {run_count}/{total_runs}: {name} | params={params} ===")

            # Train and optionally calibrate
            model = ModelClass(**params)
            if 'random_state' in model.get_params():
                model.set_params(random_state=random_state)
            pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            pipe.fit(X_tr, y_tr)
            if apply_calibration:
                clf = CalibratedClassifierCV(pipe, method='sigmoid', cv='prefit')
                clf.fit(X_tr, y_tr)
            else:
                clf = pipe

            # Predict probabilities
            prob_tr = clf.predict_proba(X_tr)[:, 1]
            prob_te = clf.predict_proba(X_te)[:, 1]

            # Compute AUCs
            auc_tr = roc_auc_score(y_tr, prob_tr)
            auc_te = roc_auc_score(y_te, prob_te)

            # Loop thresholds
            for t in thresholds:
                mask_tr = prob_tr >= t
                mask_te = prob_te >= t
                n_tr = int(mask_tr.sum())
                n_te = int(mask_te.sum())

                if n_tr < min_samples or n_te < min_samples:
                    continue

                prec_tr = precision_score(y_tr, mask_tr)
                prec_te = precision_score(y_te, mask_te)

                profits_tr = np.where(
                    mask_tr & (y_tr.values == 1), odds_tr - 1,
                    np.where(mask_tr, -1, 0)
                )
                profits_te = np.where(
                    mask_te & (y_te.values == 1), odds_te - 1,
                    np.where(mask_te, -1, 0)
                )

                total_pl_tr = profits_tr.sum()
                total_pl_te = profits_te.sum()

                roi_tr = total_pl_tr / n_tr
                roi_te = total_pl_te / n_te

                pval_tr = running_ttest_p_profit(profits_tr[mask_tr].tolist())
                pval_te = running_ttest_p_profit(profits_te[mask_te].tolist())

                ratio = prec_te / (prec_tr + 1e-10)

                # Print train/test metrics
                print(
                    f"Threshold {t:.2f} | "
                    f"Train: bets={n_tr}, auc={auc_tr:.3f}, prec={prec_tr:.3f}, pl={total_pl_tr:.2f}, roi={roi_tr:.3f}, pval={pval_tr:.3f} | "
                    f"Test:  bets={n_te}, auc={auc_te:.3f}, prec={prec_te:.3f}, pl={total_pl_te:.2f}, roi={roi_te:.3f}, pval={pval_te:.3f} | "
                    f"Ratio: {ratio:.3f}"
                )

                # Prepare record with default full-data fields
                rec = {
                    'model': name,
                    'params': params,
                    'threshold': t,
                    'bets_train': n_tr,
                    'auc_train': round(auc_tr, 4),
                    'prec_train': round(prec_tr, 4),
                    'pl_train': round(total_pl_tr, 2),
                    'roi_train': round(roi_tr, 4),
                    'pval_train': round(pval_tr, 4),
                    'bets_test': n_te,
                    'auc_test': round(auc_te, 4),
                    'prec_test': round(prec_te, 4),
                    'pl_test': round(total_pl_te, 2),
                    'roi_test': round(roi_te, 4),
                    'pval_test': round(pval_te, 4),
                    'prec_ratio': round(ratio, 4),
                    'bets_full': np.nan,
                    'pl_full': np.nan,
                    'roi_full': np.nan,
                    'pval_full': np.nan
                }

                # If both profits positive, compute full-data metrics
                if total_pl_tr > 0 and total_pl_te > 0:
                    prob_full = clf.predict_proba(data[features])[:, 1]
                    mask_full = prob_full >= t
                    n_full = int(mask_full.sum())
                    if n_full >= min_samples:
                        profits_full = np.where(
                            mask_full & (data['target'].values == 1), odds_full - 1,
                            np.where(mask_full, -1, 0)
                        )
                        overall_pl = profits_full.sum()
                        overall_roi = overall_pl / n_full
                        overall_pval = running_ttest_p_profit(profits_full[mask_full].tolist())

                        # Print full-data metrics
                        print(
                            f"Full data @ threshold {t:.2f}: bets={n_full}, pl={overall_pl:.2f}, roi={overall_roi:.3f}, pval={overall_pval:.3f}"
                        )

                        # Update record
                        rec.update({
                            'bets_full': n_full,
                            'pl_full': round(overall_pl, 2),
                            'roi_full': round(overall_roi, 4),
                            'pval_full': round(overall_pval, 4)
                        })
                    else:
                        print(
                            f"Full data @ threshold {t:.2f}: fewer than {min_samples} bets, skipping overall metrics"
                        )

                all_metrics.append(rec)

    # Finalize DataFrame
    df = pd.DataFrame(all_metrics)
    # Only save if at least one run has pl_train>0 and pl_test>0
    positive_mask = (df['pl_train'] > 0) & (df['pl_test'] > 0)
    df_to_save = df[positive_mask]
    if not df_to_save.empty:
        df_to_save.sort_values('prec_ratio', ascending=False, inplace=True)
        csv = f"model_probs_{filename_feature}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
        df_to_save.to_csv(csv, index=False)
        print(f"Results saved to {csv}")
    else:
        print("No runs with positive P/L on both train and test; CSV not saved.")
    return df

def run_models_with_probs_v3(
        data,
        features,
        models_params=None,
        thresholds=np.arange(0.1, 0.9, 0.1),
        test_size=0.2,
        random_state=42,
        apply_calibration=True,
        min_samples=25,
        filename_feature='model_probs'
):
        """
        1. Splits `data` into train/test by `test_size` (default 80/20).
        2. For each model/parameter combination:
           - Trains model (with optional calibration) on train set (uses eval_set if early_stopping_rounds passed for XGBoost).
           - Predicts probabilities on both train and test sets.
           - For each threshold, computes:
             * Number of bets
             * AUC, Precision
             * Profit/Loss using `over_25_odds`
             * ROI, P-value
             * Precision ratio (test/train)
           - If BOTH train and test P/L > 0, runs model on full data and computes overall Bets, P/L, ROI, P-value.
             * Prints overall metrics with threshold included.
        3. Aggregates runs into DataFrame, ensures full-data columns exist.
        4. Saves CSV only if at least one run has both `pl_train>0` and `pl_test>0`.
        """
        models_params = [
            (
                'XGBoost',
                XGBClassifier,
                {
                    'n_estimators': [100, 300, 500],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'gamma': [0.0, 0.1],
                    'reg_alpha': [0.0, 0.1],
                    'reg_lambda': [1.0, 2.0],
                    'scale_pos_weight': [1, (data['target'] == 0).sum() / (data['target'] == 1).sum()],
                    #'early_stopping_rounds': [50]
                }
            ),
            (
                'MLP',
                MLPClassifier,
                {
                    'hidden_layer_sizes': [(100,), (100, 50), (100, 50, 25)],
                    'alpha': [1e-4, 1e-3, 1e-2],
                    'learning_rate_init': [1e-3],
                    'max_iter': [1000],
                    'early_stopping': [True],
                    'validation_fraction': [0.1],
                    'n_iter_no_change': [20]
                }
            )
        ]
        # Split data
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
        X_tr, y_tr = train_data[features], train_data['target']
        X_te, y_te = test_data[features], test_data['target']

        odds_tr = train_data['over_25_odds'].values
        odds_te = test_data['over_25_odds'].values
        odds_full = data['over_25_odds'].values

        all_metrics = []
        total_runs = sum(len(list(ParameterGrid(grid))) for _, _, grid in models_params)
        run_count = 0

        for name, ModelClass, grid in models_params:
            for params in ParameterGrid(grid):
                run_count += 1
                print(f"\n=== Run {run_count}/{total_runs}: {name} | params={params} ===")

                # Copy and extract early stopping for XGB
                params_copy = params.copy()
                es_rounds = params_copy.pop('early_stopping_rounds', None)

                # Instantiate model
                model = ModelClass(**params_copy)
                if hasattr(model, 'random_state'):
                    model.set_params(random_state=random_state)

                pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])

                # Fit with early stopping if XGB
                if isinstance(model, XGBClassifier) and es_rounds is not None:
                    pipe.fit(
                        X_tr, y_tr,
                        model__eval_set=[(X_te, y_te)],
                        model__early_stopping_rounds=es_rounds,
                        model__verbose=False
                    )
                else:
                    pipe.fit(X_tr, y_tr)

                # Calibration
                if apply_calibration:
                    clf = CalibratedClassifierCV(pipe, method='sigmoid', cv='prefit')
                    clf.fit(X_tr, y_tr)
                else:
                    clf = pipe

                # Predict probabilities
                prob_tr = clf.predict_proba(X_tr)[:, 1]
                prob_te = clf.predict_proba(X_te)[:, 1]

                # Compute AUCs
                auc_tr = roc_auc_score(y_tr, prob_tr)
                auc_te = roc_auc_score(y_te, prob_te)

                # Loop thresholds
                for t in thresholds:
                    mask_tr = prob_tr >= t
                    mask_te = prob_te >= t
                    n_tr = int(mask_tr.sum())
                    n_te = int(mask_te.sum())

                    if n_tr < min_samples or n_te < min_samples:
                        continue

                    prec_tr = precision_score(y_tr, mask_tr)
                    prec_te = precision_score(y_te, mask_te)

                    profits_tr = np.where(
                        mask_tr & (y_tr.values == 1), odds_tr - 1,
                        np.where(mask_tr, -1, 0)
                    )
                    profits_te = np.where(
                        mask_te & (y_te.values == 1), odds_te - 1,
                        np.where(mask_te, -1, 0)
                    )

                    total_pl_tr = profits_tr.sum()
                    total_pl_te = profits_te.sum()

                    roi_tr = total_pl_tr / n_tr
                    roi_te = total_pl_te / n_te

                    pval_tr = running_ttest_p_profit(profits_tr[mask_tr].tolist())
                    pval_te = running_ttest_p_profit(profits_te[mask_te].tolist())

                    ratio = prec_te / (prec_tr + 1e-10)

                    # Print train/test metrics
                    print(
                        f"Threshold {t:.2f} | "
                        f"Train: bets={n_tr}, auc={auc_tr:.3f}, prec={prec_tr:.3f}, pl={total_pl_tr:.2f}, roi={roi_tr:.3f}, pval={pval_tr:.3f} | "
                        f"Test:  bets={n_te}, auc={auc_te:.3f}, prec={prec_te:.3f}, pl={total_pl_te:.2f}, roi={roi_te:.3f}, pval={pval_te:.3f} | "
                        f"Ratio: {ratio:.3f}"
                    )

                    # Prepare record
                    rec = {
                        'model': name,
                        'params': params_copy,
                        'threshold': t,
                        'bets_train': n_tr,
                        'auc_train': round(auc_tr, 4),
                        'prec_train': round(prec_tr, 4),
                        'pl_train': round(total_pl_tr, 2),
                        'roi_train': round(roi_tr, 4),
                        'pval_train': round(pval_tr, 4),
                        'bets_test': n_te,
                        'auc_test': round(auc_te, 4),
                        'prec_test': round(prec_te, 4),
                        'pl_test': round(total_pl_te, 2),
                        'roi_test': round(roi_te, 4),
                        'pval_test': round(pval_te, 4),
                        'prec_ratio': round(ratio, 4),
                        'bets_full': np.nan,
                        'pl_full': np.nan,
                        'roi_full': np.nan,
                        'pval_full': np.nan
                    }

                    # Full-data metrics
                    if total_pl_tr > 0 and total_pl_te > 0:
                        prob_full = clf.predict_proba(data[features])[:, 1]
                        mask_full = prob_full >= t
                        n_full = int(mask_full.sum())
                        if n_full >= min_samples:
                            profits_full = np.where(
                                mask_full & (data['target'].values == 1), odds_full - 1,
                                np.where(mask_full, -1, 0)
                            )
                            overall_pl = profits_full.sum()
                            overall_roi = overall_pl / n_full
                            overall_pval = running_ttest_p_profit(profits_full[mask_full].tolist())

                            print(
                                f"Full data @ threshold {t:.2f}: bets={n_full}, pl={overall_pl:.2f}, roi={overall_roi:.3f}, pval={overall_pval:.3f}"
                            )

                            rec.update({
                                'bets_full': n_full,
                                'pl_full': round(overall_pl, 2),
                                'roi_full': round(overall_roi, 4),
                                'pval_full': round(overall_pval, 4)
                            })
                        else:
                            print(
                                f"Full data @ threshold {t:.2f}: fewer than {min_samples} bets, skipping overall metrics"
                            )

                    all_metrics.append(rec)

        # Finalize DataFrame
        df = pd.DataFrame(all_metrics)
        # Save only if positive P/L trains/tests
        mask_pos = (df['pl_train'] > 0) & (df['pl_test'] > 0)
        df_to_save = df[mask_pos]
        if not df_to_save.empty:
            df_to_save.sort_values('prec_ratio', ascending=False, inplace=True)
            csv = f"model_probs_{filename_feature}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
            df_to_save.to_csv(csv, index=False)
            print(f"Results saved to {csv}")
        else:
            print("No runs with positive P/L on both train and test; CSV not saved.")

        return df







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
