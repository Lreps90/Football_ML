import datetime
import random as rd
from collections import Counter
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV


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
    optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Optimal number of components to explain 95% variance: {optimal_components}")
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

        # SVC
        pipelines['SVC'] = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(svd_solver='randomized', random_state=42)),
            ('classifier', SVC(probability=True))
        ])
        param_grids['SVC'] = {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma': ['scale', 'auto']
        }

        # Logistic Regression
        pipelines['LogisticRegression'] = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(svd_solver='randomized', random_state=42)),
            ('classifier', LogisticRegression(max_iter=10000))
        ])
        param_grids['LogisticRegression'] = {
            'classifier__C': [0.1, 1, 10],
            'classifier__solver': ['lbfgs', 'saga']
        }

        # K-Nearest Neighbours
        pipelines['KNN'] = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(svd_solver='randomized', random_state=42)),
            ('classifier', KNeighborsClassifier())
        ])
        param_grids['KNN'] = {
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__weights': ['uniform', 'distance']
        }

        # AdaBoost
        pipelines['AdaBoost'] = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(svd_solver='randomized', random_state=42)),
            ('classifier', AdaBoostClassifier(random_state=42, algorithm='SAMME'))
        ])
        param_grids['AdaBoost'] = {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.5, 1.0]
        }

        # Stacking Ensembles
        ensemble_configs = {
            "StackingEnsemble1": [
                ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
                ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
                ('ada', AdaBoostClassifier(random_state=42, algorithm='SAMME'))
            ],
            "StackingEnsemble2": [
                ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
                ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
                ('mlp', MLPClassifier(random_state=42, max_iter=10000))
            ],
            "StackingEnsemble3": [
                ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
                ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
                ('svc', SVC(probability=True))
            ],
            "StackingEnsemble4": [
                ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
                ('ada', AdaBoostClassifier(random_state=42, algorithm='SAMME')),
                ('knn', KNeighborsClassifier())
            ],
            "StackingEnsemble5": [
                ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
                ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
                ('ada', AdaBoostClassifier(random_state=42, algorithm='SAMME')),
                ('mlp', MLPClassifier(random_state=42, max_iter=10000))
            ],
        }
        for ens_name, estimators in ensemble_configs.items():
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=10000)
            )
            pipelines[ens_name] = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(svd_solver='randomized', random_state=42)),
                ('classifier', stacking)
            ])
            param_grids[ens_name] = {
                'classifier__final_estimator__C': [0.1, 1, 10]
            }

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

        pipelines['RandomForest'] = Pipeline([
            ('scaler', StandardScaler()),
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

        pipelines['SVC'] = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(probability=True))
        ])
        param_grids['SVC'] = {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma': ['scale', 'auto']
        }

        pipelines['LogisticRegression'] = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=10000))
        ])
        param_grids['LogisticRegression'] = {
            'classifier__C': [0.1, 1, 10],
            'classifier__solver': ['lbfgs', 'saga']
        }

        pipelines['KNN'] = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
        param_grids['KNN'] = {
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__weights': ['uniform', 'distance']
        }

        pipelines['AdaBoost'] = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', AdaBoostClassifier(random_state=42, algorithm='SAMME'))
        ])
        param_grids['AdaBoost'] = {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.5, 1.0]
        }

        ensemble_configs = {
            "StackingEnsemble1": [
                ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
                ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
                ('ada', AdaBoostClassifier(random_state=42, algorithm='SAMME'))
            ],
            "StackingEnsemble2": [
                ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
                ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
                ('mlp', MLPClassifier(random_state=42, max_iter=10000))
            ],
            "StackingEnsemble3": [
                ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
                ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
                ('svc', SVC(probability=True))
            ],
            "StackingEnsemble4": [
                ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
                ('ada', AdaBoostClassifier(random_state=42, algorithm='SAMME')),
                ('knn', KNeighborsClassifier())
            ],
            "StackingEnsemble5": [
                ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
                ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
                ('ada', AdaBoostClassifier(random_state=42, algorithm='SAMME')),
                ('mlp', MLPClassifier(random_state=42, max_iter=10000))
            ],
        }
        for ens_name, estimators in ensemble_configs.items():
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=10000)
            )
            pipelines[ens_name] = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', stacking)
            ])
            param_grids[ens_name] = {
                'classifier__final_estimator__C': [0.1, 1, 10]
            }

    return pipelines, param_grids


def run_models(data, features, filename_feature, min_samples=100, apply_pca=True):
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
    #lower_bound = max(np.ceil(current_ratio * 100) / 100, 0.55)
    upper_bound = 0.95  # Adjust as necessary
    #smote_strategies = [round(x, 2) for x in np.arange(lower_bound, upper_bound, 0.05)] + [None]
    smote_strategies = [round(x, 2) for x in np.arange(current_ratio + 0.01, upper_bound, 0.05)] + [None]
    probability_thresholds = [round(x, 2) for x in np.arange(0.2, 0.81, 0.01)]

    metrics_list = []

    for apply_pca in [True, False]:
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
                                if (auc_ratio > 0.8 and
                                        precision_ratio > 0.8):
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
                            if (auc_ratio > 0.8 and
                                    precision_ratio > 0.8):
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
                                    'Var_Threshold': var_threshold,
                                    'Params': params
                                })

    print(f"Total tests performed: {master_test_counter}")
    # Save the results to a CSV file with a timestamp in the filename that includes the filename_feature.
    filename_out = f"model_metrics_{filename_feature}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    metrics_df = pd.DataFrame(metrics_list)
    # Sort by Precision_Test (greatest at the top)
    metrics_df.sort_values(by='Precision_Test', ascending=False, inplace=True)
    metrics_df.to_csv(filename_out, index=False)
    print(f"Model metrics saved to {filename_out}")


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