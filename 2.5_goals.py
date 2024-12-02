import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

data = data.drop(columns=['home_team', 'away_team', 'home_goals', 'away_goals', 'total_goals', 'o2.5_odds'])
# Apply one-hot encoding to 'Country' and 'League'
data = pd.get_dummies(data, columns=['Country', 'League'])

# Split the data into training and testing sets by date
cut_off_date = data['Date'].quantile(0.8)
train_data = data[data['Date'] <= cut_off_date]
y_train = train_data["over_2.5_goals"]
train_data = train_data.drop(columns=['Date', 'over_2.5_goals'])
test_data = data[data['Date'] > cut_off_date]
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
    {
        'model': [LogisticRegression()],
        'model__C': [0.1, 1, 10, 100],
        'model__solver': ['liblinear', 'lbfgs'],
        'model__max_iter': [100, 200],
    },
    # {
    #     'model': [RandomForestClassifier()],
    #     'model__n_estimators': [50, 100, 200],
    #     'model__max_depth': [None, 10, 20],
    #     'model__min_samples_split': [2, 5, 10],
    # },

]

# Initialize GridSearchCV with multiple models
grid_search = GridSearchCV(estimator=Pipeline([('model', LogisticRegression())]),
                           param_grid=param_grid,
                           cv=3,
                           scoring='accuracy',
                           verbose=10,
                           n_jobs=-1)

# Fit the GridSearch to the training data
grid_search.fit(train_data_scaled, y_train)

# Retrieve the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(test_data_scaled)

# Evaluate the model with suitable metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print confusion matrix and metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Calculate betting results
profit = 0
for index, pred in enumerate(y_pred):
    if pred == 1:
        # Simulate placing a bet with potential profit or loss
        odds = data.iloc[test_data.index[index]]['o2.5_odds']
        if y_test.iloc[index] == 1:
            profit += (odds * 1) - 1  # Calculate profit
        else:
            profit -= 1  # Lose the bet

# Total number of bets placed
total_stake = y_pred.sum()
roi = profit/total_stake

# Print betting results
print(f"\nTotal Stake: £{total_stake}")
print(f"Net Profit: £{profit:.2f}")
print(f"ROI: {roi:.1f}%")

# Print the best parameters found
print("\nBest parameters:", grid_search.best_params_)
