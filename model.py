import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import xgboost as xgb

# Load your dataset
data = pd.read_csv('covid_19_clean_complete.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Sample the dataset to speed up training
data = data.sample(frac=0.5, random_state=42)

# Split the data before normalization
split = int(0.8 * len(data))
train_data = data.iloc[:split].copy()
test_data = data.iloc[split:].copy()

# Normalize training data
scaler = MinMaxScaler()
train_data['cases_normalized'] = scaler.fit_transform(train_data[['Confirmed']])

# Normalize test data using the same scaler
test_data['cases_normalized'] = scaler.transform(test_data[['Confirmed']])

# Create lag features
def create_lag_features(data, lags):
    df = pd.DataFrame(data)
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[0].shift(lag)
    df.dropna(inplace=True)
    return df

lags = 10
lagged_train_data = create_lag_features(train_data['cases_normalized'].values, lags)
lagged_test_data = create_lag_features(test_data['cases_normalized'].values, lags)

# Extract features and target
X_train = lagged_train_data.drop(columns=0).values
y_train = lagged_train_data[0].values
X_test = lagged_test_data.drop(columns=0).values
y_test = lagged_test_data[0].values

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [10, 50],
    'max_depth': [3, 4, 5],
    'min_samples_split': [5, 10]
}

model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')

# Train the best model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f'Mean Squared Error on Training Set: {mse_train}')
print(f'Mean Squared Error on Test Set: {mse_test}')

# Cross-validation score
cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
print(f'Cross-Validation MSE: {-cv_scores.mean()}')

# Plot actual vs predicted cases
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Cases', color='b')
plt.plot(y_pred_test, label='Predicted Cases', color='r')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Cases')
plt.title('Actual vs Predicted Cases')
plt.legend()
plt.show()

# Optional: Using XGBoost for comparison
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=50, max_depth=5)
xgb_model.fit(X_train, y_train)

# Predict and evaluate XGBoost model
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'Mean Squared Error on Test Set (XGBoost): {mse_xgb}')
data['Moving_Avg'] = data['Confirmed'].rolling(window=7).mean()




