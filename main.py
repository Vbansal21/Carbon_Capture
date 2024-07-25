import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import plot_partial_dependence
import shap
import lime
import lime.lime_tabular

# Optional: Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Step 1: Read the Data from CSV
data = pd.read_csv('data.csv', header=1)  # assuming the first row contains the headers
print(data.head())

# Step 2: Exploratory Data Analysis (EDA)

# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Visualize missing values
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot
sns.pairplot(data)
plt.show()

# Distribution of features
data.hist(bins=30, figsize=(20, 15))
plt.show()

# Step 3: Clean the Data
data.fillna(data.mean(), inplace=True)

# Step 4: Split the Data into Features and Target
X = data[['SA (m2/g)', 'TPV (cm3/g)', 'MPV (cm3/g)', 'C%', 'H%', 'N%', 'O%', 'T (℃)', 'P (bar)']]  # specify the columns correctly
y = data['CO2 uptake (mmol/g)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize the Features (if necessary)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    return r2, mae, mse, rmse, mape

# Hyperparameter Tuning and Cross-Validation
def hyperparameter_tuning(model, param_grid):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    print(f'Best Parameters: {grid_search.best_params_}')
    return best_model

# Define parameter grids for each model
param_grids = {
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'GBDT': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    'LightGBM': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    'CatBoost': {'iterations': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
}

# Train and Evaluate Various Models
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'GBDT': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42),
    'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
    'AdaBoost': AdaBoostRegressor(random_state=42)
}

results = {}
best_models = {}

for name, model in models.items():
    print(f'Tuning {name}...')
    tuned_model = hyperparameter_tuning(model, param_grids[name])
    best_models[name] = tuned_model
    results[name] = evaluate_model(tuned_model, X_test_scaled, y_test)

# Display the results
for name, metrics in results.items():
    print(f"{name} - R^2: {metrics[0]:.2f}, MAE: {metrics[1]:.2f}, MSE: {metrics[2]:.2f}, RMSE: {metrics[3]:.2f}, MAPE: {metrics[4]:.2f}")

# SHAP and LIME Explanations, and Partial Dependence Plots for each model
features = [
    ('SA (m2/g)', 'TPV (cm3/g)'), ('SA (m2/g)', 'MPV (cm3/g)'), ('SA (m2/g)', 'N%'), ('SA (m2/g)', 'O%'), ('SA (m2/g)', 'T (℃)'),
    ('MPV (cm3/g)', 'TPV (cm3/g)'), ('C%', 'H%'), ('N%', 'H%')
]

for name, model in best_models.items():
    if not name == 'AdaBoost':
        print(f"Generating SHAP values for {name}...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)
        shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
        plt.title(f'SHAP Summary Plot for {name}')
        plt.show()

        # Generate SHAP force plot for the first instance
        shap.initjs()
        shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], matplotlib=True)
        plt.title(f'SHAP Force Plot for {name} - Instance 0')
        plt.show()

    # LIME Explanation
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train_scaled, feature_names=X.columns, mode='regression')
    print(f"Generating LIME explanations for {name}...")
    for i in range(5):  # Explaining first 5 instances
        exp = lime_explainer.explain_instance(X_test_scaled[i], model.predict, num_features=10)
        exp.show_in_notebook(show_table=True)

    # Partial Dependence Plots
    print(f"Generating Partial Dependence Plots for {name}...")
    plot_partial_dependence(model, X_train_scaled, features, feature_names=X.columns, grid_resolution=50)
    plt.suptitle(f'Partial Dependence Plots for {name}')
    plt.show()
