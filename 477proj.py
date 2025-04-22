"""


This script performs the following steps:
1. Loads the dataset.
2. Performs exploratory data analysis using descriptive statistics and visualizations.
3. Preprocesses the data (encoding categorical variables and handling missing values if any).
4. Splits the data into training (80%) and testing (20%) sets.
5. Trains a Random Forest Regressor on the training data.
6. Evaluates model performance using R² and RMSE metrics.
7. Visualizes the actual versus predicted values.

The Random Forest algorithm is an ensemble method that builds many decision trees using random subsets of data
and features. The final prediction is made by averaging the outputs (for regression) over all trees.
This approach helps reduce overfitting and improves generalization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn tools for splitting the data and training the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# ------------------------------
# 1. Load and Explore the Dataset
# ------------------------------
# Load the dataset; ensure that "car_dataset.csv" is in your working directory.
df = pd.read_csv('/Users/colegermino/Desktop/car_dataset.csv')
print("Dataset shape:", df.shape)
print("\nSummary statistics:")
print(df.describe(include='all'))

# ------------------------------
# 2. Exploratory Data Analysis (EDA)
# ------------------------------

# Plot histograms for numerical variables to check their distribution.
num_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[num_columns].hist(bins=15, figsize=(15, 10))
plt.suptitle("Histograms of Numerical Variables")
plt.show()

# Create scatter plots for key relationships.
# Example: If the target is 'Price', we can look at Price vs. EngineSize.
# Modify the column names based on your dataset.
target_variable = 'Present_Price'  # CHANGE this to the actual target column name
example_feature = 'EngineSize'  # CHANGE this to an example input feature name

if target_variable in df.columns and example_feature in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=example_feature, y=target_variable)
    plt.title(f"Scatter Plot of {target_variable} vs. {example_feature}")
    plt.xlabel(example_feature)
    plt.ylabel(target_variable)
    plt.show()

# ------------------------------
# 3. Data Preprocessing
# ------------------------------

# Identify categorical variables: here we assume objects are categorical.
cat_columns = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns detected:", cat_columns)

# For simplicity, use one-hot encoding for all categorical variables.
if cat_columns:
    df_encoded = pd.get_dummies(df, columns=cat_columns, drop_first=True)
else:
    df_encoded = df.copy()

# Check for missing values and fill or drop if needed.
if df_encoded.isnull().sum().sum() > 0:
    # For the purpose of this project, we simply fill numerical NaNs with the median;
    # more sophisticated methods can be used.
    for col in df_encoded.columns:
        if df_encoded[col].dtype in ['int64', 'float64']:
            df_encoded[col].fillna(df_encoded[col].median(), inplace=True)
        else:
            df_encoded[col].fillna(df_encoded[col].mode()[0], inplace=True)

print("\nData after preprocessing (encoded) shape:", df_encoded.shape)

# ------------------------------
# 4. Splitting Data into Training and Testing Sets
# ------------------------------

# Choose input features (X) and output variable (y).
# Modify this selection based on which columns are relevant for your project.
if target_variable not in df_encoded.columns:
    raise ValueError(f"The target variable '{target_variable}' was not found in the dataset.")

X = df_encoded.drop(target_variable, axis=1)
y = df_encoded[target_variable]

# Split the data: 80% training and 20% testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} samples; Testing set size: {X_test.shape[0]} samples")

# ------------------------------
# 5. Train Random Forest Regressor
# ------------------------------

# Initialize the model.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data.
rf_model.fit(X_train, y_train)
print("\nRandom Forest model trained.")

# ------------------------------
# 6. Model Evaluation
# ------------------------------

# Predict the target variable on the test set.
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics.
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# ------------------------------
# 7. Visualization of Results
# ------------------------------

# Plot Actual vs Predicted values.
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title("Actual vs. Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

# Plot feature importances from the Random Forest model.
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title("Feature Importances")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()
# ------------------------------
# Eval
# ------------------------------
# 1. Print the full list of predictor variables
print("Predictor variables used in the model:")
print(X.columns.tolist())

# 2. Compute and display feature importances
import pandas as pd

feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

# Print all importances
print("\nFeature importances (sorted):")
print(feature_importances)

# Or just the top 10 most important features:
print("\nTop 10 features by importance:")
print(feature_importances.head(10))


# ------------------------------
# 4. Simplify to top‐4 features only
# ------------------------------

# List of top 4 predictors from feature_importances_
top_features = [
    'Selling_Price',
    'Car_Name_land cruiser',
    'Car_Name_fortuner',
    'Kms_Driven'
]

# Subset your encoded DataFrame to only those columns
X = df_encoded[top_features]
y = df_encoded[target_variable]

# Now split and train as before

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Simplified Model Metrics:")
print(f"  RMSE: {rmse:.2f}")
print(f"  R²:   {r2:.2f}")
# ------------------------------
# End of Project Script
# ------------------------------