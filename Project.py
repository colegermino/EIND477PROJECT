import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np  # Added for RMSE calculation

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv("C:/Users/freds/OneDrive/Documents/EIND477/EIND477PROJECT/car_dataset.csv")

print(df.head())

label_encoders = {}
categorical_cols = ['Car_Name', 'Year', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop(['Selling_Price','Present_Price'], axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
# Manually calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)



importances = model.feature_importances_
feature_names = X.columns

# Create a Series for better visualization
feat_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)

# Plot
feat_series.plot(kind='barh', title="Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()


from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(model.estimators_[0], feature_names=X.columns, filled=True, max_depth=3)  # depth limited for clarity
plt.show()


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # reference line
plt.grid(True)
plt.show()

