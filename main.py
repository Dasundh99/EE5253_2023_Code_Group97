# Concrete Strength Prediction
# Group no. 97
# EG/2020/3964 - Hettiarachchi UADD
# EG/2020/4061 - Maduwantha BKTR

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the Dataset
file_path = "ConcreteStrengthData.csv"
data = pd.read_csv(file_path)

# Displaying Sample Dataset
print("Sample of the Dataset:")
print(data.head())

# Display statistical features distribution for each feature
print("Statistical Features Distribution:")
print(data.describe())
print(data.describe(include='all'))

# Display missing data heatmap before handling missing data
plt.figure(figsize=(12, 8))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap (Before Handling)")
plt.savefig("missing_data_heatmap_before.png")
plt.show()

# Extract features and target variable
features = data.iloc[:, :-1]
target = data.iloc[:, -1]

# Handling missing data
imputer = SimpleImputer(strategy='mean')
features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

# Display missing data heatmap after handling missing data
plt.figure(figsize=(12, 8))
sns.heatmap(features.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap (After Handling)")
plt.savefig("missing_data_heatmap_after.png")
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Results:")
print("Mean Squared Error: {}".format(mse))
print("R-squared: {}".format(r2))

# Scatter plot for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Concrete Strength")
plt.ylabel("Predicted Concrete Strength")
plt.savefig("linear_regression_scatter_plot.png")

# Residual plot for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, y_test - y_pred)
plt.title("Linear Regression: Residual Plot")
plt.xlabel("Predicted Concrete Strength")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.savefig("linear_regression_residual_plot.png")

# Decision Trees
# Create a decision tree model
tree_model = DecisionTreeRegressor()

# Train the decision tree model
tree_model.fit(X_train, y_train)

# Make predictions on the test set using the decision tree model
tree_y_pred = tree_model.predict(X_test)

# Evaluate the decision tree model
tree_mse = mean_squared_error(y_test, tree_y_pred)
tree_r2 = r2_score(y_test, tree_y_pred)

print("\nDecision Tree Results:")
print("Mean Squared Error: {}".format(tree_mse))
print("R-squared: {}".format(tree_r2))

# Measure accuracy for Decision Tree algorithm
accuracy = tree_model.score(X_test, y_test)
print("Accuracy for Decision tree: {:.2%}".format(accuracy))

# Scatter plot for Decision Tree Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, tree_y_pred)
plt.title("Decision Tree Regression: Actual vs Predicted")
plt.xlabel("Actual Concrete Strength")
plt.ylabel("Predicted Concrete Strength")
plt.savefig("decision_tree_scatter_plot.png")

# Residual plot for Decision Tree Regression
plt.figure(figsize=(10, 6))
plt.scatter(tree_y_pred, y_test - tree_y_pred)
plt.title("Decision Tree Regression: Residual Plot")
plt.xlabel("Predicted Concrete Strength")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.savefig("decision_tree_residual_plot.png")

# Save the plots as image files
plt.close('all')  # Close all open figures

# Correlation Analysis
# Include the target variable in the features for correlation analysis
all_features = pd.concat([features, target], axis=1)

# Compute the correlation matrix
correlation_matrix = all_features.corr()

# Plot the correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix (Including Target Variable)")
plt.savefig("correlation_matrix_heatmap.png")
plt.show()

# Scatter plot for Cement vs Compressive Strength
plt.figure(figsize=(10, 6))
plt.scatter(features['CementComponent '], target)
plt.title("Cement vs Compressive Strength")
plt.xlabel("Cement Content")
plt.ylabel("Compressive Strength")
plt.savefig("cement_vs_compressive_strength_scatter_plot.png")
plt.show()

# Scatter plot for Fine Aggregate vs Compressive Strength
plt.figure(figsize=(10, 6))
plt.scatter(features['FineAggregateComponent'], target)
plt.title("Fine Aggregate vs Compressive Strength")
plt.xlabel("Fine Aggregate Content")
plt.ylabel("Compressive Strength")
plt.savefig("fine_aggregate_vs_compressive_strength_scatter_plot.png")
plt.show()

