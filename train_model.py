import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Load dataset
df = pd.read_excel('student_data.xlsx')

X = df.drop('final_score', axis=1)
y = df['final_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

print("Linear Regression Results")
print("MAE:", mean_absolute_error(y_test, lr_preds))
print("R2 Score:", r2_score(y_test, lr_preds))
print()

# Random Forest model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("Random Forest Results")
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("R2 Score:", r2_score(y_test, rf_preds))


pickle.dump(lr, open('student_score_model.pkl', 'wb'))

print("Model saved as student_score_model.pkl")
