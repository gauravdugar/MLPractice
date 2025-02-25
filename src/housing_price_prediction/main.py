import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from models.linear_regression_model import train_linear_regression
from models.decision_tree_model import train_decision_tree
from models.random_forest_model import train_random_forest
from models.svr_model import train_svr

def load_csv_data(file_path):
    return pd.read_csv(file_path)

def main():
    file_path = "./data/housing_sample.csv"
    data = load_csv_data(file_path)
    print("Data Preview:")
    print(data.head())

    x = data.drop('MedHouseVal', axis=1)
    y = data['MedHouseVal']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print("Training Linear Regression...")
    lr_pred, lr_mse, lr_r2 = train_linear_regression(x_train, y_train, x_test, y_test)

    print("Training Decision Tree...")
    dt_pred, dt_mse, dt_r2 = train_decision_tree(x_train, y_train, x_test, y_test)

    print("Training Random Forest...")
    rf_pred, rf_mse, rf_r2 = train_random_forest(x_train, y_train, x_test, y_test)

    print("Training SVR...")
    svr_pred, svr_mse, svr_r2 = train_svr(x_train, y_train, x_test, y_test)

    print("\nModel Comparison:")
    print(f"Linear Regression - MSE: {lr_mse:.3f}, R^2: {lr_r2:.3f}")
    print(f"Decision Tree     - MSE: {dt_mse:.3f}, R^2: {dt_r2:.3f}")
    print(f"Random Forest     - MSE: {rf_mse:.3f}, R^2: {rf_r2:.3f}")
    print(f"SVR               - MSE: {svr_mse:.3f}, R^2: {svr_r2:.3f}")

    plt.scatter(y_test, lr_pred, color='blue', alpha=0.5, label="Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Ideal")
    plt.xlabel("Actual Median House Value")
    plt.ylabel("Predicted Median House Value")
    plt.title("Actual vs Predicted House Prices (Linear Regression)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
