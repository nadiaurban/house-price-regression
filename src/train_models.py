#Regresja na danych o cenach nieruchomości - preprocessing, eksploracja i modele 

import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
import numpy as np 
from xgboost import XGBRegressor

houses = pd.read_csv('houses.csv')

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

num_features = houses.select_dtypes(include=['int64', 'float64']).columns.to_list()
cat_features = houses.select_dtypes(include=['object']).columns.to_list()

houses[num_features] = num_imputer.fit_transform(houses[num_features])
houses[cat_features] = cat_imputer.fit_transform(houses[cat_features])

Q1 = houses['Cena'].quantile(0.25)
Q3 = houses['Cena'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

houses[(houses['Cena'] <= lower_bound) | (houses['Cena'] >= upper_bound)]
houses = houses[(houses['Cena'] >= lower_bound) & (houses['Cena'] <= upper_bound)]

# One-hot encoding dla cech kategorycznych
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = pd.DataFrame(encoder.fit_transform(houses[cat_features]))
encoded.index = houses.index 

encoded.columns = encoder.get_feature_names_out(cat_features)

houses = houses.drop(cat_features, axis=1).join(encoded)

#train test split
X = houses.drop(columns=["Cena"])
y = houses["Cena"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#skalowanie

scaler = StandardScaler()
num_features.remove("Cena")

X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])


# Lista modeli do przetestowania
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, subsample=0.8)
}

# Słowniki do przechowywania wyników
metrics = {"Model": [], "MAE": [], "RMSE": [], "R² Score": []}
predictions = {name: None for name in models}
coefficients = {name: None for name in models}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    
    metrics["Model"].append(name)
    metrics["MAE"].append(mean_absolute_error(y_test, y_pred))
    metrics["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics["R² Score"].append(r2_score(y_test, y_pred))

    if hasattr(model, 'coef_'):
        coefficients[name] = model.coef_

results_df = pd.DataFrame(metrics)
importances = models["XGBoost"].feature_importances_
# 1. Zrób Series i posortuj malejąco
sorted_importances = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

# 2. Wykres
plt.figure(figsize=(16,10))
plt.bar(range(len(sorted_importances)), sorted_importances)
plt.xticks(range(len(sorted_importances)), sorted_importances.index, rotation=42, ha="right")
plt.title('Ważność cech w modelu XGBoost (posortowana)')
plt.xlabel("Cechy")
plt.ylabel("Ważność")
plt.tight_layout()
plt.savefig('XGBoost_feature_importance.png')  # Zapisz wykres
plt.show()

def plot_model_vs_predictions(ax, model_name, y_true, y_pred):
    ax.scatter(y_true, y_pred, color="purple", alpha=0.3)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="black",linestyle='--')
    ax.set_xlabel('Rzeczywiste ceny')
    ax.set_ylabel('Przewidywane ceny')
    ax.set_title(f'{model_name} – Rzeczywiste vs. Przewidywane')
    ax.grid(True)
    
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

plot_model_vs_predictions(axes[0, 0], "Linear Regression", y_test, predictions["Linear Regression"])
plot_model_vs_predictions(axes[0, 1], "Ridge Regression", y_test, predictions["Ridge Regression"])
plot_model_vs_predictions(axes[1, 0], "Lasso Regression", y_test, predictions["Lasso Regression"])
plot_model_vs_predictions(axes[1, 1], "XGBoost", y_test, predictions["XGBoost"])
plt.tight_layout()
plt.savefig('model_predictions_comparison.png')  # Zapisz wykres
plt.show()