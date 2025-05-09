# 🏠 Predykcja cen nieruchomości – regresja z użyciem scikit-learn i XGBoost

Projekt polega na zbudowaniu i porównaniu kilku modeli regresyjnych do przewidywania cen domów na podstawie danych tabelarycznych. Został zrealizowany w ramach nauki regresji w `scikit-learn`, jako część zajęć akademickich z analizy danych.

---

## 🧰 Użyte technologie

- Python 3.10+
- pandas, numpy
- scikit-learn
- XGBoost
- matplotlib, seaborn

---

## 📈 Co zawiera projekt?

- 🔍 Eksploracyjna analiza danych (EDA)
- 🧼 Czyszczenie danych i imputacja braków
- 🎛️ One-hot encoding i skalowanie zmiennych
- 📊 Trening modeli:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - XGBoost
- 📉 Ocena jakości modeli: MAE, RMSE, R²
- 🧠 Interpretacja ważności cech (feature importance)
- 🖼️ Wizualizacja wyników predykcji i wykresów pomocniczych

---

## 🗂️ Struktura projektu
```
main
│
├── data/ # Dane źródłowe
│ └── houses.csv
│
├── notebooks/ # Analiza krok po kroku
│ └── houses.ipynb
│
├── src/ # Pipeline kodowy (do uruchomienia bez notebooka)
│ └── train_models.py
│
├── outputs/ # Wygenerowane wykresy i wyniki
│ ├── XGBoost_feature_importance.png
│ ├── model_predictions_comparison.png
│ 
│
├── requirements.txt # Lista bibliotek
├── README.md # Opis projektu
```

---

## 🚀 Uruchomienie projektu

1. Upewnij się, że masz zainstalowane wymagane biblioteki:

```bash
pip install -r requirements.txt
