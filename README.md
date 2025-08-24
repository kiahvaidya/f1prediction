# 🏎️ F1 Podium Prediction Using FastF1 & Machine Learning

This project uses historical F1 qualifying and race data powered by the [FastF1](https://theoehrly.github.io/Fast-F1/) library and a Random Forest Regressor to predict podium finishers for upcoming Formula 1 races.

---

## 📌 Project Overview

- Predicts top 3 finishers of a Grand Prix using the qualifying session data.
- Leverages real-time telemetry and session data from FastF1.
- Uses machine learning regression (scikit-learn RandomForestRegressor).
- Evaluates the model using standard classification metrics.

---

## 🚀 Features

- Pulls qualifying session data via FastF1
- Extracts fastest lap per driver
- Encodes categorical variables (driver, team, tyre compound, event)
- Predicts final race positions
- Outputs top 3 predicted finishers
- Trains and evaluates model using:
  - **MSE**
  - **R^2**

  
---
## 🖇 Conclusion
[Visit now!!](https://f1prediction.onrender.com/)
## 🏁 Predicted Podium for 2025 Hungarian Grand Prix

| Driver | Team              | Predicted Position |
|--------|-------------------|--------------------|
| NOR    | McLaren           | 1                  |
| PIA    | McLaren           | 2                  |
| RUS    | Mercedes          | 3                  |



