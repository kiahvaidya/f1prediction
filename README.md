# 🏎️ F1 Podium Prediction Using FastF1 & Machine Learning

This project uses historical F1 qualifying and race data powered by the [FastF1](https://theoehrly.github.io/Fast-F1/) library and a Gradient Boosting Classifier to predict podium finishers (P1, P2, P3) for upcoming Formula 1 races.

---

## 📌 Project Overview

- Predicts top 3 finishers of a Grand Prix using the qualifying session data.
- Leverages real-time telemetry and session data from FastF1.
- Uses machine learning classification (scikit-learn GradientBoostingClassifier).
- Evaluates the model using standard classification metrics.

---

## 🚀 Features

- Pulls qualifying session data via FastF1
- Extracts fastest lap per driver
- Encodes categorical variables (driver, team, tyre compound, event)
- Predicts final race positions
- Outputs top 3 predicted finishers
- Trains and evaluates model using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
  - **Support**
 
  
---
## 🖇 Conclusion
🏁 Predicted Podium for 2025 British Grand Prix:
     Driver             Team  PredictedPosition
   -  PIA            McLaren                  3
   -  VER    Red Bull Racing                  4
   -  STR       Aston Martin                  4


