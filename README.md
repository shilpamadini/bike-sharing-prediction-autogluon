This project predicts bike rental demand based on historical data from the Kaggle Bike Sharing Demand competition.
The goal is to build an accurate regression model to forecast hourly rental counts using weather, calendar, and temporal data.
We leverage AutoGluon’s TabularPredictor for rapid model development, feature engineering, and hyperparameter optimization, progressing from a simple baseline to an advanced ensemble model.

1. Data Exploration (EDA)

Analyzed distributions and correlations between datetime, weather, and rental counts.

Identified strong temporal and seasonal patterns, including peak hours and weekday/weekend trends.

2. Baseline Model

Trained initial AutoGluon TabularPredictor with default parameters.

Baseline performance:

RMSE ≈ 1.41 (on validation set)

3. Feature Engineering

New features were derived to capture temporal and behavioral patterns:

Extracted: hour, day, month, year, weekday from datetime

Encoded: season, weather as categorical variables

Created: day_type — a composite feature combining holiday and workingday to classify each day as workingday, holiday, or weekend.

4. Feature-Enriched Model

Retrained model with engineered features.

Achieved a significantly improved RMSE ≈ 0.46 (~67% reduction in error).

5. Hyperparameter Optimization (HPO)

Performed tuning across multiple model types:

LightGBM, CatBoost, XGBoost

Used search spaces for learning rate, depth, number of leaves, regularization, and sampling parameters.

Best performing model: WeightedEnsemble_L2

RMSE ≈ 34.98

Gains were modest compared to feature engineering, suggesting feature quality had greater impact than tuning.

Results Summary
Stage	Model	Description	RMSE
1️. Baseline	Default AutoGluon fit	1.41
2️. Feature Engineered	Added datetime & categorical features	0.46
3️. Hyperparameter Tuned	LightGBM + XGBoost + CatBoost Ensemble	34.98
