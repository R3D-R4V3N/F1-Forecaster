# F1 Forecaster

This repository contains code for predicting the top-3 finishers in Formula 1 races using historical data from multiple sources.

The `src` directory contains a pipeline script that retrieves data from various APIs, performs feature engineering, trains an XGBoost model with time-series cross-validation, and evaluates using a custom top-3 accuracy metric.

The code is designed to be run offline once datasets are available locally.
