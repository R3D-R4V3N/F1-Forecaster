# 2025_f1_predictions

# 🏎️ F1 Predictions 2025 - Machine Learning Model

Welcome to the **F1 Predictions 2025** repository! This project uses **machine learning, FastF1 API data, and historical F1 race results** to predict race outcomes for the 2025 Formula 1 season.

## 🚀 Project Overview
This repository contains a **Gradient Boosting Machine Learning model** that predicts race results based on past performance, free practice sessions, qualifying times, weather data, and other structured F1 data. The model leverages:
- FastF1 API for historical race data
- 2024 race results
- 2025 qualifying session results
- Over the course of the season we will be adding additional data to improve our model as well
- Feature engineering techniques to improve predictions

## 📊 Data Sources
- **FastF1 API**: Fetches lap times, race results, and telemetry data
- **OpenF1 API**: Provides detailed session timing, pit stop and weather info
- **2025 Qualifying Data**: Used for prediction
- **Historical F1 Results**: Processed from FastF1 for training the model
- **OpenWeather Forecasts**: Used to include expected weather conditions

## 🏁 How It Works
1. **Data Collection**: The script pulls relevant F1 data using the FastF1 API.
2. **Preprocessing & Feature Engineering**: Converts lap times, normalizes driver names, and structures race data.
3. **Model Training**: A **Gradient Boosting Regressor** is trained using 2024 race results.
4. **Prediction**: The model predicts race times for 2025 and ranks drivers accordingly.
5. **Evaluation**: Model performance is measured using **Mean Absolute Error (MAE)**.

### Dependencies
Install all required packages with:
```bash
pip install -r requirements.txt
```
- `fastf1`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `flask`
- `python-dotenv`
- `ipykernel` (for running the Jupyter notebook)
- `requests` (for fetching weather forecasts)
- (optional) `openf1` if you prefer the official Python client

## File Structure
- `predict.py` – unified command line interface to run predictions.
- `webapp.py` – small Flask app for adding race data manually.
- `races.yaml` – configuration file containing driver practice and qualifying times along with weather data for each race.
- `notebooks/f1_multiyear.ipynb` – Jupyter notebook showing multi-season training and prediction.
- Legacy `prediction*.py` files are kept for reference.

Example entry in `races.yaml`:

```yaml
chinese_gp:
  weather:
    air_temp: 0
    track_temp: 0
    rainfall: 0   # set to 1 when rain is expected
    overtake_potential: 0
  drivers:
    - Driver: "Oscar Piastri"
      FP1Time_s: 0
      FP2Time_s: 0
      FP3Time_s: 0
      QualifyingTime_s: "1:30.641"
```

Drivers must be listed using their three-letter FastF1 codes
(e.g. `VER` for Max Verstappen). These codes should also be used
when entering data via the web form.

Times in `races.yaml` can be entered either as plain seconds (`90.641`) or in
`m:ss.sss` format like `1:30.641`. They will be converted automatically when
running predictions or using the web form.

The web form fetches recent weather and calculates overtake potential
automatically. If rainfall is detected the value stored is `1`.

To estimate a track's `overtake_potential` value you can run:

```bash
python3 calc_overtake_potential.py 2024 "Bahrain Grand Prix"
```

Pass multiple years separated by commas to average across seasons.

The value represents the **average grid improvement** for drivers who gained
positions during the race. A higher number therefore indicates that overtaking
is easier at that circuit. Tracks where passing is very difficult, such as
Monaco, will typically have a low value.

Times in `races.yaml` can be entered either as plain seconds (`90.641`) or in
`m:ss.sss` format like `1:30.641`. They will be converted automatically when
running predictions or using the web form.

## 🔧 Usage
1. Copy `.env.example` to `.env`. Provide your `OPENWEATHER_API_KEY` if you
   want to fetch weather data, and optionally set `FASTF1_CACHE_DIR` to change
   where FastF1 stores its cached session data (defaults to `f1_cache`).
   Session fastest lap times fetched by the web form are stored under
   `SESSION_TIMES_DIR` (defaults to `session_times`). These times are pulled
   from the OpenF1 API instead of using FastF1 so no local session data is
   required.
2. Add free practice and qualifying times for a race in `races.yaml` or run `webapp.py`
   to open a friendly form. **Select drivers from a dropdown**, pick the Grand Prix from
   the schedule and watch the race key, recent weather, rainfall flag and overtake
   potential auto-fill ☀️🌧️:
   ```bash
   python3 webapp.py
   ```
3. Run the prediction script specifying the race key **or the round number** 🔢. Use
   `--train-years` to provide a comma separated list of seasons used for training:
```bash
python3 predict.py chinese_gp --train-years 2022,2023,2024
# or simply
python3 predict.py --round 5 --year 2025
```
4. Estimate a track's overtake potential with `calc_overtake_potential.py`.
   Provide one or more seasons followed by the Grand Prix name:
```bash
python3 calc_overtake_potential.py 2023 2024 "Bahrain Grand Prix"
```
   The script prints the value for each season and the average when multiple
   years are given.

### Web Interface
Run the Flask web app to add new race data with a nicer interface 🎨:
```bash
python3 webapp.py
```
It listens on [http://localhost:5000/](http://localhost:5000/) where you can pick a Grand Prix from the season schedule,
drivers from a dropdown, and the weather fields will auto-fill based on recent data.
The schedule is fetched from the OpenF1 ``/v1/meetings`` endpoint to match the 2025
API. If no data is returned it falls back to the older ``sessions`` endpoint and
finally to FastF1 so the app keeps working when the API changes. Weather values
are averaged from the ``/v1/weather`` endpoint using the event's meeting key.

## 📈 Model Performance
The Mean Absolute Error (MAE) metric is used to evaluate how well the model predicts race times. Lower MAE values indicate more accurate predictions. During preliminary testing on the 2024 season the model achieved an MAE of about **0.15 seconds per lap**.

## 📌 Future Improvements
- Add **pit stop strategies** into the model
- Explore **deep learning** models for improved accuracy
- @mar_antaya on Instagram and TikTok will update with the latest predictions before every race of the 2025 F1 season

## 📜 License
This project is licensed under the MIT License.


🏎️ **Start predicting F1 races like a data scientist!** 🚀✨

