# Projekt: European Electricity Demand Forecasting

## Ziel

Ein Data-Science-/Data-Analyst-Portfolio-Projekt zur Vorhersage des stündlichen Stromverbrauchs in Europa, mit Fokus auf:

**Energy Analytics + Time Series + Wetter + Kalenderfeatures + Power BI Storytelling**

Primäres Ziel: **Electricity Load Forecasting**, z.B. für Deutschland oder 3–5 ausgewählte Länder.

---

## Datenquellen

### 1. Stromverbrauch

**Europe Electricity Load (Hourly, 2019–2025)**
Quelle: Kaggle, basierend auf ENTSO-E Transparency Platform. Der Datensatz enthält stündlich aggregierte Stromlast je Land von 2019 bis 2025. ENTSO-E selbst stellt historische Daten stündlich, monatlich und jährlich bereit, aggregiert nach Land. ([Kaggle][1])

Verwendung:

* `datetime`
* `country`
* `load_mw`

Lizenzhinweis:

* ENTSO-E attribution in README aufnehmen.
* CC BY-SA 4.0 beachten, falls der Kaggle-Datensatz das so angibt.

---

### 2. Historische Wetterdaten

**Open-Meteo Historical Weather API**
Open-Meteo bietet historische Wetterdaten zurück bis 1940 und ab 2017 mit neueren Wettermodellen in ca. 9 km Auflösung. Die API liefert stündliche Variablen wie Temperatur, Niederschlag, Wind und weitere Wettergrößen. ([Open Meteo][2])

`https://archive-api.open-meteo.com/v1/archive?latitude=city_lat&longitude=city_lon&start_date=2019-01-01&end_date=2025-09-30&hourly={','.join(weather_variables)}&timezone=auto`

Geplante Variablen:

* `temperature_2m`
* `apparent_temperature`
* `precipitation`
* `rain`
* `snowfall`
* `wind_speed_10m`
* `shortwave_radiation`
* optional: `cloud_cover`, `relative_humidity_2m`

Format:

* API liefert JSON
* in Pandas DataFrame umwandeln
* lokal als CSV/Parquet speichern

---

### 3. Feiertage

**python-holidays**
Die Library unterstützt länder- und subdivisionsspezifische Feiertage, also z.B. deutsche Bundesländer. 
([holidays.readthedocs.io][3])

Features:

* `is_public_holiday`
* `holiday_ratio`
* optional: `is_bridge_day`
* covid:  covid_period = 1 if 2020-03 bis 2022-03 else 0
	covid_phase =
    0 = pre-covid   # 2019
    1 = lockdown    # 2020.03 - 2020.12
    2 = recovery    # 2021, 2022

Deutschland:

* nationale Feiertage = 1.0
* regionale Feiertage = nach Bevölkerungsanteil der Bundesländer gewichten

---

### 4. Schulferien

Für Deutschland optional über eine Schulferien-API. Die Deutsche Schulferien API stellt Ferien aller 16 Bundesländer bereit; die aktuell sichtbare API nennt Jahre 2022–2028, daher reicht sie eventuell nicht vollständig für 2019–2021. ([ferien-api.maxleistner.de][4])

Empfehlung für 2 Wochen:

* **Basisversion ohne Schulferien**
* Moderate Verbesserung: `school_holiday_ratio`, falls Daten für den Zeitraum sauber verfügbar sind

---

# Projekt-Scope für 2 Wochen

## Basisversion: Muss fertig werden

Fokus auf:

* Deutschland
* Stromlast 2019–2024
* Wetteraggregation über Top-Städte 2024 (Wiki)
	Berlin:      3.69 Mio
	Hamburg:     1.86
	München:     1.51
	Köln:        1.02
	Frankfurt M: 0.76
	
* Feiertage
* Baseline + ML-Modell
* Power BI Dashboard
* sauberes README

## Moderate Verbesserung: falls Zeit bleibt

Zusätzlich:

* mehrere Länder + Frankreich, Spanien, Österreich, Italien
* gewichtete Wetteraggregation nach Stadtbevölkerung
* holiday_ratio = Länder mit Feiertag / 16
* Brückentage = Tag vor oder nach Feiertag
* Schulferienratio = = Länder mit Ferien / 16
* Modellvergleich mit Feature Importance 
* Forecast für 24h / 7 Tage

* Daten von 2025 als Validation/Test

---

# Zentrale Modellidee

## Zielvariable

```text
load_mw
```

oder besser:

```text
load_mw_next_24h
```

Praktisch für den Start:

```text
load_mw zum aktuellen Zeitpunkt vorhersagen
```

mit Lag-Features aus der Vergangenheit.

---

# Geplante Features

## Zeitfeatures

Basis:

* `hour`
* `day_of_week`
* `month`
* `year`
* `is_weekend`
* `is_workday`

Besser:

* `sin_hour`, `cos_hour`
* `sin_dayofyear`, `cos_dayofyear`

---

## Lag-Features

Basis:

* `load_lag_1h`
* `load_lag_24h`
* `load_lag_168h`

Moderate Verbesserung:

* `load_rolling_24h_mean`
* `load_rolling_168h_mean`
* `load_rolling_24h_std`

---

## Wetterfeatures

Basis:

* `temp_weighted`
* `wind_weighted`
* `precipitation_weighted`
* `shortwave_radiation_weighted`

Moderate Verbesserung:

* `temp_min`
* `temp_max`
* `temp_std`
* `heating_degree_days`
* `cooling_degree_days`

Beispiel:

```text
heating_degree = max(0, 18 - temperature)
cooling_degree = max(0, temperature - 22)
```

---

## Kalenderfeatures

Basis:

* `is_public_holiday`

Besser:

* `holiday_ratio`
* `is_bridge_day`

Optional:

* `school_holiday_ratio`

---

# Wetteraggregation

## Idee

Da Stromdaten auf Länderebene vorliegen, lokale Wetterdaten aber auf Stadt-/Koordinatenebene, werden Wetterdaten je Land aggregiert.

Empfohlen:

```text
Top 5 Städte pro Land + Gewichtung nach Bevölkerung
```

Formel:

```text
weighted_temp = sum(temp_city * population_city) / sum(population_city)
```

Für Deutschland z.B.:

* Berlin
* Hamburg
* München
* Köln
* Frankfurt am Main

README-Formulierung:

```text
Weather data was collected for representative high-population cities and aggregated using population-weighted averages to approximate country-level weather exposure.
```

---

# Python-/Notebook-Aufteilung

## Python-Skripte: wiederverwendbare Pipeline

```text
src/
├── config.py
├── load_energy.py
├── fetch_weather.py
├── aggregate_weather.py
├── calendar_features.py
├── feature_engineering.py
├── train_model.py
└── evaluate.py
```

### `config.py`

Enthält:

* Länder
* Städte
* Koordinaten
* Bevölkerungsgewichte
* Zeitraum
* Wettervariablen
* Pfade

---

### `load_energy.py`

Aufgaben:

* Kaggle/CSV-Daten laden
* Spalten vereinheitlichen
* Datetime parsen
* Länder filtern
* Zeitzone prüfen
* Rohdaten speichern

Output:

```text
data/processed/energy_clean.parquet
```

---

### `fetch_weather.py`

Aufgaben:

* Open-Meteo API pro Stadt aufrufen
* JSON → DataFrame
* Rohdaten pro Stadt speichern

Output:

```text
data/raw/weather/{country}_{city}.parquet
```

---

### `aggregate_weather.py`

Aufgaben:

* Wetter pro Stadt laden
* nach Bevölkerung gewichten
* je Land und Stunde aggregieren

Output:

```text
data/processed/weather_country_hourly.parquet
```

---

### `calendar_features.py`

Aufgaben:

* Feiertage generieren
* Feiertagsratio berechnen
* optional Brückentage
* optional Schulferienratio

Output:

```text
data/processed/calendar_features.parquet
```

---

### `feature_engineering.py`

Aufgaben:

* Energy + Weather + Calendar mergen
* Zeitfeatures erzeugen
* Lag-/Rolling-Features erzeugen
* finale Modellmatrix speichern

Output:

```text
data/processed/model_dataset.parquet
```

---

### `train_model.py`

Aufgaben:

* Train/Validation/Test Split zeitbasiert
* Baseline-Modell
* ML-Modell
* Modell speichern

Modelle:

* Baseline: Seasonal Naive
* ML: RandomForest oder HistGradientBoosting
* Optional: XGBoost / LightGBM

---

### `evaluate.py`

Metriken:

* MAE
* RMSE
* MAPE oder sMAPE

Visuals:

* Actual vs Predicted
* Error by hour
* Error by month
* Feature Importance

---

# Notebook-Aufteilung

## `01_data_understanding.ipynb`

Ziel:

* Datensatz verstehen
* Länder, Zeiträume, fehlende Werte prüfen
* erste Zeitreihenplots

Inhalt:

* Load nach Jahr/Monat
* Tagesprofile
* Wochenprofile
* Peak Load Analyse

---

## `02_weather_calendar_features.ipynb`

Ziel:

* Wetter- und Kalenderfeatures validieren

Inhalt:

* Wetterdaten je Land prüfen
* Temperatur vs Load
* Feiertage vs Load
* Wochenenden vs Werktage
* Feature-Korrelationen

---

## `03_modeling_forecasting.ipynb`

Ziel:

* Forecasting-Modell bauen und bewerten

Inhalt:

* Time-based Split
* Baseline
* ML-Modell
* Modellvergleich
* Feature Importance
* Fehleranalyse

---

## `04_business_insights_dashboard_prep.ipynb`

Ziel:

* Export für Power BI vorbereiten

Output:

```text
dashboard/powerbi_energy_forecast_export.csv
```

Enthält:

* Datum/Zeit
* Land
* tatsächliche Last
* Vorhersage
* Fehler
* Wetterfeatures
* Kalenderfeatures

---

# Power BI Dashboard

## Seite 1: Executive Overview

KPIs:

* Average Load
* Peak Load
* Forecast MAE
* Forecast RMSE
* MAPE/sMAPE

Visuals:

* Actual vs Predicted
* Load by Country
* Monthly Load Trend

---

## Seite 2: Weather Impact

Visuals:

* Load vs Temperature
* Load vs Shortwave Radiation
* Load vs Wind
* Heating/Cooling Degree Effekt

---

## Seite 3: Calendar Impact

Visuals:

* Werktag vs Wochenende
* Feiertag vs Nicht-Feiertag
* Durchschnittliches Tagesprofil
* optional Schulferienratio

---

## Seite 4: Model Performance

Visuals:

* Error by Hour
* Error by Month
* Biggest Forecast Errors
* Feature Importance


## Links

[1]: https://www.kaggle.com/datasets/dsersun/europe-electricity-load-hourly-20192025 
Europe Electricity Load (Hourly, 2019–2025)

[2]: https://open-meteo.com/en/docs/historical-weather-api
Historical Weather API

[3]: https://holidays.readthedocs.io/ 
holidays - Read the Docs

[4]: https://ferien-api.maxleistner.de/ 
Deutsche Schulferien API