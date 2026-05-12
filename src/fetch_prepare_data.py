
# --------- fetch the data from Kaggle and save it to the raw data folder ----------

import shutil
import os
from typing import Literal

dataset_link = "dsersun/europe-electricity-load-hourly-20192025"  # just the owner/dataset part
destination = "../data/raw"
def fetch_kaggle_dataset(in_dataset_link=dataset_link, in_destination=destination):
    '''
    Fetch the dataset from Kaggle and save it to the raw data folder.
    '''
    import kagglehub  # pip install kagglehub
    cache_path = kagglehub.dataset_download(in_dataset_link)
    #print(f"Downloaded to cache: {cache_path}")

    # Copy all files from cache to destination
    for file in os.listdir(cache_path):
        shutil.copy(os.path.join(cache_path, file), in_destination)
        print(f"Copied: {file} → {in_destination}")


# --------- read the combined energy and weather data ----------

import pandas as pd

orig_file_path = "../data/raw/MHLV_2019_2025_combined.csv"
processed_file_path = "../data/processed/energy_weather_2019_2025.csv"


def read_energy_weather_data(file_path):
    df = pd.read_csv(file_path)
    return df

def read_energy_data_from_kaggle_file(file_path=orig_file_path):
    df_energy = pd.read_csv(file_path, parse_dates=['DateUTC'])
    start_date = df_energy['DateUTC'].min().strftime("%Y-%m-%d")
    end_date = df_energy['DateUTC'].max().strftime("%Y-%m-%d")

    df_energy_de = df_energy[df_energy['CountryCode'] == 'DE']  # filter for Germany, since we want to predict German energy demand
    df_energy_de = df_energy_de.drop(columns=[col for col in df_energy_de.columns   if col not in ['DateUTC', 'Value']], errors='ignore')  # keep only the relevant columns, ignore if they are not present
    
    df_energy_de = rename_time_column(df_energy_de)
    df_energy_de = create_time_based_features(df_energy_de, in_year=df_energy_de['time'].dt.year.max())
    df_energy_de = create_energy_features(df_energy_de)
    return df_energy_de, start_date, end_date


def retrieve_train_test_data_from_csv(file_path=processed_file_path):
    df = pd.read_csv(file_path)
    train_data = df[df['time'] < '2025-01-01']
    test_data = df[df['time'] >= '2025-01-01']

    out_features_train = train_data.drop(['time', 'EnergyDemand'], axis=1)
    out_target_train = train_data['EnergyDemand']
    out_features_test = test_data.drop(['time', 'EnergyDemand'], axis=1)
    out_target_test = test_data['EnergyDemand']
    
    return out_features_train, out_target_train, out_features_test, out_target_test


# --------- scrape SMARD data ----------

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

SMARD_BASE = "https://www.smard.de/app/chart_data"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; smard-fetcher/1.0)"}

# Filter IDs
FILTER_NETZLAST = 410          # Realisierter Stromverbrauch – Netzlast


def _get_index(filter_id: int, region: str = "DE", resolution: str = "hour") -> list[int]:
    """Return the list of weekly bucket timestamps (Unix ms) available for the given filter."""
    url = f"{SMARD_BASE}/{filter_id}/{region}/index_{resolution}.json"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()["timestamps"]


def _fetch_week(filter_id: int, timestamp_ms: int, region: str = "DE", resolution: str = "hour") -> list:
    """Fetch the raw series [[ts_ms, value], ...] for one weekly bucket."""
    url = f"{SMARD_BASE}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp_ms}.json"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json().get("series", [])


def fetch_smard_netzlast(
    in_start_date: str,
    in_end_date: str,
    output_file: str | None = None,
    region: str = "DE",
    resolution: str = "hour",
    filter_id: int = FILTER_NETZLAST,
    sleep: float = 0.3
) -> pd.DataFrame:
    """
    Fetch Realisierter Stromverbrauch (Netzlast) from the SMARD chart_data API.

    Parameters
    ----------
    in_start_date : str
        Inclusive start in 'YYYY-MM-DD' format (local CET/CEST time).
    in_end_date : str
        Inclusive end in 'YYYY-MM-DD' format.
    output_file : str | None
        If given, save the result as CSV to this path.
    region : str
        SMARD region code, default 'DE'.
    resolution : str
        'hour' or 'quarterhour'.
    filter_id : int
        SMARD filter ID (410 = Netzlast).
    sleep : float
        Seconds to sleep between requests (be polite to the server).

    Returns
    -------
    pd.DataFrame with columns ['timestamp', 'load_MWh'].
    """
    # Convert date strings to UTC millisecond boundaries
    start_dt = datetime.strptime(in_start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(in_end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    # Include the full end day
    end_ms   = int(end_dt.timestamp() * 1000) + 86_400_000 - 1
    start_ms = int(start_dt.timestamp() * 1000)

    #print(f"Fetching index for filter {filter_id} / {region} / {resolution} ...")
    all_timestamps = _get_index(filter_id, region, resolution)

    # Keep only buckets that can overlap with [start_ms, end_ms].
    # Each bucket covers roughly one week (604_800_000 ms).
    week_ms = 7 * 24 * 3600 * 1000
    relevant = [ts for ts in all_timestamps if ts <= end_ms and ts + week_ms >= start_ms]

    if not relevant:
        print("No data available for the requested period.")
        return pd.DataFrame(columns=["timestamp", "load_MWh"])

    #print(f"Fetching {len(relevant)} weekly bucket(s) ...")
    rows = []
    for ts in relevant:
        series = _fetch_week(filter_id, ts, region, resolution)
        rows.extend(series)
        time.sleep(sleep)

    # Build DataFrame and clip to the exact requested range
    df = pd.DataFrame(rows, columns=["ts_ms", "load_MWh"])
    df = df.dropna(subset=["load_MWh"])
    df = df[(df["ts_ms"] >= start_ms) & (df["ts_ms"] <= end_ms)]
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert("Europe/Berlin")
    df = df[["timestamp", "load_MWh"]].sort_values("timestamp").reset_index(drop=True)

    #print(f"Retrieved {len(df)} rows from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        df.to_csv(output_file, index=False)
        #print(f"Saved to {output_file}")

    return df

# --------- add time based features for the energy demand data ----------

# rename the time column to 'time' for consistency across datasets
def rename_time_column(in_df):  
    known_time_cols = ('time', 'timestamp', 'DateUTC')
    for col in known_time_cols:
        if col in in_df.columns and col != 'time':
            in_df = in_df.rename(columns={col: 'time'})
            break
    return in_df

# add holiday ratio depending the number of states in Germany with a holiday on that day
import holidays 

def holiday_ratio(date):
    count = sum([1 for state in holidays.Germany(years=[date.year]).items() if state[0] == date])
    return count / 16

def create_time_based_features(in_df, in_year, time_column='time'):
    '''
    Create time-based features such as hour of day, day of week, and month of year.
    '''
    out_df = in_df.copy()
    out_df['year'] = out_df[time_column].dt.year
    out_df['hour'] = out_df[time_column].dt.hour
    out_df['weekday'] = out_df[time_column].dt.dayofweek
    out_df['month'] = out_df[time_column].dt.month
    out_df['is_weekend'] = out_df[time_column].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    de_holidays = holidays.Germany(years=range(2019, in_year + 1))
    out_df['is_holiday'] = out_df[time_column].dt.date.apply(lambda x: 1 if x in de_holidays else 0)
    out_df['holiday_ratio'] = out_df[time_column].dt.date.apply(holiday_ratio)

    return out_df

def create_energy_features(in_df):
    out_df = in_df.copy()   

    known_load_cols = ('EnergyDemand', 'load_MWh', 'Value')
    for col in known_load_cols:
        if col in out_df.columns and col != 'EnergyDemand':
            out_df = out_df.rename(columns={col: 'EnergyDemand'})
            break

    # add lagged features for energy demand (shifted by 24 hours, 168 hours (1 week) to capture daily, weekly, and yearly patterns)
    out_df['EnergyDemand_lag_24h'] = out_df['EnergyDemand'].shift(24)   # 1 day
    out_df['EnergyDemand_lag_168h'] = out_df['EnergyDemand'].shift(168)   # 1 week
    # out_df['EnergyDemand_lag_8760h'] = out_df['EnergyDemand'].shift(8760) # 1 year, removed after feature importance analysis showed it was not useful

    # rolling mean of past demand (shift first to avoid leakage)
    out_df['EnergyDemand_rolling_mean_24h'] = out_df['EnergyDemand'].shift(1).rolling(24).mean()   # daily pattern
    out_df['EnergyDemand_rolling_mean_168h'] = out_df['EnergyDemand'].shift(1).rolling(168).mean() # weekly pattern
    # out_df['EnergyDemand_rolling_mean_8760h'] = out_df['EnergyDemand'].shift(1).rolling(8760).mean() # yearly pattern, removed after feature importance analysis showed it was not useful

    # drop rows with missing values
    out_df = out_df.dropna()

    return out_df

# =========== prepare energy data =============

def prepare_energy_data(
        purpose: Literal['modeling', 'prediction'] = 'modeling', 
        in_start_date=None, 
        in_end_date=None):
    '''
    Prepare energy data for modeling: fetch the energy data from SMARD, create time-based features, and return the prepared DataFrame.
    '''
    if purpose == 'modeling':
        df_energy, start_date, end_date = read_energy_data_from_kaggle_file()
        df_energy = rename_time_column(df_energy)
        df_energy = create_time_based_features(df_energy, in_year=pd.to_datetime(start_date).year)
        df_energy = create_energy_features(df_energy)

    elif purpose == 'prediction':
        # for prediction, we need to fetch the most recent energy data to create lagged features, since the Kaggle dataset only goes up to 2025-09-30
        df_energy = fetch_smard_netzlast(in_start_date, in_end_date)
        df_energy = rename_time_column(df_energy)
        df_energy = create_energy_features(df_energy)
        
        # create time hourly of in_end_day+1 features for the future prediction date, since the lagged features will be based on the past 24h and 168h of energy demand, we need to have at least 168h of energy data up to the day before the prediction date
        time_24h_after_end_date = pd.to_datetime(in_end_date) + pd.Timedelta(hours=24)
        df_energy['time'] = time_24h_after_end_date
        df_energy = create_time_based_features(df_energy, in_year=pd.to_datetime(start_date).year)
        start_date, end_date = in_start_date, in_end_date  # for consistency with the modeling case, we return the start and end date of the energy data we have, which can be used to fetch the corresponding weather data for the same period

    else:
        raise ValueError("Invalid purpose. Must be 'modeling' or 'prediction'.")

    return df_energy, start_date, end_date


# ----------- fetch weather data from open-meteo  ----------

import time
import requests
import pandas as pd 

# apparent_temperature is the perceived temperature, which takes into account factors such as humidity and wind speed to provide a more accurate representation of how the temperature feels to humans. It is calculated using a formula that combines the actual air temperature with the effects of humidity and wind chill. The apparent temperature can be higher than the actual temperature in hot and humid conditions, and lower than the actual temperature in cold and windy conditions.
# precipitation is the amount of water that falls from the atmosphere to the ground in the form of rain, snow, sleet, or hail. It is typically measured in millimeters (mm) or inches (in) and can be used to assess the amount of moisture in the air and the likelihood of certain weather conditions, such as flooding or drought.
# shortwave_radiation is the amount of solar radiation that reaches the Earth's surface in the form of shortwave electromagnetic waves. It is typically measured in watts per square meter (W/m²) and can be used to assess the amount of energy available for photosynthesis, as well as the potential for solar power generation.
weather_variables = ['temperature_2m', 'apparent_temperature', 'rain', 'snowfall', 'wind_speed_10m', 'shortwave_radiation']

# get latitude and longitude of German cities: Berlin, Hamburg, München, Köln, Frankfurt
selected_cities = {  
    'Berlin': {'latitude': 52.5200, 'longitude': 13.4050},
    'Hamburg': {'latitude': 53.5511, 'longitude': 9.9937},
    'München': {'latitude': 48.1351, 'longitude': 11.5820},
    'Köln': {'latitude': 50.9375, 'longitude': 6.9603},
    'Frankfurt': {'latitude': 50.1109, 'longitude': 8.6821}
}

start_date = "2019-01-01" # Kaggle dataset starts from 2019-01-01
end_date = "2025-09-30" # Kaggle dataset ends at 2025-09-30

def fetch_weather_data_for_cities(in_selected_cities=selected_cities, 
                                  in_start_date=start_date, 
                                  in_end_date=end_date, 
                                  in_weather_variables=weather_variables):
    '''
    Fetch weather data from open-meteo archive API for the selected cities and return a dictionary of city name to weather DataFrame.
    '''
    weather_city_dict = {}
    for city, coords in in_selected_cities.items():
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={coords['latitude']}&longitude={coords['longitude']}&start_date={in_start_date}&end_date={in_end_date}&hourly={','.join(in_weather_variables)}&timezone=auto"
        response = requests.get(url)
        weather_data = response.json()
        df_weather_city = pd.DataFrame(weather_data['hourly'])
        df_weather_city['time'] = pd.to_datetime(df_weather_city['time'], utc=True).dt.tz_convert("Europe/Berlin")
        #print(f"weather for {city}: {len(df_weather_city)} rows")
        #print(df_weather_city.head(3))
        weather_city_dict.update({city:df_weather_city})
        time.sleep(1)  # sleep for 1 second to avoid hitting API rate limits
    return weather_city_dict

city_population = {
    'Berlin': 3644826, 
    'Hamburg': 1841179, 
    'München': 1471508,     
    'Köln': 1085664, 
    'Frankfurt': 753056 
}

raw_tmp_path = "../data/raw/tmp/"

# calculate the weight of the cities based on their population size and use it to create a weighted average of the weather variables for Germany
def merge_weather_data_with_city_weights(in_weather_city_dict, 
                                         in_city_population=city_population, 
                                         in_weather_variables=weather_variables):
    '''
    Merge the weather data for the selected cities into a single DataFrame for Germany, 
    using population weights to calculate a weighted average of the weather variables.
    '''
    total_population = sum(in_city_population.values())
    df_weather_germany = pd.DataFrame() 
    for city, df_city in in_weather_city_dict.items():
        weight = in_city_population[city] / total_population
        df_city_weighted = df_city.copy()
        for var in in_weather_variables:
            df_city_weighted[var] = df_city[var] * weight
        if df_weather_germany.empty:
            df_weather_germany = df_city_weighted
        else:
            df_weather_germany[in_weather_variables] += df_city_weighted[in_weather_variables]

    return df_weather_germany

# feature engineering: create new features based on existing ones, such as rolling averages, lagged variables, or interaction terms

base_temperature_heating = 15  # base temperature for heating degree days
base_temperature_cooling = 25  # base temperature for cooling degree days

pandemic_start = pd.to_datetime('2020-03-01', utc=True)
pandemic_end = pd.to_datetime('2021-12-31', utc=True)

def create_weather_features(in_df, 
                    in_base_temperature_heating=base_temperature_heating, 
                    in_base_temperature_cooling=base_temperature_cooling, 
                    in_pandemic_start=pandemic_start, 
                    in_pandemic_end=pandemic_end, 
                    time_column='time'):
    '''
    Create new features based on existing ones, such as rolling averages, lagged variables, or interaction terms.
    '''
    out_df = in_df.copy()

    # add rolling average and lagged variable for apparent_temperature
    out_df['apparent_temperature_rolling_mean_24h'] = out_df['apparent_temperature'].shift(1).rolling(window=24).mean()
    out_df['apparent_temperature_lag_24h'] = out_df['apparent_temperature'].shift(24)

    # add rolling average and lagged varirable for shortwave_radiation_0m
    out_df['shortwave_radiation_0m_rolling_mean_24h'] = out_df['shortwave_radiation'].shift(1).rolling(window=24).mean()
    out_df['shortwave_radiation_0m_lag_24h'] =   out_df['shortwave_radiation'].shift(24)

    # add heating degree days (HDD) and cooling degree days (CDD) features
    out_df['heating_degree'] = out_df['apparent_temperature'].apply(lambda x: max(0, in_base_temperature_heating - x))  # HDD is calculated as the difference between a base temperature (e.g., 18°C) and the actual temperature, but only if the actual temperature is below the base temperature
    out_df['cooling_degree'] = out_df['apparent_temperature'].apply(lambda x: max(0, x - in_base_temperature_cooling))  # CDD is calculated as the difference between the actual temperature and a base temperature (e.g., 25°C), but only if the actual temperature is above the base temperature

    # add pandemic feature
    out_df['is_pandemic_time'] = out_df[time_column].apply(lambda x: 1 if (x >= in_pandemic_start) and (x <= in_pandemic_end) else 0)

    return out_df

# ============ prepare weather data ============

# fetch weather data for the selected cities, merge it with population weights to get a Germany-wide weather dataset, and save it to the processed data folder
def prepare_weather_data(in_start_date = start_date, 
                        in_end_date = end_date, 
                        in_selected_cities=selected_cities,
                        in_weather_variables=weather_variables, 
                        in_city_population=city_population, 
                        in_file_path: str | None = None):
    '''
    Prepare weather data for modeling: fetch weather data for the selected cities, 
    merge it with population weights to get a Germany-wide weather dataset, and save it to the processed data folder.
    '''
    weather_city_dict = fetch_weather_data_for_cities(in_selected_cities, in_start_date, in_end_date, in_weather_variables)
    df_weather_germany = merge_weather_data_with_city_weights(weather_city_dict, in_city_population, in_weather_variables)
    df_weather_germany = rename_time_column(df_weather_germany)
    df_weather_germany = create_weather_features(df_weather_germany)
    if in_file_path:
        out_file_path = f"{in_file_path}weather_germany.csv"
        df_weather_germany.to_csv(out_file_path, index=False)
        print(f"Germany-wide weather dataset saved to: {out_file_path}")
    return df_weather_germany


# ---------- comnbine energy and weather dataset for modeling ----------

processed_file_path = "../data/processed/"
def combine_energy_weather_dataset(in_energy_df, 
                                    in_weather_df, 
                                    in_file_path: str | None = None):
    '''
    Prepare the combined energy and weather dataset for modeling: merge the energy and weather datasets on the timestamp, 
    drop columns with high correlation, and save the combined dataset to the processed data folder.
    '''
    df_combined = pd.DataFrame()

    in_energy_df['time'] = pd.to_datetime(in_energy_df['time'], utc=True).dt.tz_convert("Europe/Berlin")
    in_weather_df['time'] = pd.to_datetime(in_weather_df['time']).dt.tz_convert("Europe/Berlin")
    df_combined = pd.merge(in_energy_df, in_weather_df, on='time', how='inner')
    
    if in_file_path:
        out_file_path = f"{in_file_path}energy_weather_germany.csv"
        df_combined.to_csv(out_file_path, index=False)
        print(f"Combined dataset saved to: {out_file_path}")

    return df_combined

# =========== prepare the combined energy and weather dataset for modeling ============

def prepare_data_for_modeling(in_file_path=orig_file_path):
    '''
    Prepare the combined energy and weather dataset for modeling: merge the energy and weather datasets on the timestamp, 
    drop columns with high correlation, and save the combined dataset to the processed data folder.
    '''
    df_energy, start_date, end_date = prepare_energy_data(purpose='modeling')
    df_weather = prepare_weather_data(in_start_date=start_date, in_end_date=end_date)
    df_combined = combine_energy_weather_dataset(df_energy, df_weather)
    return df_combined

# ---------- prepare future features for prediction ----------

def get_start_end_date(prediction_date):
    #prediction_date = "2026-05-07"
    # Need at least 15 days of history so that the 168h lag/rolling features have enough rows
    start_date = (pd.to_datetime(prediction_date) - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    end_date = (pd.to_datetime(prediction_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return start_date, end_date 

# ============= prepare future features for prediction ============ 
# create a DataFrame with the same structure as the training data, but with future timestamps and corresponding weather features, which can be used for making predictions for future energy demand
def prepare_future_features(prediction_date):
    '''
    Prepare future features for prediction: create a DataFrame with the same structure as the training data, but with future timestamps and corresponding weather features, which can be used for making predictions for future energy demand.
    '''
    start_date, end_date = get_start_end_date(prediction_date)
    df_energy = prepare_energy_data(purpose='prediction', in_start_date=start_date, in_end_date=end_date)
    df_weather = prepare_weather_data(in_start_date=start_date, in_end_date=end_date)
    df_future = combine_energy_weather_dataset(in_energy_df=df_energy, in_weather_df=df_weather)
    df_future = df_future.drop(columns=['EnergyDemand'])  # drop the target variable, since we want to predict it
    df_future = df_future.dropna()  # drop rows with missing values, which can occur due to lagged features
    return df_future