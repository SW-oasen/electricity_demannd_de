# pip install kagglehub
#import kagglehub
# Download latest version
#file_path = kagglehub.dataset_download("dsersun/europe-electricity-load-hourly-20192025", path="/data/europe-electricity-load-hourly-20192025.csv")

import pandas as pd

# Load the dataset
file_path = "./data/MHLV_2019_2025_combined.csv"
energy_data = pd.read_csv(file_path)

print(energy_data.head())





