

import sys
import os

# Add the src directory to the system path to allow importing custom modules
sys.path.insert(0, os.path.abspath('../src'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd


from fetch_prepare_data import *
from train_model_predict import *

from lightgbm import LGBMRegressor
import pandas as pd
from fetch_prepare_data import *
from train_model_predict import *

def main():
    df_for_modeling = load_model_from_pickle('../data/processed/energy_weather_data_for_modeling.pkl')
    if df_for_modeling is None:
        df_for_modeling = prepare_data_for_modeling()

    features_train, target_train, features_test, target_test = train_test_split_by_date(df_for_modeling, 
                                                                                        date_column='time',
                                                                                        target_column='EnergyDemand', 
                                                                                        split_date='2025-01-01')
    #print(f"Training features shape: {features_train.shape}, Training target shape: {target_train.shape}")
    #print(f"Testing features shape: {features_test.shape}, Testing target shape: {target_test.shape}")

    # define continuous parameter grid for LightGBM
    param_lgbm_continuous = { 
        'model__n_estimators': (50, 500),  # range for n_estimators
        'model__learning_rate': (0.01, 0.3),  # range for learning_rate
        'model__max_depth': (3, 15)  # range for max_depth
    }

    model_lgbm =LGBMRegressor(random_state=42, force_col_wise=True)


    pipeline_lgbm = init_model_pipeline(in_df=features_train, 
                                        model=model_lgbm)
    best_model_lgbm, best_params_lgbm = tune_model_bayesian(
        model_pipeline=pipeline_lgbm, 
        in_param_bayes=param_lgbm_continuous,
        in_features_train=features_train, 
        in_target_train=target_train)

    print(f"Best hyperparameters: {best_params_lgbm}")
    print()

    prediction_lgbm = best_model_lgbm.predict(features_test)

    print_scores('LightGBM', target_test, prediction_lgbm)  

    save_model_to_pickle(best_model_lgbm, '../models/best_lgbm_model_bayesian.pkl')

if __name__ == "__main__":
    main()