# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:45:45 2019

@author: ohadz
"""

if Pollutant_Calibration.lower()=='no':
    dataset_Calibration_10min_NA_features_list=[  'no_aux', 'no_we',] 
elif Pollutant_Calibration.lower()=='no2':
    dataset_Calibration_10min_NA_features_list=[  'no2_aux', 'no2_we',] 
elif Pollutant_Calibration.lower()=='o3':
    dataset_Calibration_10min_NA_features_list=[  'ox_aux', 'ox_we','no2_aux', 'no2_we',] 
elif Pollutant_Calibration.lower()=='co':
    dataset_Calibration_10min_NA_features_list=[  'co_aux', 'co_we',] 
