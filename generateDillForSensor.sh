#!/bin/bash

#!/usr/bin/env python
# vim:tabstop=2:autoindent:
#
# Authors: Chiara Bachechi, Javier Cacheiro
# Purpose: Launch the SUMO simulation for TRAFAIR
# Usage:
#     generateDillForSensor.sh <id_sensor>
#
# Changelog
#  V1: 03/11/2020 JC
#      Refactoring						   
										   															
# Command-line input parameters


ID_SENSOR="$1"
echo $ID_SENSOR

# Global variables

python3 main.py --id_sensor "$ID_SENSOR" --begin_time "2019-08-19 00:00:00" --end_time "2020-04-01 00:00:00" --feature_list "no_we-no_aux-no2_we-no2_aux-ox_we-ox_aux-co_we-co_aux-temperature-humidity" --label_list "no" --pollutant_label "no" --trainer_class_name "Calib_LSTM_FunctionTrainer_001" --trainer_module_name "calib_LSTM_FunctionTrain" --action "TrainToDBtest" --anomaly "True"															  
python3 main.py --id_sensor "$ID_SENSOR" --begin_time "2019-08-19 00:00:00" --end_time "2020-04-01 00:00:00" --feature_list "no_we-no_aux-no2_we-no2_aux-ox_we-ox_aux-co_we-co_aux-temperature-humidity" --label_list "no2" --pollutant_label "no2" --trainer_class_name "Calib_LSTM_FunctionTrainer_001" --trainer_module_name "calib_LSTM_FunctionTrain" --action "TrainToDBtest" --anomaly "True"															  
#python3 main.py --id_sensor $ID_SENSOR --begin_time "2019-08-19 00:00:00" --end_time "2020-04-01 00:00:00" --feature_list "no_we-no_aux-no2_we-no2_aux-ox_we-ox_aux-co_we-co_aux-temperature-humidity" --label_list "co" --pollutant_label "co" --trainer_class_name "Calib_LSTM_FunctionTrainer_001" --trainer_module_name "calib_LSTM_FunctionTrain" --action TrainToDBtest --anomaly True															  
python3 main.py --id_sensor "$ID_SENSOR" --begin_time "2019-08-19 00:00:00" --end_time "2020-04-01 00:00:00" --feature_list "no_we-no_aux-no2_we-no2_aux-ox_we-ox_aux-co_we-co_aux-temperature-humidity" --label_list "o3" --pollutant_label "o3" --trainer_class_name "Calib_LSTM_FunctionTrainer_001" --trainer_module_name "calib_LSTM_FunctionTrain" --action "TrainToDBtest" --anomaly "True"															  

   
   
