# AQCalibrationFramework
A calibration framework that allows automatically calibration electrochemical air quality sensors. The models are generated in 2 ways: using csv data as input, reading directly from a postgreSQL database instance generated with the provided script.

## Generation of the models and applycation of them

    Generate dill file for calibration, training, testing and tuning
   --id_sensor The id of the sensors which is willing to be calibrated separeted by a - or all for all the available sensors.
   --begin_time, -b Insert the date and time to start the calibration from. Formatted as YYYY-MM-DD HH:MM:SS
   --end_time, -e Insert the date and time to end the calibration. Formatted as YYYY-MM-DD HH:MM:SS
   --feature_list, -l Insert the name of the pollutants separated by a -.
   --label_list Insert the name of the pollutants separated by a -.
   --pollutant_label, -p Insert the name of the pollutant to calibrate.
   --trainer_class_name Insert the name of the class that contains the definition of the trainer.
   --trainer_module_name Insert the name of the module (the file) that contains the definition of the trainer class.
   --df_csv_file_prefix The name of the csv file, set --from_csv True
   --interval_of_aggregation, -t The number of minutes to aggregate the raw data and station data.
   --test_size A number between 0 and 1 indicating the percentage of data to use to test the algorithm.
   --action The framework action to perform.
   --dill_file_name The file name of a trained calibrator.
   --info_file_name The file name of a trained calibrator.
   --algorithm_parameters A Json string with specific algorithm information.
   --do_persist_data Makes the db values persistent - it is not a dry run.
   --number_of_previous_observations The temporal window to consider with LSTM
   --id_calibration Insert sensor_calibration's row id to calibrate
