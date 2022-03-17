import dill
import json
import calibrationAlgorithmTrainer_interfaces
import calibrationAlgorithmFramework


def doTrain(info):
    #
    framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
    print(" -------------- calibrating for  ", str(info["id_sensor"]), " pollutant ", info["pollutant_label"] )
    framework.initFromInfo(info)
    framework.createTrainingAndTestingDB()
    print(json.dumps(info, sort_keys=True, indent=2))
    framework.initFromInfo(info)
    framework.trainCalibrator()
    print(" ----- expected labels ")
    print(str(framework.get_df_test_labels()))
    print(" ----- output ")
    print(str(framework.getCalibrator().apply_df(framework.get_df_test_features())))


def main(args=None):
    #
    #
    # example,
    #  re-create all dills
    #    for the pollutant "no"
    #    for the given sensors
    #    from the data of given range of time 
    #
    sensors=[4003, 4004, 4005]
    dates=  {
        "end": "2019-08-20 00:00:00",
        "start": "2019-08-01 00:00:00" }
    interval = "10T"
    test_size = 0.2
    for id_sensor in sensors:
        info = \
            {
                "dates": dates,
                "feat_order": ["no2_we", "no2_aux"],
                "label_list": [ "no2"],
                "id_sensor": id_sensor,
                "interval": interval,
                "pollutant_label": "no2",
                "test_size": test_size,
                "trainer_class_name": "Calib_RF_FunctionTrainer_001",
                "trainer_module_name": "calib_RF_FunctionTrain"
            }
        doTrain(info)
        #
        #
        #
        info = \
            {
                "dates": dates,
                "feat_order": ["no2_we", "no2_aux", "ox_we", "ox_aux"],
                "algorithm_parameters": {
                    "feat_order_no2": [
                        "no2_we",
                        "no2_aux"
                    ]
                },
                "label_list": [ "no2","o3"],
                "id_sensor": id_sensor,
                "interval": interval,
                "pollutant_label": "no2",
                "test_size": test_size,
                "trainer_class_name": "Calib_RF_FunctionTrainer_o3_001",
                "trainer_module_name": "calib_RF_FunctionTrain"
            }
        doTrain(info)

    
main()
