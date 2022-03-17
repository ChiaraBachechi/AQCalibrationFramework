
from calibrationAlgorithmTrainer_interfaces import *


class CalibrationAlgorithmTrainer_dummy(CalibartionAlgorithmTrainer_interface):
    """
    A trainer example
    """
    def init(self,info_dictionary):
        self.info_dictionary=info_dictionary
        self.calibrator=None

    def getCalibrator(self):
        return self.calibrator

    def doTrain(self,df_train_features, df_train_labels):
        print("Hello! I am training...")
        self.calibrator = CalibrationAlgorithm_dummy()
        self.calibrator.init(self.info_dictionary)
        return self.calibrator


class CalibrationAlgorithm_dummy(CalibartionAlgorithm_interface):
    """
    the calibrator produced by the trainer - example
    """
    def init(self,feature_list):
        super().init(feature_list)
        info=self.info_dictionary
        info["python_env"]={}
        info["features"]={}
        info["label"]={}
        info["hyper_parameters"]={}

    def apply_df(self, data_frame_in):
        return None
    #
    #
    @abstractfunc
    def get_info(self):
        return(self.info_dictionary)
