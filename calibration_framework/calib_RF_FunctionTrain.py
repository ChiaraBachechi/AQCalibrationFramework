import pandas as pd
import dill
import numpy as np
import psycopg2
import sklearn
import operator

import json
import datetime # this is to pring the current time during the run

from calibrationAlgorithmTrainer_interfaces import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import scipy
import scipy.stats as st




class Calib_RF_FunctionTrainer_001(CalibartionAlgorithmTrainer_interface):
    """
    A trainer example

   cat <<EOF > Calib_RF_FunctionTrainer_001.json
{
  "dates": {
    "end": "2019-08-20 00:00:00",
    "start": "2019-08-01 00:00:00"
  },
  "feat_order": [
    "no2_we",
    "no2_aux"
  ],
  "label_list": [
    "no2"
  ],
  "id_sensor": "4003",
  "interval": "10T",
  "pollutant_label": "no2",
  "test_size": 0.2,
  "trainer_class_name": "Calib_RF_FunctionTrainer_001",
  "trainer_module_name": "calib_RF_FunctionTrain",
  "units_of_measure": {
    "no2": {
      "conversions": [
        {
          "factor": 1.912,
          "from": "ppb"
        }
      ],
      "unit_of_measure": "ug/m^3"
    }
  }
}
EOF
       python3 main.py \
         --action trainAndSaveDillToFileFromInfoAndDf \
         --info_file_name      Calib_RF_FunctionTrainer_001.json \
         --dill_file_name      calibrator001.dill \
         --df_csv_file_prefix "data/calibrator001"
       python3 main.py \
         --action getInfoFromDillFile \
         --dill_file_name      calibrator001.dill \
    """
    def init(self,info_dictionary):
        self.info_dictionary=info_dictionary
        self.calibrator=None
    def getCalibrator(self):
        return self.calibrator
    def doTrain(self,df_train_features, df_train_labels_full):
        updateInfoData(self.info_dictionary,df_train_features, df_train_labels_full)
        pollutant_label=self.info_dictionary['pollutant_label']
        df_train_labels=df_train_labels_full[pollutant_label]
        #
        #
        #print('conversion_factor_for :'+self.info_dictionary['pollutant_label'])
        #print(" --- conversion_factor_for:\n",
        #    json.dumps(self.info_dictionary['units_of_measure'][self.info_dictionary['pollutant_label']]))
        factor=self.info_dictionary['units_of_measure'][pollutant_label]['conversions'][0]['factor']
        print (" --- factor: " + str(factor))
        #
        #
        # rf-training
        #
        #
        # n_jobs means the CPU. -1 is all :)
        d=self.info_dictionary['algorithm_parameters']['hyper_parameters']
        trees=d['trees']
        leaves=d['leaves']
        rf = general_rf_calibration(trees, leaves, df_train_features, df_train_labels)
        #
        #
        # linear-training
        #
        #
        # reg_lin_temp  set of trained linear algorithms - we search for the best fitting one
        reg_lin_set={}
        #
        # list of the statistics to check the best one
        reg_lin_set_stats={}
        for feat_name in self.info_dictionary['feat_order']:
            dataset_linear=np.array([df_train_features[feat_name]])
            dataset_linear=np.transpose(dataset_linear)
            #
            reg_lin_set[feat_name]=\
                LinearRegression()\
                .fit(dataset_linear, np.array(df_train_labels))
            reg_lin_set_stats[feat_name]=\
                st.spearmanr(reg_lin_set[feat_name].predict(dataset_linear)\
                             ,df_train_labels)[0]
        best_feat_name=max(reg_lin_set_stats.items(), key=operator.itemgetter(1))[0]
        reg_lin=reg_lin_set[best_feat_name]
        #
        #
        #
        self.calibrator = Calib_RF_Function_001()
        self.calibrator.init(self.info_dictionary, rf, best_feat_name, reg_lin)
        return self.calibrator



class Calib_RF_Function_001(CalibartionAlgorithm_interface):
    """
    the calibrator produced by the trainer - example
    """
    def init(self,info_dictionary, rf, best_feat_name, reg_lin):
        super().init(info_dictionary)
        #
        self.rf = rf
        self.best_feat_name = best_feat_name
        self.reg_lin = reg_lin
        #
        maxvalues={}
        for feature_name in info_dictionary['feat_order']:
            maxvalues[feature_name]=float(info_dictionary['features'][feature_name]['range'][1])
        self.maxvalues = maxvalues

    def apply_df(self, data_frame_in):
        # print(" ---- data_frame_in: " + str(data_frame_in))
        dataset_feature=data_frame_in[self.info_dictionary['feat_order']]
        dataset_na=dataset_feature.dropna()
        #need to add a test for outside of the bounds. they should be percent
        if dataset_na.empty:
            return np.nan
        else:
            #this apply doesn't work for o3 if it doesn't contain NO2 raw features in it's calibration
            #for NO, NO2, and CO the RF results are in ug/m3 directly.
            dataset_for_extrapolation=dataset_na.copy()
            for key in self.maxvalues:
                dataset_for_extrapolation=dataset_for_extrapolation[dataset_for_extrapolation[key]>self.maxvalues[key]]
            dataset_for_rf=dataset_na[~dataset_na.index.isin(dataset_for_extrapolation.index)]
            print('dataset_for_extrapolation')
            print(dataset_for_extrapolation)
            print('dataset_for_rf')
            print(dataset_for_rf)
            dataset_for_rf['prediction']= self.rf.predict(dataset_for_rf)
            print("rv_rf")
            print(dataset_for_rf['prediction'])
            dataset_for_extrapolation['prediction'] = self.reg_lin.predict(dataset_for_extrapolation[[self.best_feat_name]].to_numpy())
            print("rv_ex")
            print(dataset_for_extrapolation['prediction'])
            result=pd.concat([dataset_for_rf[['prediction']],dataset_for_extrapolation[['prediction']]])
            print('concat dataframe')
            print(result)
            rv=result[['prediction']]
            
            #
            # df=pd.DataFrame(data=rv, index=dataset_na.index)
            # #
            # # linear regression for point outside the feature ranges
            # #
            # ##rv_lin=self.reg_lin.predict(dataset_na[self.best_feat_name])
            # ##df_lin=pd.DataFrame(data=rv_lin, index=dataset_na.index)
            # maxindex=dataset_na > self.maxvalues
            # print(" ---- maxindex: ")
            # print(maxindex)
            # dataset_max=dataset_na[maxindex].dropna(how='all')
            # print(" ---- dataset_max: ")
            # print(dataset_max)
            # dataset_max=dataset_na[self.best_feat_name][dataset_max.index]
            # print(" ---- dataset_max one: ")
            # print(dataset_max)
            # dataset_for_linear=dataset_na.loc[dataset_max.index,:]
            # print(" ---- dataset_for_linear: ")
            # print(dataset_for_linear)
            # # dataset_max=dataset_max[self.best_feat_name].dropna()
            # ##print(" ---- dataset_max again: ")
            # ##print(dataset_max)
            # dataset_for_rf = dataset_na[dataset_na.index not in dataset_max.index]
            # print(" ---- dataset_for_rf: ")
            # print(dataset_for_rf)
            # ## df[dataset_max.index]=df_lin[dataset_max.index]
            # #
            # # full input
            # #dataset_na
            # #
            # # timestamps to send to linear
            # #dataset_max
            # 
            # #dataset_na[dataset_na.no > ]
            # # if not dataset_max.empty:
            #     # df[self.info_dictionary['pollutant_label']][maxindex[self.best_feat_name]]=\
            #     #     self.extrapolate(dataset_max)
        return rv
    #
    #
    @abstractfunc
    def get_info(self):
        return(self.info_dictionary)

    # def extrapolate(self,dataset): #only run this after boundaries check, only send one column
    #     #this is a linear function for when the RF is out of bounds
    #     import pandas as pd
    #     import numpy  as np
    #     linear_dataset=dataset
    #     if isinstance(linear_dataset, (pd.DataFrame,pd.Series)):
    #         linear_dataset=linear_dataset.to_numpy()
    #     linear_dataset=np.transpose(linear_dataset)
    #     predicted=self.reg_lin.predict(linear_dataset.reshape(-1, 1))
    #     return predicted
    


    
class Calib_RF_FunctionTrainer_o3_001(CalibartionAlgorithmTrainer_interface):
    """
    A trainer example
   cat <<EOF > Calib_RF_FunctionTrainer_o3_001.json
{
  "dates": {
    "end": "2019-08-20 00:00:00",
    "start": "2019-08-01 00:00:00"
  },
  "feat_order": [
    "no2_we",
    "no2_aux",
    "ox_we",
    "ox_aux"
  ],
  "algorithm_parameters": {
    "feat_order_no2": [
      "no2_we",
      "no2_aux"
    ]
  },
  "label_list": [
    "no2","o3"
  ],
  "id_sensor": "4003",
  "interval": "10T",
  "pollutant_label": "o3",
  "test_size": 0.2,
  "trainer_class_name": "Calib_RF_FunctionTrainer_o3_001",
  "trainer_module_name": "calib_RF_FunctionTrain",
  "units_of_measure": {
    "no2": {
      "conversions": [
        {
          "factor": 1.912,
          "from": "ppb"
        }
      ],
      "unit_of_measure": "ug/m^3"
    },
    "o3": {
      "conversions": [
        {
          "factor": 2.0,
          "from": "ppb"
        }
      ],
      "unit_of_measure": "ug/m^3"
    }
  }
}
EOF
       python3 main.py \
         --action trainAndSaveDillToFileFromInfoAndDf \
         --info_file_name      Calib_RF_FunctionTrainer_o3_001.json \
         --dill_file_name      calibrator_o3_001.dill \
         --df_csv_file_prefix "data/calibrator_o3_001"
       python3 main.py \
         --action getInfoFromDillFile \
         --dill_file_name      calibrator_o3_001.dill \
    """
    def init(self,info_dictionary):
        self.info_dictionary=info_dictionary
        self.calibrator=None
    def getCalibrator(self):
        return self.calibrator
    def doTrain(self,df_train_features, df_train_labels_full):
        updateInfoData(self.info_dictionary,df_train_features, df_train_labels_full)
        #
        #
        print('current time:'+str(datetime.datetime.now()))
        
        # n_jobs means the CPU. -1 is all :)
        d=self.info_dictionary['algorithm_parameters']['hyper_parameters']
        conversion_no2=self.info_dictionary['units_of_measure']['no2']['conversions'][0]['factor']
        conversion_o3=self.info_dictionary['units_of_measure']['o3']['conversions'][0]['factor']
        #print(" --- conversion_no2: " + str(conversion_no2))
        #print(" --- conversion_o3: " + str(conversion_o3))
        trees=d['trees']
        leaves=d['leaves']
        #
        df_train_features_no2=\
            df_train_features[self.info_dictionary['algorithm_parameters']['feat_order_no2']]
        df_train_labels_no2=df_train_labels_full["no2"]
        rf_no2 = general_rf_calibration(trees, leaves, df_train_features_no2, df_train_labels_no2)
        #
        df_train_features_o3=df_train_features[self.info_dictionary['feat_order']]
        df_train_labels_o3= \
            df_train_labels_full["o3"]/conversion_o3 + df_train_labels_no2/conversion_no2 
        rf_o3 = general_rf_calibration(trees, leaves, df_train_features_o3, df_train_labels_o3)
        #
        self.calibrator = Calib_RF_Function_o3_001()
        self.calibrator.init(self.info_dictionary, rf_no2, rf_o3)
        return self.calibrator
    
class Calib_RF_Function_o3_001(CalibartionAlgorithm_interface):
    """
    the calibrator produced by the trainer - example
    """
    def init(self,info_dictionary, rf_no2, rf_o3):
        super().init(info_dictionary)
        info=self.info_dictionary
        #
        self.rf_no2 = rf_no2
        self.rf_o3  = rf_o3
    def apply_df(self, data_frame_in):
        dataset_feature_o3=data_frame_in[self.info_dictionary['feat_order']]
        dataset_na_o3=dataset_feature_o3.dropna()
        #
        conversion_no2=self.info_dictionary['units_of_measure']['no2']['conversions'][0]['factor']
        conversion_o3=self.info_dictionary['units_of_measure']['o3']['conversions'][0]['factor']
        #print(" --- conversion_no2: " + str(conversion_no2))
        #print(" --- conversion_o3: " + str(conversion_o3))
        #
        dataset_feature_no2=data_frame_in[self.info_dictionary['algorithm_parameters']['feat_order_no2']]
        dataset_na_no2=dataset_feature_no2.dropna()
        #
        if (dataset_na_o3.empty) or (dataset_na_no2.empty):
            return np.nan
        else:
            #this apply doesn't work for o3 if it doesn't contain NO2 raw features in it's calibration
            #for NO, NO2, and CO the RF results are in ug/m3 directly.
            # 1 ppb is 1.9120 ug/m3
            # for o3 only, the results of the ranfomforest is in PPB.
            # this finds the difference between O3+NO2 ppb and NO2 ppb
            # to extract O3, and convert to ug/m3 in STP only.
            # 1ppb is 1.9950 ug/m3
            rv_no2=self.rf_no2.predict(dataset_na_no2)/conversion_no2
            rv = (self.rf_o3.predict(dataset_na_o3) - rv_no2) * conversion_o3
        return rv
    #
    #
    @abstractfunc
    def get_info(self):
        return(self.info_dictionary)



def general_rf_calibration(trees, leaves, df_train_features, df_train_labels):
    rf = RandomForestRegressor(n_estimators = trees, n_jobs =-1, \
                               min_samples_leaf=leaves,criterion="mae",max_features="auto") 
    # A smaller leaf makes the model more prone to capturing noise in train data.
    # max_depth represents the depth of each tree in the forest.
    # model overfits for large depth values
    #
    # Train the model on training data
    rf.fit(df_train_features, df_train_labels)
    print (rf)
    return rf


def updateInfoData(info_dictionary,df_train_features, df_train_labels_full):
    df_train_labels=df_train_labels_full[info_dictionary['pollutant_label']]
    #
    print("Hello! updateInfoData " \
          + info_dictionary['trainer_class_name']\
          + "."+info_dictionary['trainer_module_name']\
          +" ...")
    #print(str(df_train_features))
    #print("--- labels ---")
    # print(str(df_train_labels))
    #
    trees=1500   # roughly the optimum need to recheck for all sensors
    leaves=10    # the minimum number of leaves per a split
    if ('hyper_parameters' in info_dictionary['algorithm_parameters']):
        d=info_dictionary['algorithm_parameters']['hyper_parameters']
        trees=d['trees']
        leaves=d['leaves']
    info_dictionary['algorithm_parameters']['hyper_parameters']=\
        {'trees':trees,'leaves':leaves}
    info_dictionary["python_env"]={'sklearn':sklearn.__version__, 
                                   'pandas':pd.__version__,
                                   'dill':dill.__version__,
                                   'numpy':np.__version__,
                                   'psycopg2':psycopg2.__version__,
                                   'scipy':scipy.__version__}
    info_dictionary["features"]={}
    #
    ACmin_values=np.percentile(df_train_features,0,axis=0)
    ACmax_values=np.percentile(df_train_features,100,axis=0)
    for idx,feat in enumerate(info_dictionary['feat_order']):
        info_dictionary['features'][feat]={}
        # add the max and min values
        info_dictionary['features'][feat]['range']=  ['%.2f' %float(ACmin_values[idx]),'%.2f' %float(ACmax_values[idx])]
        info_dictionary['features'][feat]['unit_of_measure']=  'mV'
    #
    for label_name in info_dictionary['label_list']:
       l="label_"+ label_name
       info_dictionary[l]={}
       info_dictionary[l]['range']=['%.2f' %float(df_train_labels.min()),'%.2f' %float(df_train_labels.max())]
       [min(df_train_labels),max(df_train_labels)]
       info_dictionary[l]['unit_of_measure']=info_dictionary['units_of_measure'][label_name]['unit_of_measure']
    #
    l="label_"+ info_dictionary['pollutant_label']
    info_dictionary[l]
    info_dictionary['pollutant_unit_of_measure']=info_dictionary['units_of_measure'][info_dictionary['pollutant_label']]['unit_of_measure']


