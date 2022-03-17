# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:44:19 2019

@author: Ohad Zivan
"""
#add to class the pollutant currently calibrated. just to make sure :)

import time
import dill
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import os
import datetime # this is to pring the current time during the run
import matplotlib.dates as mdates
import csv 
import sklearn
import json
import psycopg2
import pandas.io.sql as sqlio
import argparse

# class for the trained algorythem. it checks for the max and min, and reutrn nan if df is empty
class CalibartionAlgorythem():
    import pandas as pd
    def __init__(self):
        {}
    def PollutantTrained(self,pollutant):
        self.pollutant=pollutant
    def TimeUsed(self,timestep):
        self.timestep=timestep
    def AddMinValues(self,minvalues): # doesn't need NO2 max_min as they are implemented in the code
        self.minvalues=minvalues
    def AddMaxValues(self,maxvalues):
        self.maxvalues=maxvalues
    def AddFeatureList(self,feature_list):
        self.feature_list=feature_list
    def AddFeatureList_NO2(self,feature_list_NO2):
        self.feature_list_NO2=feature_list_NO2
    def AddBuiltCalibration(self,builtCalibration):
        self.builtCalibration=builtCalibration
    def AddBuiltCalibration_NO2(self,builtCalibration_NO2):
        self.builtCalibration_NO2=builtCalibration_NO2
    def AddLinear(self,feature_max,reg_lin):
        self.feature_max=feature_max
        self.reg_lin=reg_lin
   
    def Add_info(self,json_info): # saves a json with all the information for the model
        self.get_json=json_info
    
    def Extrapolate(self,dataset): #only run this after boundaries check, only send one column
        #this is a linear function for when the RF is out of bounds
        linear_dataset=dataset
        linear_dataset=np.array([linear_dataset]) # do not remove the []. they are important for the shape!
        linear_dataset=np.transpose(linear_dataset)
        return self.reg_lin.predict(linear_dataset)
    
    def apply_np (self,dataset): # gets a numbery arrary of preselected features and returns calibrated values
        dataset_df=pd.DataFrame(data=dataset,    # values
        columns=self.feature_list)  # 1st row as the column names
        dataset_apply= self.apply_df(dataset_df)
        return dataset_apply[self.pollutant].to_numpy() # returns only the value
    
    def apply_df(self,dataset): #this function applies the calibration to new data
        dataset_feature=dataset[self.feature_list]
        dataset_na=dataset_feature.dropna()
        #need to add a test for outside of the bounds. they should be percent
        if dataset_na.empty:
            return np.nan
        else:
            #this apply doesn't work for o3 if it doesn't contain NO2 raw features in it's calibration
            #for NO, NO2, and CO the RF results are in ug/m3 directly.
            predicted=self.builtCalibration.predict(dataset_na)
           
            try: # change to if with polllutant used
                dataset_na_NO2=dataset_na[self.feature_list_NO2]
                #this turns NO2 ug/m3 into ppb. based on STP only.
                predicted_NO2=self.builtCalibration_NO2.predict(dataset_na_NO2)*(1/1.9120) # 1 ppb is 1.9120 ug/m3
                # for o3 only, the results of the ranfomforest is in PPB.
                #this finds the difference between O3+NO2 ppb and NO2 ppb to extract O3, and convert to ug/m3 in STP only.
                predicted=(predicted-predicted_NO2)*1.9950 # 1ppb is 1.9950 ug/m3
            
            except:
                {}
                
            df=pd.DataFrame(data=predicted, index=dataset_na.index)
            df.columns=[self.pollutant]
            #makes the linear interpolation for values outside the boundaries
            #need to add a check
            minindex=dataset_na<self.minvalues
            dataset_min=dataset_na[minindex]
            dataset_min=dataset_min[self.feature_max].dropna()
            if not dataset_min.empty:
                df[self.pollutant][minindex[self.feature_max]]=self.Extrapolate(dataset_min)
            maxindex=dataset_na>self.maxvalues
            dataset_max=dataset_na[maxindex]
            dataset_max=dataset_max[self.feature_max].dropna()
            if not dataset_max.empty:
                df[self.pollutant][maxindex[self.feature_max]]=self.Extrapolate(dataset_max)
            df[df <0 ] =0
            return  df

# run this to check
# os.system("python -i RF_FunctionTrain_DB_Ex_11122019.py -h") 
# this should end with sys.exit error. it's not an error, just how it works. on linux/bash it should be fine
"""            
parser = argparse.ArgumentParser(description='Train Random forest')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
args = parser.parse_args()
"""

#
#
# input sections
#
#   file_Calibration
#   Date_Calibration_start
#   Date_Calibration_finish
#   QA_Date_Calibration_start
#   QA_Date_Calibration_finish
#   file_Calibration_AQM
#   TimeSample  (10T)
#   sensors_list
#   All_Pollutant
#
#   Dirforplots
#   
#
#
#
#

json_for_dill={}
json_for_dill['python_env']={'sklearn':sklearn.__version__, 
            'pandas':pd.__version__,
            'dill':dill.__version__,
            'numpy':np.__version__,
            'psycopg2':psycopg2.__version__}



dill_path='.' #this is for the NO2 dills - for O3 calibration 
def walklevel(some_dir, level=1): #copied from stackexchange
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def trafair_db_getConnection(): #port 5432 if installed on the server
  conn = psycopg2.connect("dbname='trafair' user='trafair_ro' host='localhost' port='5532' password='trafair_ro'")
  return(conn)

conn = trafair_db_getConnection()

DB_id=1 # the id for the new calibration in the db
timestr = time.strftime("%Y%m%d_%H%M%S")



Date_Calibration_start=input('Date to Start (yyyy-mm-dd) (leave empty for default: 2019-08-19): \n')
Date_Calibration_finish=input('Date to Finish (yyyy-mm-dd) (leave empty for default: 2019-11-04): \n')

if not Date_Calibration_start: # calibration starts on the 8th, since the first few days, the AQM values were wrong
 #Date_Calibration_start='2019-10-08' # for 4009-4014
 Date_Calibration_start='2019-08-19' # for 4003-4008 (exluding 4004)
if not Date_Calibration_finish:
 #Date_Calibration_finish='2019-10-15'
 Date_Calibration_finish='2019-11-04' # for 4003-4008 (exluding 4004)


""" the stations in Modena: 
id	name	description	geom
9627	GIARDINI	MODENA VIA GIARDINI	0101000020E6100000BC76E4003ACF254097B8435669514640
9625	PARCO FERRARI	MODENA PARCO FERRARI	0101000020E61000003F07C10F0AD02540E6F0603E47534640
for the 1min data, the Parco is 42. 
ignoring the number for now, since the files will contain only one AQM.
"""



TimeSample=input('Choose timestep to average (leave empty for default: 10 minutes): \n options: 1T, 2T, 5T 10T, 30T, 1H \n') # ac.hae.desp.csv  AC009.csv
if not TimeSample:
 TimeSample='10T'

sql_lowcost=""" select o.phenomenon_time
        , ss.status
        , ss.id_sensor_low_cost
        , o.humidity
        , o.temperature
        , o.no_we
        , o.no_aux
        , o.no2_we
        , o.no2_aux 
        , o.ox_we
        , o.ox_aux
        , o.co_we
        , o.co_aux
   from sensor_raw_observation o
      , sensor_low_cost_status ss
  where o.phenomenon_time >= '%s' and o.phenomenon_time <= '%s'
    and id_sensor_low_cost_status = ss.id
    and ss.status = 'calibration' 
   ;""" %(Date_Calibration_start, Date_Calibration_finish)




Dirforplots='RF_Extraplotation_'+timestr # this makes the directory the same as the current one the file is in. 
os.makedirs(Dirforplots)
file_= open(os.path.join(Dirforplots,'Run_all_'+'_'+Date_Calibration_start+'_'+Date_Calibration_finish+'_'+timestr+'.txt'),'w+')
file_.write('Calibrating all pollutants \n')

file_.write('Date to Start (yyyy-mm-dd) :  ' +Date_Calibration_start+'\n')
file_.write('Date to Finish (yyyy-mm-dd) : ' +Date_Calibration_finish+'\n')
file_.write('Time Step used: ' + TimeSample+'\n')


dataset_Calibration_original=sqlio.read_sql_query(sql_lowcost, conn)
#Dataset_Calibration_original=Dataset_Calibration_original.drop(['battery_voltage'], axis=1) #drop the battery coloum it's useless
dataset_Calibration_original.index=dataset_Calibration_original.phenomenon_time # change the index to datetime format

All_Pollutant_all=['no','no2','o3','co']
All_Pollutant=input('Choose the pollutants you want to calibrate, leave empty for all \n' + str(All_Pollutant_all) +' \n')
if not All_Pollutant: # check if empty -  if so, get the entire list
    All_Pollutant=All_Pollutant_all
else: # gets the input and convert to int list
    All_Pollutant=All_Pollutant.replace(",", " ").split() # split the str to list    
    for poll in All_Pollutant: # check the numbers of the sensors match
        if poll not in All_Pollutant_all:
                raise ValueError(poll + ' is not a valid pollutant')

if 'no2' in All_Pollutant:
    dill_path=Dirforplots


#choose the sesnors you want to calibrate
sensors_list_all=list(set(dataset_Calibration_original['id_sensor_low_cost'])) #gets the entire sensors list in the data
sensors_list=input('Choose the sensor numbers you want to calibrate, leave empty for all \n' + str(sensors_list_all)+' \n')
if not sensors_list: # check if empty -  if so, get the entire list
    sensors_list=sensors_list_all
else: # gets the input and convert to int list
    sensors_list=sensors_list.replace(",", " ").split() # split the str to list
    sensors_list=[int(i) for i in sensors_list] # change the list to int list
    
    for sen in sensors_list: # check the numbers of the sensors match
        if sen not in sensors_list_all:
                raise ValueError(str(sen) + ' is not a valid sensor number')




# this following is for the AQM 
sql_AQM="""  select o.id_aq_legal_station
        , o.phenomenon_time
        , o.c6h6
        , o.co
        , o.no
        , o.no2
        , o.o3
        , ss.datetime
        , ss.datetime_end
        , ss.status
        , sf.code
   from aq_legal_station_observation_one_minute_not_validated o
      , sensor_low_cost_feature sf
      , ( select ss1.id_sensor_low_cost
               , ss1.status
               , ss1.id_sensor_low_cost_feature
               , ss1.operator
               , ss1.datetime
               , (select ss2.datetime
                    from sensor_low_cost_status ss2
                   where ss1.id_sensor_low_cost = ss2.id_sensor_low_cost
                     and ss1.datetime < ss2.datetime
                    order by ss2.datetime
                    limit 1
                   ) as datetime_end
                  from sensor_low_cost_status ss1
         ) ss
  where o.phenomenon_time >= '%s' and o.phenomenon_time <= '%s'
    and o.phenomenon_time >= ss.datetime and (o.phenomenon_time < ss.datetime_end  
                                              or ss.datetime_end is null )
    and ss.id_sensor_low_cost_feature = sf.id
    and ss.status = 'calibration' 
    and o.id_aq_legal_station = sf.id_aq_legal_station
    order by o.phenomenon_time
""" %(Date_Calibration_start, Date_Calibration_finish)

dataset_AQM_original=sqlio.read_sql_query(sql_AQM, conn)

dataset_AQM_original.index=dataset_AQM_original.phenomenon_time




"""
one min data codes
select the relevant data before joining dates.
1;SO2
2;NOX
3;NO
4;NO2
5;PTS
6;CO
7;O3
"""

file_.write('Calibration results: \n')
file_.write('--------------------------------------------- \n')
file_.write('\n')


json_for_dill['timestep']='10T'

Sensor_predictions= {}

saved_features={}
for Pollutant_Calibration in All_Pollutant:
    """  
    if Pollutant_Calibration.lower()=='no':
        AQM_Pollutant=3
        dataset_AQM_selected=dataset_AQM_original[dataset_AQM_original.Misura==AQM_Pollutant]

    elif Pollutant_Calibration.lower()=='no2':
        AQM_Pollutant=4
        dataset_AQM_selected=dataset_AQM_original[dataset_AQM_original.Misura==AQM_Pollutant]

    elif Pollutant_Calibration.lower()=='o3':
        #correct ug/m3 to ppb, since the lowcost sensor measures chemical reactions, and we need ths sum of no2 and o3
        AQM_Pollutant=7
        TempAQM_Pollutant_O3=pd.DataFrame(data=dataset_AQM_original[dataset_AQM_original.Misura==AQM_Pollutant].Valore)
        #this turns O3 ug/m3 into ppb. based on STP only.
        TempAQM_Pollutant_O3=TempAQM_Pollutant_O3*(1/1.9950) # 1ppb is 1.9950 ug/m3
        TempAQM_Pollutant_NO2=pd.DataFrame(data=dataset_AQM_original[dataset_AQM_original.Misura==4].Valore )#the number for NO2 - the Ox measure both NO2 and O3
        #this turns NO2 ug/m3 into ppb. based on STP only.
        TempAQM_Pollutant_NO2=TempAQM_Pollutant_NO2*(1/1.9120) # 1 ppb is 1.9120 ug/m3
        TempAQM_Pollutant_O3.columns=['o3']
        TempAQM_Pollutant_NO2.columns=['no2']
        TempAQM_Pollutant_combined=TempAQM_Pollutant_NO2.merge(TempAQM_Pollutant_O3, how='outer', right_index=True, left_index=True).dropna()
        TempAQM_Pollutant_combined['Valore']=TempAQM_Pollutant_combined.no2+TempAQM_Pollutant_combined.o3
        dataset_AQM_selected=TempAQM_Pollutant_combined

    elif Pollutant_Calibration.lower()=='co':
        AQM_Pollutant=6
        dataset_AQM_selected=dataset_AQM_original[dataset_AQM_original.Misura==AQM_Pollutant]
    """
    dataset_Calibration=dataset_Calibration_original[Date_Calibration_start:Date_Calibration_finish]
    dataset_AQM=dataset_AQM_original[Date_Calibration_start:Date_Calibration_finish]
    

# applies the userspecified dates. 
    saved_features[Pollutant_Calibration]={}

    for x in sensors_list:
        starttime = time.time()
    
        dataset_Calibration_10min=dataset_Calibration.loc[dataset_Calibration['id_sensor_low_cost'] == x]
    
        dataset_Calibration_10min=dataset_Calibration_10min.resample(TimeSample).mean() # this average over 10min :D
        dataset_AQM_10min=dataset_AQM.resample(TimeSample).mean() # this make sure they are at the same dates
        
        dataset_Calibration_10min=dataset_Calibration_10min.merge(dataset_AQM_10min, how='outer', right_index=True, left_index=True)
        
        
        # this part removes everything below the detection limit. it can be zero to remove negative values only
        Cutoff_Value=0.1 # this is the value to cut everything below. applied for both the calibration and re-location
        
        dataset_Calibration_10min=dataset_Calibration_10min.astype('float')
        Check=dataset_Calibration_10min[Pollutant_Calibration].copy() 
        a=0
        for current, next in zip(Check, Check[1:]): # this loop removes constant values.
            if current==next:
                Check[a]=np.nan
            a=a+1
        
        Check=[float('nan') if x<Cutoff_Value else x for x in Check] # this removes values below the LOD
            
        dataset_Calibration_10min[Pollutant_Calibration.lower()]=Check.copy()
        dataset_Calibration_10min_NA=dataset_Calibration_10min # drop everything NA
        
        
        # labels are the values we want to predict (target values)
        # list(dataset_hae_10min)
        
        # the list of coloumn names to use in the RF from the Aircubes. should be similar across Trafair
        
        if Pollutant_Calibration.lower()=='no':
            dataset_Calibration_10min_NA_features_list=[  'no_aux', 'no_we',] 
        elif Pollutant_Calibration.lower()=='no2':
            dataset_Calibration_10min_NA_features_list=[  'no2_aux', 'no2_we',] 
        elif Pollutant_Calibration.lower()=='o3':
            dataset_Calibration_10min_NA_features_list=[  'ox_aux', 'ox_we','no2_aux', 'no2_we',] 
        elif Pollutant_Calibration.lower()=='co':
            dataset_Calibration_10min_NA_features_list=[  'co_aux', 'co_we',] 
        
       
        #dataset_Calibration_10min_NA_features_list=[  'no_aux', 'no_we','no2_aux', 'no2_we','ox_aux', 'ox_we', 'co_aux', 'co_we','humidity','temperature']
        dataset_Calibration_10min_NA_temp=dataset_Calibration_10min_NA[dataset_Calibration_10min_NA_features_list+[Pollutant_Calibration]].dropna()
        dataset_Calibration_10min_NA_features = dataset_Calibration_10min_NA_temp[dataset_Calibration_10min_NA_features_list]
        dataset_Calibration_10min_NA_label = dataset_Calibration_10min_NA_temp[Pollutant_Calibration]
        dataset_Calibration_10min_NA_features_label=dataset_Calibration_10min_NA_features_list

        # the minimum values obtained from the AC. this is for applying the calibration afterwards. if the future values are below this, the calibration doesn't work
        # 0 and 100 are percentile of the values in the dataset. it's possible to change to lower values
        ACmin_values=np.percentile(dataset_Calibration_10min_NA_features,0,axis=0)
        ACmax_values=np.percentile(dataset_Calibration_10min_NA_features,100,axis=0)
        json_for_dill['feat_order']=dataset_Calibration_10min_NA_features_list
        json_for_dill['features']={}
        for idx,feat in enumerate(dataset_Calibration_10min_NA_features_list):
            json_for_dill['features'][feat]={}
            json_for_dill['features'][feat]['range']=  [ACmin_values[idx],ACmax_values[idx]] # add the max and min values
        json_for_dill['label']={}
        json_for_dill['label']['range']=[min(dataset_Calibration_10min_NA_label),max(dataset_Calibration_10min_NA_label)]
        
        # Using Skicit-learn to split data into training and testing sets
        from sklearn.model_selection import train_test_split
        # Split the data into training and testing sets
        train_features, test_features, train_labels, test_labels = \
        train_test_split(dataset_Calibration_10min_NA_features, dataset_Calibration_10min_NA_label,\
                         test_size = 0.20, )
        # make the maximum and minimum 
        file_.write('\n')
        file_.write('Sensor number:' + str(x)+', Pollutant: '+Pollutant_Calibration.upper()+'\n')
        file_.write('Training Features Shape:' + str(train_features.shape)+'\n')
        file_.write('Training Labels Shape:' + str(train_labels.shape)+'\n')
        file_.write('Testing Features Shape:' + str(test_features.shape)+'\n')
        file_.write('Testing Labels Shape:'+ str(test_labels.shape)+'\n')
        
        endtime = time.time()
        file_.write('Loading time:'+ str(endtime - starttime) + ' seconds'+'\n')
        
        # Import the model we are using
        from sklearn.ensemble import RandomForestRegressor


        #  model with 1500 decision trees 
        trees=1500 #roughly the optimum need to recheck for all sensors
        leaves=10 # the minimum number of leaves per a split
        json_for_dill['hyper_parameters']={'trees':trees,'leaves':leaves}
        file_.write('Number of trees:'+ str(trees) +'\n')
        print('starting the Random Forest model build for sensor: '+str(x)+' and gas: '+str(Pollutant_Calibration)+', this might take a while')
        print('current time:'+str(datetime.datetime.now()))
        
        rf = RandomForestRegressor(n_estimators = trees, n_jobs =-1, \
                                   min_samples_leaf=leaves,criterion="mae",max_features="auto") 
        # n_jobs means the CPU. -1 is all :)
        
        # A smaller leaf makes the model more prone to capturing noise in train data.
        # max_depth represents the depth of each tree in the forest. model overfits for large depth values
        
        # Train the model on training data

        rf.fit(train_features, train_labels);
        from mlxtend.evaluate import feature_importance_permutation
        def Spearman_func(X,Y):
            tmp=st.spearmanr(X,Y)
            return tmp[0]
        imp_vals, temp = feature_importance_permutation(predict_method=rf.predict, X=np.array(test_features), y=np.array(test_labels), metric=Spearman_func,num_rounds=10, seed=10)

        endtime = time.time()
        file_.write('Random Forest building:'+ str(endtime - starttime) + ' seconds'+'\n')
        print('Finished building the model, now getting some statistics')
        print('current time:'+str(datetime.datetime.now()))

        # do linear regression for extrapolating
        from sklearn.linear_model import LinearRegression
        feature_max=max(zip(imp_vals,dataset_Calibration_10min_NA_features_list))[1]
        dataset_linear=np.array([train_features[feature_max]])
        dataset_linear=np.transpose(dataset_linear)
        reg_lin=LinearRegression().fit(dataset_linear, np.array(train_labels))
        file_.write('Linear regression feature used: '+feature_max+' \n')
        file_.write('Linear regression score (r square): '+str(reg_lin.score(dataset_linear, train_labels))+'\n')
        
        # the actual dates used for calibration
        json_for_dill['dates']={'start':dataset_Calibration_10min_NA_features.index[0].strftime("%d-%b-%Y %H:%M"),
                     'end':dataset_Calibration_10min_NA_features.index[-1].strftime("%d-%b-%Y %H:%M")}
        # save the rf model 
        rf_filename=os.path.join(Dirforplots,str(x)+'_'+Pollutant_Calibration.upper()+'_'+str(DB_id)+'.dill')
        rf_dict=CalibartionAlgorythem()
        rf_dict.PollutantTrained(Pollutant_Calibration)
        rf_dict.TimeUsed(TimeSample)

         # this is for the dill
        #load NO2 dill for the O3 calibration - need to update the dill_path at the top of the code
        if Pollutant_Calibration.lower()=='o3':
            for root, dirs, files in walklevel(dill_path,level=0):  #level 0 means only this directory
                for filename in files:
                    if filename.endswith(".dill"):
                        if str(x) in filename:
                            if 'NO2_' in filename:
                                rf_NO2=dill.load(open(os.path.join(dill_path,filename),'rb'))
                                print('Found NO2 file to load: '+filename)
                                
                                #finds the collected O3 and NO2 min and max values 
                                min_index=np.percentile(dataset_Calibration_10min_NA_features[rf_NO2.feature_list],0,axis=0)>=rf_NO2.minvalues
                                max_index=np.percentile(dataset_Calibration_10min_NA_features[rf_NO2.feature_list],0,axis=0)<=rf_NO2.maxvalues
                                #ACmin_values=rf_NO2.minvalues[min_index] #need to fix this somehow. currently, the NO2 limit isn't considered.
                                #ACmax_values=rf_NO2.maxvalues[max_index]
                                rf_dict.AddFeatureList_NO2(rf_NO2.feature_list)
                                rf_dict.AddBuiltCalibration_NO2(rf_NO2.builtCalibration)
        
        rf_dict.AddMinValues(ACmin_values)
        rf_dict.AddMaxValues(ACmax_values)
        rf_dict.AddFeatureList(dataset_Calibration_10min_NA_features_list)
        rf_dict.AddBuiltCalibration(rf)
        rf_dict.AddLinear(feature_max,reg_lin) #this adds the linear reg to the dill
        # pickle.dump(rf_dict, open(rf_filename, 'wb'))   # previous pickle dumb. upgraded to dill
        # Get numerical feature importances
        DB_id=DB_id+1

        importances = list(rf.feature_importances_)
        file_.write('Feature Importance from RF'+'\n')
        for feature in zip(dataset_Calibration_10min_NA_features_label, importances):
            file_.write(str(feature)+'\n')
        
        file_.write('\n Feature Importance from permutation'+'\n')
        
        saved_features[Pollutant_Calibration][x]={}
        for feature in zip(dataset_Calibration_10min_NA_features_label, imp_vals):
            file_.write(str(feature)+'\n')
            saved_features[Pollutant_Calibration][x][feature[0]]= feature[1]
        # Use the forest's predict method on the test data
        predictions = rf_dict.apply_df(test_features)
        predictions_np=np.array(predictions)
        #predictions=rf.predict(test_features)
        errors = abs(predictions_np - np.array(test_labels))
        errors_Percent = np.mean((errors/np.array((test_labels))))
        Spearman_R=st.spearmanr(predictions,test_labels)
        file_.write(' Mean Aircube in calibration: '+str(round(np.mean(predictions_np), 2)) + ' ppb \n')
        file_.write(' Max Aircube in calibration: '+str(round(np.max(predictions_np), 2)) + ' ppb \n')
        file_.write(' Min Aircube in calibration: '+str(round(np.min(predictions_np), 2)) + ' ppb \n')
        file_.write(' Mean AQM in calibration: ' + str(round(np.mean(dataset_Calibration_10min_NA_label), 2)) + ' ppb \n')
        
        file_.write(' Mean Absolute Error: '+ str(round(np.mean(errors), 2)) + ' ppb \n')
        file_.write(' RMSE: ' + str(round(np.sqrt(((predictions_np - np.array(test_labels)) ** 2).mean()), 2)) + ' ppb \n')
        file_.write(' Coefficient of variation: '+ str(round(np.std(errors)/np.mean(errors)*100, 2) )+ '% \n')
        file_.write(' Spearman rank correlation: '+str(round(Spearman_R[0],3))+' \n')
        
        # this two lines are to calculate the predicated data. and add to general df    
        json_for_dill['Statistics']={'Spearman correlation coefficient [-]':round(Spearman_R[0],2),
                     'Mean Absolute Error [ug/m3]' : round(np.mean(errors), 2), 
                     'Root Mean Square Error [ug/m3]': round(np.sqrt(((predictions_np - np.array(test_labels)) ** 2).mean()), 2),
                     'Coefficient of variation [%]':round(np.std(errors)/np.mean(errors)*100, 2) }
        endtime = time.time()
        file_.write(' Predictions calculation time '+ str(endtime - starttime) + ' seconds \n')
        
        rf_dict.Add_info(json.dumps(json_for_dill))
        
        dill.dump(rf_dict, open(rf_filename, 'wb'))


        # plot aqm vs predicated 
        fig1= plt.figure(figsize=(16, 8.0))
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.plot(predictions,test_labels,'.')
        ax1.set_title(Pollutant_Calibration+' Validation results')
        ax1.set_xlabel('AirCube '+str(x) +': '+Pollutant_Calibration.upper())
        ax1.set_ylabel('AQM ParcoFerrari')
        plt.show()
        plt.savefig(os.path.join(Dirforplots,'Run_'+Pollutant_Calibration+'_g1_'+Date_Calibration_start+'_'+Date_Calibration_finish+'_'+timestr+'_Sensor_'+str(x)+'.png'))
    
        # plot timeseries of AQM and Aircube
        fig2= plt.figure(figsize=(16, 8.0))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.plot(dataset_Calibration_10min_NA_features.index,rf.predict(dataset_Calibration_10min_NA_features),'-.',label='Calibrated Aircube '+str(x))
        ax2.plot(dataset_Calibration_10min_NA.index,dataset_Calibration_10min_NA[Pollutant_Calibration],'-*',label='AQM ParcoFerrari')
        ax2.set_title(Pollutant_Calibration+' Time Series')
        ax2.set_xlabel('Date and Time')
        ax2.set_ylabel(Pollutant_Calibration.upper()+' Concentration '+r'$[\mu g \cdot m^{-3}]$')
        Days = mdates.DayLocator()   # every day
        Hours = mdates.HourLocator()  # every hour
        Days_fmt = mdates.DateFormatter('%d-%b')
        """
        ax2.xaxis.set_major_locator(Days)
        ax2.xaxis.set_major_formatter(Days_fmt)
        ax2.xaxis.set_minor_locator(Hours)
        """
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width , box.height])
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=2)
    
        plt.show()
        plt.savefig(os.path.join(Dirforplots,'Run_'+Pollutant_Calibration+'_g2_'+Date_Calibration_start+'_'+Date_Calibration_finish+'_'+timestr+'_Sensor_'+str(x)+'.png'))
    
    
        fig3= plt.figure(figsize=(16, 8.0))
        ax3 = fig3.add_subplot(1, 1, 1)
        ax3.plot(dataset_Calibration_10min_NA[dataset_Calibration_10min_NA_features_list])
        ax3.set_xlabel('Date and Time for sensor '+str(x))
        Days = mdates.DayLocator()   # every day
        Hours = mdates.HourLocator()  # every hour
        Days_fmt = mdates.DateFormatter('%d-%b')
        """
        ax3.xaxis.set_major_locator(Days)
        ax3.xaxis.set_major_formatter(Days_fmt)
        ax3.xaxis.set_minor_locator(Hours)
        """
        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width , box.height])
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=4, labels=dataset_Calibration_10min_NA_features_list)
        plt.show()
    
        plt.savefig(os.path.join(Dirforplots,'Run_'+Pollutant_Calibration+'_g3_'+Date_Calibration_start+'_'+Date_Calibration_finish+'_'+timestr+'_Sensor_'+str(x)+'.png'))
    
        plt.close("all")
    
with open(os.path.join(Dirforplots,'feature_output_'+timestr+'.csv'), 'w') as csv_file:
    csvwriter = csv.writer(csv_file, delimiter=',')
    for poll in saved_features:
        for session in saved_features[poll]:
            for item in saved_features[poll][session]:
                csvwriter.writerow([poll,session, item, saved_features[poll][session][item]])

file_.close() 
plt.close("all")

