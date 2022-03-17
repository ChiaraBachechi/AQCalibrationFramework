# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:45:08 2019

@author: ohadz
"""



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
