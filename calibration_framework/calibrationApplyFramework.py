import sys
import json
import psycopg2
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
import datetime

sys.path.insert(0, '../..')
from trafair_db_config import trafair_db_getConnection

"""
 A tool library for applying calibration
 requirements
   pip3 install numpy pandas
"""

class CalibrationApplyFramework():
    def applyCalibrationSensorPollutantDillDf(self
                                              , calibrator
                                              , begin_time
                                              , end_time
                                              , id_sensor
                                              , interval_in_minutes
                                              , pollutant_label
                                              , do_persist_data
    ):
        """
        This function never write the DB,
        because lacks data: id_sensor_calibration and
        the other pollutants.
        """
        #
        #
        # getting data to calibrate
        #
        records = self.getDataToApplyAsDataFrame(begin_time
                                                 , end_time
                                                 , id_sensor
                                                 , interval_in_minutes
        )
        records = records.rename(columns={'phenomenon_time_rounded':'phenomenon_time'})
        output=records.copy()
        output['result_time']=datetime.datetime.now()
        #
        output[pollutant_label] = calibrator.apply_df(records)
        #
        print('--input size--')
        print(records.shape)
        print('--output size--')
        print(records.shape)
        print('----output----')
        print(output)
        #sqlio.to_sql(output,table_name,sql_engine,if_exists='append',index=False)
        #
        # if (do_persist_data):
        #     conn = trafair_db_getConnection()
        #     print("writing data to db..")
        #     # conn.commit()


    def getDataToApply_theSqlQuery(self):
        rv="""
         select status.id_sensor_low_cost 
           , (to_timestamp(ceil(extract(epoch from phenomenon_time::timestamp with time zone) / (60 * %s )) * (60 * %s)))::timestamp as phenomenon_time_rounded
           ,  count(id_sensor_low_cost_status) as coverage
           , status.id_sensor_low_cost_feature,
            avg(no_aux) as no_aux,
            avg(no_we) as no_we,
            avg(no2_aux) as no2_aux,
            avg(no2_we) as no2_we,
            avg(ox_aux) as ox_aux,
            avg(ox_we) as ox_we,
            avg(co_aux) as co_aux,
            avg(co_we) as co_we,
            avg(humidity) as humidity ,
            avg(temperature) as temperature
           from sensor_raw_observation as raw, sensor_low_cost_status as status
           where status.id_sensor_low_cost = %s
             and raw.id_sensor_low_cost_status = status.id
             and phenomenon_time < %s
             and phenomenon_time >= %s
             and (status.status = 'running' or status.status = 'calibration')
           group by(status.id_sensor_low_cost,status.id_sensor_low_cost_feature
                            , phenomenon_time_rounded)
           order by phenomenon_time_rounded
           ;
        """
        return(rv)
   
    def getDataToApplyAsCursor(self, begin_time
                               , end_time
                               , id_sensor
                               , interval_in_minutes
        ):
        conn = trafair_db_getConnection()
        cur = conn.cursor()
        sqlLowCost2=self.getDataToApply_theSqlQuery()
        cur.execute(sqlLowCost2,
                    (
                        str(interval_in_minutes)
                        , str(interval_in_minutes)
                        , id_sensor
                        , end_time
                        , begin_time
                    ))
        return(cur)

    def getDataToApplyAsDataFrame(self, begin_time
                                  , end_time
                                  , id_sensor
                                  , interval_in_minutes
        ):
        conn = trafair_db_getConnection()
        sqlLowCost2=self.getDataToApply_theSqlQuery()
        records=sqlio.read_sql_query(sqlLowCost2,conn
                                     , params=(
                                         str(interval_in_minutes)
                                         , str(interval_in_minutes)
                                         , id_sensor
                                         , end_time
                                         , begin_time
                                     ))
        return(records)

        
