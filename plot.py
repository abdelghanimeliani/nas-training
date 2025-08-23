'''
this script is used to plot the results of the experiments
The nni framework stores the resutls in an sqlite database inside /home/username/nni-experiments/experiment_id/db/nni.sqlite
the database contains 3 main tables: TrialJobEvent,MetricData, ExperimentProfile
this script connects get incide every dir and get the results of the experiments
# and plots the results using matplotlib
'''
import sqlite3
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import json
import csv
from ast import literal_eval
from utils import convert_ExpirimentProfile_tables_to_csv , convert_MetricData_table_to_csv

def get_expiriments_ids_list(base_path):
    ids= [p.name for p in Path(base_path).iterdir() if p.is_dir() ]
    ids.remove('_latest')
    return ids  

def convert_exp_tables_to_csv(base_path, experiments_ids_list):
    for exp_id in experiments_ids_list:
        conn = sqlite3.connect(base_path + '/' + exp_id + '/db/nni.sqlite')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM  ExperimentProfile")
        data= cursor.fetchall()
        convert_ExpirimentProfile_tables_to_csv(data, "./csv/exp_profiles/" + exp_id +".csv")
        cursor.close()
        conn= sqlite3.connect(base_path + '/' + exp_id + '/db/nni.sqlite')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM MetricData")
        data= cursor.fetchall()
        convert_MetricData_table_to_csv(data, "./csv/metric_data/" + exp_id +".csv")
        cursor.close()
        conn.close()
    
        
    

if __name__ == "__main__":
    exp_ids=get_expiriments_ids_list("/home/meliani/nni-experiments")
    print(str(len(exp_ids)) + " experiments found.") 
    print("Available experiment IDs:")
    print(exp_ids)
    print('=============================================================')
    convert_exp_tables_to_csv("/home/meliani/nni-experiments",exp_ids)