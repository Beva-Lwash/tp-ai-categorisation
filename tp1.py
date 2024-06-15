import csv
import pandas as pd

def loadData(file):
    return pd.read_csv(file, header=0)

dataset_api_results = loadData('lovoo_v3_users_api-results.csv')
dataset_instances = loadData('lovoo_v3_users_instances.csv')

df = pd.merge(dataset_api_results, dataset_instances, on="userId") 
print(df) 
        