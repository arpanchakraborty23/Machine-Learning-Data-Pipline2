import os
import sys
import yaml,json
import pandas as pd
import pickle

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from pathlib import Path


def read_yaml(file_path:Path):
    try:
        with open(file_path) as f:
            file=yaml.safe_load(f)
        return file
    except Exception as e:
        print(f'error in {str(e)}')
       
import pickle

def save_obj(file_path, obj):
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


    
def load_obj(file_path):
    with open(file_path,'rb') as f:
        data=pickle.load(f)

    return data

def save_json(data,filename):
  
        with open(filename,'w') as j:
            json.dump(data,j,indent=4)