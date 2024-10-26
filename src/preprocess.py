import pandas as pd
import sys
import yaml
import os
from src.utils import read_yaml

# Load Params from yaml file
params = read_yaml('params.yaml')['preprocess']

def preprocess(input_path, output_path): 
    try:
        # Load data
        data = pd.read_csv(input_path)
        print(data.head())

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save preprocessed data
        data.to_csv(output_path, index=False)
        print(f'Preprocess data saved to {output_path}')

    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    preprocess(params['input'], params['output'])
