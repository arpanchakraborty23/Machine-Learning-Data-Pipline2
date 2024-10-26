import pandas as pd
import mlflow
import os
from dotenv import load_dotenv
from src.utils import read_yaml, load_obj
from sklearn.metrics import accuracy_score

# Load environment variables
load_dotenv()
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

# Load parameters
params = read_yaml('params.yaml')['train']

def evaluate(data_path, model_path):
    try:
        # Load dataset
        data = pd.read_csv(data_path)
        x = data.drop('Outcome', axis=1)
        y = data['Outcome']

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        
        # Load model
        model = load_obj(model_path)
        
        # Start MLflow run context
        with mlflow.start_run():
            # Model prediction and accuracy calculation
            prediction = model.predict(x)
            accuracy = accuracy_score(y, prediction)
            
            # Log metrics
            mlflow.log_metric('Accuracy', accuracy)
            print(f'Model accuracy: {accuracy}')
    
    except Exception as e:
        print(f"Error in evaluation process: {e}")

if __name__ == "__main__":
    evaluate(data_path=params['data'], model_path=params['model'])
