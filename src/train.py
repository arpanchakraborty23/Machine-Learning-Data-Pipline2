import pandas as pd
import mlflow
import os
import pickle
from dotenv import load_dotenv
from urllib.parse import urlparse
from src.utils import read_yaml, save_obj
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from mlflow.models import infer_signature

# Load environment variables
load_dotenv()
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

def hyperparameter_tuning(x_train, y_train, param_grid):
    try:
        model = RandomForestClassifier()
        gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=3)
        gs.fit(x_train, y_train)
        
        print(f'Best params: {gs.best_params_}')
        print(f'Best score: {gs.best_score_}')
        return gs, model
    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")

# Load parameters
params = read_yaml('params.yaml')['train']

def train(data_path, model_path, random_state, n_estimators, max_depth):
    try:
        data = pd.read_csv(data_path)
        x = data.drop('Outcome', axis=1)
        y = data['Outcome']

        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

        # Start mlflow
        with mlflow.start_run():
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=random_state)

            sig = infer_signature(x_train, y_train)
            param_grid = {
                'n_estimators': [50, 100],  
                'max_depth': [None, 10, 20],  
                'min_samples_split': [2,7, 5]
            }

            # Hyperparameter Tuning
            gs, model = hyperparameter_tuning(x_train, y_train, param_grid)
            best_model = model.set_params(**gs.best_params_)

            # Predict and evaluate model
            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            # Log metrics
            mlflow.log_metric('Accuracy', accuracy)
            mlflow.log_text(report, 'classification_report.txt')

            # Log best parameters
            mlflow.log_param('n_estimators', gs.best_params_['n_estimators'])
            mlflow.log_param('max_depth', gs.best_params_['max_depth'])
            mlflow.log_param('min_samples_split', gs.best_params_['min_samples_split'])

            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            mlflow.log_text(str(cm), 'confusion_matrix.txt')

            tracking_uri_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_uri_type_store != 'file':
                mlflow.sklearn.log_model(best_model, 'model', registered_model_name='best_model')
            else:
                mlflow.sklearn.log_model(best_model, 'model', signature=sig)

            # Ensure directory exists before saving model
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            save_obj(model_path, best_model)
            print(f'Model saved to {model_path}')

    except Exception as e:
        print(f"Error in training process: {e}")

if __name__ == "__main__":
    train(
        data_path=params['data'],
        model_path=params['model'],
        random_state=params['random_state'],
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth']
    )
