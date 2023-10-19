import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trainer_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting the data into training and testing')
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'LinearRegression': LinearRegression(),
                'K-Neighbors': KNeighborsRegressor(),
                'Catboosting': CatBoostRegressor(verbose=False),
                'AdaBoost': AdaBoostRegressor()
            }
            model_report=evaluate_model(X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test, models=models)
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            if(best_model_score<0.6):
                raise CustomException('No best model found')
            
            logging.info('Found the best model')
            
            save_object(
                file_path=self.model_trainer.trainer_model_file_path,
                obj=best_model
            )
            
            predicted_output=best_model.predict(X_test)
            
            score=r2_score(y_test, predicted_output)
            return score
            
        except Exception as e:
            raise CustomException(e,sys)
