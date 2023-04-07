import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self) :
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Training a model has started")
        try:
            logging.info("split training and testing input data")
            X_train, y_train, X_test,y_test=(train_array[:,:-1], train_array[:,-1],
                                             test_array[:,:-1], test_array[:,-1])
            
            # dictionary of models to try
            models = {  "Random Forest": RandomForestRegressor(),
                        "Linear Regression": LinearRegression(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Gradient Boosting":GradientBoostingRegressor(),
                        "K-Neighbors Regressor": KNeighborsRegressor(),             
                        "XGBRegressor": XGBRegressor(), 
                        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                        "AdaBoost Regressor": AdaBoostRegressor()  }

            params={
                "Random Forest":{"n_estimators":[8,16,32,64,128,256]},
                "Linear Regression":{},
                "Decision Tree":{"criterion":["squared_error","friedman_mse","absolute_error","poisson"]},
                "Gradient Boosting":{"learning_rate":[0.1,0.01,0.05,.001],
                                     "subsample":[0.6,0.7,0.75,0.8,0.85,0.9],
                                     "n_estimators":[8,16,32,64,128,256]},
                "K-Neighbors Regressor":{"n_neighbors":[6,8,10]},
                "XGBRegressor":{"learning_rate":[0.1,0.01,0.05,.001],"n_estimators":[8,16,32,64,128,256]},
                "CatBoosting Regressor":{"depth":[3,5,7,9],
                                         "learning_rate":[0.1,0.01,0.05,.001],
                                         "iterations":[30,50,100]},
                "AdaBoost Regressor":{"learning_rate":[0.1,0.01,0.05,.001],"n_estimators":[8,16,32,64,128,256]}
                    }

            #
            # model evaluation
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,
                                             X_test=X_test, y_test=y_test, models=models,params=params)
            
            # get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            # best model name from the index of the best model score
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            # best model
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            
            logging.info("Best found model on both training and testting datasets")

            # save best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            
            logging.info("Training a model has Finished")
            return r2_square


        except Exception as e:
            raise CustomException(e,sys)
        
       