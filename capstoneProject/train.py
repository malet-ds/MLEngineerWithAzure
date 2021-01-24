
import os
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from azureml.core.run import Run
from azureml.core import Dataset

def create_datasets(x,y):
    columns_x = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
    x_df = pd.DataFrame(x,columns=columns_x)  
    columns_y = ['MedHouseVal']
    y_df = pd.DataFrame(y,columns=columns_y)  
    
    x_tr,x_test,y_tr,y_test = train_test_split(x_df,y_df, test_size = 20, random_state=0)

    return x_tr,x_test,y_tr,y_test

if __name__ == '__main__':

    x_cal,y_cal = fetch_california_housing(return_X_y=True)
    x,x_test,y,y_test = create_datasets(x_cal,y_cal)
    x_train,x_valid,y_train,y_valid = train_test_split(x,y,test_size=0.20, random_state=0)
          
    run = Run.get_context()

    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=float, default=0.0001, help="Constant that multiplies the regularization term. The higher the value, the stronger the regularization.")
    parser.add_argument('--l1_ratio', type=float, default=0.15, help="The Elastic Net mixing parameter. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.")
    parser.add_argument('--eta0', type=float, default=0.01, help="The initial learning rate.")
    parser.add_argument('--power_t', type=float, default=0.25, help="The exponent for inverse scaling learning rate.")

    

    args = parser.parse_args()

    run.log("Alpha", np.float(args.alpha))
    run.log("L1 Ratio", np.float(args.l1_ratio))
    run.log("Eta0", np.float(args.eta0))
    run.log("Power t", np.float(args.power_t))

    model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=10000, 
                                                        penalty='elasticnet', 
                                                        alpha=args.alpha,
                                                        l1_ratio=args.l1_ratio,
                                                        eta0=args.eta0,
                                                        power_t=args.power_t, 
                                                        early_stopping=True,
                                                        random_state=0, 
                                                        validation_fraction=0.2, 
                                                        shuffle=True))
    model.fit(x_train, y_train)

    accuracy = model.score(x_valid,y_valid)
    run.log("Accuracy", np.float(accuracy))




