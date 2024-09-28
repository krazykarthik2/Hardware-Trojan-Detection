from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
from preprocess_data import prepare_data,prepare_predict_data
def predict(model):
    """
    This function performs classification with logistic regression.
    """
    # load model
    with open(model, 'rb') as f:
        clf = pickle.load(f)

    input__ = prepare_predict_data()
    
    # make direct params to refined 
    # predict
    # params are like 5135800 56076 45553 38942 5677500 90170
    # refined_struct is like [5135800, 56076, 45553, 38942, 5677500, 90170, ]
    

    with open('scalers/minmax_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    y_pred = clf.predict(input__)

    print("outputs are ",y_pred)


    return y_pred