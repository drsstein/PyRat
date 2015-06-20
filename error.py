import numpy as np

def squared_error(targets, predictions):
    #estimate error derivative dE/dy
    dEdy = targets - predictions
    #error
    rmse = np.sqrt(np.dot(dEdy,dEdy.T)/len(dEdy[0]))
    return rmse, dEdy
