import sys
from model import *
import numpy as np

if __name__=='__main__':
    dis0 = 1 
    dis1 = 3
    X, Y = load_data_ADNI('ADNIGO23_MRI_PET', one_hot=True, selet_label=[dis0, dis1], transpose=False)
    X = np.concatenate(X, axis = 1)

    model = EMLE()
    Wlist = model.fit(X, Y)
    print(Wlist.shape)
    normv = np.linalg.norm(Wlist,axis=1)
    featureImportance = np.argsort(normv)

    print(featureImportance)