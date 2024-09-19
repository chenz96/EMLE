import numpy as np
import pickle
import os


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_data_ADNI(path,one_hot=False, selet_label=[], transpose= False):
    # load ADNI data : MRI + PET + diagnosis, preprocessed by ADNI group, dict: "MRI", "PET", "ID", "dis"
    # transpose == True if X = d*n
    # return X, Y
    #   X= list(): X[0] = MRI, X[1] = PET
    data = load_obj(path)

   
    mriData = np.array(data['MRI'])
    petData = data['PET']
    labels = data['dis']

    if len(selet_label)!=0:
        selected_index = list()
        for p in selet_label:
            selected_index.append( np.where(labels==p))
        selected_index = np.hstack(selected_index).reshape(-1)
        #print(selected_index)
        mriData = mriData[selected_index,:]
        petData = petData[selected_index,:]
        labels = labels[selected_index]

    X=list()
    if transpose == True:
        X.append(mriData.transpose())
        X.append(petData.transpose())
    else:
        X.append(mriData)
        X.append(petData)

    Y = labels -1
    yk = np.unique(Y)
    YL = np.zeros([Y.shape[0],yk.shape[0]])

    if one_hot == True:
        for i, p in enumerate(Y):
            YL[i,np.where(yk==p)]=1
        Y = YL
    return X, Y


