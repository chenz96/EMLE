import numpy as np
import pickle
import scipy
from scipy import linalg
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import scipy
import os
from sklearn import metrics

def evaluate_cls(y_truth, y_pred, prob):
    auc = metrics.roc_auc_score(y_truth, prob)
    acc = metrics.accuracy_score(y_truth, y_pred)
    tn, fp, fn, tp = metrics.confusion_matrix(y_truth, y_pred).ravel()
    #print(tn, fp, fn, tp)
    sen = tp / (tp+fn)
    spe = tn / (tn+fp)
    return acc, sen, spe, auc

def evaluate_all(trainX,trainY,testX,testY, W, onemodel=False,latentspace = False, P = None, no_fs_eachmodal = False, no_fs_allmodal=False, fs_eachmodal = False, fs_allmodal=False):


    # W={W_i},trainX={trainX_i = R{n*di}},trainY=R{n*c}
    # ans1: no feature selection   Y = trainX * W
    # ans2:n-1: select features from each view and then train SVM with the selected features
    # select features from all views and then train SVM with the selected features


    n_views = len(testX)


    if onemodel == True:
        Wlist = list()
        curtd=0

        for i_view in range(n_views):
            curd = trainX[i_view].shape[1]
            Wlist.append(W[curtd:curtd+curd,:])
            curtd +=curd
        W = Wlist


    for i_view in range(n_views):
        if i_view ==0:
            AX = trainX[i_view]
            AXT = testX[i_view]
            AW = W[i_view]
        else:
            AX = np.concatenate((AX, trainX[i_view]), axis = 1)
            AXT = np.concatenate((AXT, testX[i_view]), axis = 1)
            AW =np.concatenate((AW, W[i_view]), axis = 0)

    ans_return  =list()

    if no_fs_eachmodal == True:
        ansmodel = list()
        if latentspace == False:
            for i_view in range(n_views):
                preds = np.matmul(testX[i_view],W[i_view])
                prob = scipy.special.softmax(preds, axis = 1)[:,1]
                ans1 = evaluate_cls(np.argmax(testY, axis=1), np.argmax(preds, axis=1), prob)
                ansmodel.append(np.array(ans1))

        else:
            for i_view in range(n_views):
                H = np.matmul(testX[i_view],W[i_view])
                preds = np.matmul(H, P)
                prob = scipy.special.softmax(preds, axis = 1)[:,1]
                ans1 = evaluate_cls(np.argmax(testY, axis=1), np.argmax(preds, axis=1), prob) 
                ansmodel.append(np.array(ans1))
        ans_return.append(ansmodel)

    if no_fs_allmodal == True:
        if latentspace == False:
            H = np.zeros((testX[0].shape[0], trainY.shape[1]))
            for i_view in range(n_views):
                H += np.matmul(testX[i_view],W[i_view])
            preds = H/n_views
            prob = scipy.special.softmax(preds, axis = 1)[:,1]
            ans1 = evaluate_cls(np.argmax(testY, axis=1), np.argmax(preds, axis=1), prob)
            ans_return.append(np.array(ans1))

        else:
            H = np.zeros((testX[0].shape[0], P.shape[0]))
            for i_view in range(n_views):
                H += np.matmul(testX[i_view],W[i_view])
            preds = np.matmul(H/n_views, P)
            prob = scipy.special.softmax(preds, axis = 1)[:,1]
            ans1 = evaluate_cls(np.argmax(testY, axis=1), np.argmax(preds, axis=1), prob) 
            ans_return.append(np.array(ans1))

    if fs_eachmodal == True:
        ansSVM = list()
        for i_view in range(n_views):
            tempans = train_svm_for_ea(trainX[i_view],trainY,testX[i_view],testY,W[i_view])
            ansSVM.append(np.array(tempans))
        ans_return.append(ansSVM)

    if fs_allmodal == True:
        ans_svm_allViews = train_svm_for_ea(AX,trainY,AXT,testY,AW)
        ans_return.append(np.array(ans_svm_allViews))

    return ans_return

def train_svm_for_ea(train_X, train_Y, test_X, test_Y, W,select_num_list=[0],kernel = 'linear'):
    if select_num_list[0]==0:
        select_num_list = [2, 4, 6, 8,10,12,14,16,18,20]


    normv = np.linalg.norm(W,axis=1)
    sort_list = np.sort(normv)
    anslist = list()
    for i_prob, select_prop in enumerate(select_num_list):
        cutv = sort_list[W.shape[0]-select_prop]
        selected_index = np.where(normv>=cutv )[0]
        train_X_svm = train_X[:,selected_index]
        test_X_svm = test_X[:,selected_index]

        svc = SVC(kernel = kernel,probability=True)
        svc.fit(train_X_svm, np.argmax(train_Y,axis=1))
        preds = svc.predict(test_X_svm)
        prob = svc.predict_proba(test_X_svm)[:,1]
        anslist.append(evaluate_cls(np.argmax(test_Y,axis=1), preds ,prob))

    return anslist




def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_data_ADNI2(path,one_hot=False, selet_label=[], transpose= False):
    # load ADNI2 data : MRI + PET + diagnosis, preprocessed by ADNI group, dict: "MRI", "PET", "ID", "dis"
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



if __name__ == '__main__':
    X,Y = load_data_ADNI2('ADNI2_MRI_PET',one_hot=False, selet_label=[1], transpose= False)
    print(Y.shape)