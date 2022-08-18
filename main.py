from utils import *
import numpy as np
import time



def mcp_S(S, lamba, gama):
    Sm = np.zeros_like(S)
    for i in range(Sm.shape[0]):
        if S[i] < lamba:
            Sm[i] = 0
        elif S[i] >= lamba and gama > S[i]:
            Sm[i] = (S[i] - lamba)/(1-lamba/gama)
        else:
            Sm[i] = S[i]
    return np.diag(Sm)

def mcp_21_E(E, A,lamba, gama):
    ite_s = 100
    E_c = E.transpose()
    A = A.transpose()

    for i_ite in range(ite_s):
        loss = 0.5*np.linalg.norm(E_c-A)*np.linalg.norm(E_c-A)
        loss += lamba*cal_mcp(np.linalg.norm(E_c, axis = 1),gama)
        Dnorm = np.linalg.norm(E_c, axis = 1)
        D = 1 /( 2*Dnorm + 0.000000001)
        Dnorm = 1 - Dnorm / gama
        Dnorm = Dnorm * (Dnorm >=0)
        D = D * Dnorm
        D = np.diag(D)
        E_old = E_c * 1
        E_c = np.matmul(np.linalg.inv(2*lamba*D + np.eye(D.shape[0])), A)
        if np.linalg.norm(E_c - E_old) <1e-5 :
            break
    return E_c.transpose()



def mcp_21_W2(W, A,B,C,De ,lamba, beta, gama):
    ite_s = 1
    W_c = W.transpose()
    for i_ite in range(ite_s):
        Dnorm = np.linalg.norm(W_c, axis = 1)
        D = 1 /( 2*Dnorm + 0.000000001)
        Dnorm = 1 - Dnorm / gama
        Dnorm = Dnorm * (Dnorm >=0)
        D = D * Dnorm
        D = np.diag(D)
        W_old = W_c * 1
        W_c = np.linalg.inv(2*lamba*D + np.matmul(A, A.transpose()) +2*beta*np.matmul(C, C.transpose()))
        W_c = np.matmul(W_c, np.matmul(A, B.transpose()) + 2*beta*np.matmul(C, De.transpose()))
        if np.linalg.norm(W_c - W_old) <1e-5:
            break
    return W_c.transpose()

def cal_mcp(a,gama):
    ans =0
    for i in a:
        if abs(i)>=gama:
            ans+=gama/2
        else:
            ans+= (abs(i) - i*i/2/gama)
    return ans
# X： d*n Y: c*n
def run(X, Y, testX, testY, lamda1, lamda2, lamda3,gama):
    # alpha: weight list of multi-source data
    # regularization parameters: lamda1, lamda2, p


    n_iters = 800
    niu = 1
    fac = 1.1
    niu_max =1e6
    fi = 1e-5
    Y=Y.transpose()
    HW = Y.shape[0]
    n_views = len(X)
    Wlist = list()
    Elist = list()
    C1list = list()

    Z = np.zeros((X[0].shape[1], X[0].shape[1]))
    J = np.zeros((X[0].shape[1], X[0].shape[1]))
    C2 = np.zeros((X[0].shape[1], X[0].shape[1]))

    for i_view in range(n_views):
        Wlist.append(np.zeros((HW, X[i_view].shape[0])))
        Elist.append(np.zeros((HW, X[i_view].shape[1])))
        C1list.append(np.zeros((HW, X[i_view].shape[1])))

    for ite in range(n_iters):
        if ite>5:
            loss0 = np.linalg.norm(np.matmul(Wlist[0], X[0]) - Elist[0] - np.matmul(np.matmul(Wlist[0], X[0]), Z),ord = np.inf)
            loss1 = np.linalg.norm(np.matmul(Wlist[1], X[1]) - Elist[1] - np.matmul(np.matmul(Wlist[1], X[1]), Z),ord = np.inf)
            loss2 = np.linalg.norm(Z-J, ord = np.inf)
            # print(ite, loss1, loss2)
            if loss1 <fi and loss2 <fi and loss0<fi:
                break

        # update J
        SVD = Z + C2 / niu
        U, S, VH = np.linalg.svd(SVD)
        S = mcp_S(S,lamba = 1/niu,gama=gama)
        J = np.matmul(np.matmul(U, S), VH)

        # update Z
        Z_A = J - C2 / niu
        Z_Blist = list()
        Z_Clist = list()
        for i_view in range(n_views):
            Z_Blist.append(np.matmul(Wlist[i_view], X[i_view]))
            Z_Clist.append(Z_Blist[i_view] - Elist[i_view] + C1list[i_view]/niu)
            if i_view == 0:
                Z_1 = np.matmul(Z_Blist[i_view].transpose(), Z_Blist[i_view])
                Z_2 = np.matmul(Z_Blist[i_view].transpose(), Z_Clist[i_view])
            else:
                Z_1 += np.matmul(Z_Blist[i_view].transpose(), Z_Blist[i_view])
                Z_2 += np.matmul(Z_Blist[i_view].transpose(), Z_Clist[i_view])
        Z_1 =  np.eye(Y.shape[1]) + Z_1
        Z_2 = Z_A  +Z_2
        Z = np.matmul(np.linalg.inv(Z_1), Z_2)

        # update Wlist
        for i_view in range(n_views):
            W_A = X[i_view] - np.matmul(X[i_view], Z)
            W_B = Elist[i_view] -  C1list[i_view]/niu
            W_C = X[i_view]*1
            for inn in range(n_views):
                if inn != i_view:
                    W_D = Y - np.matmul(Wlist[inn],X[inn])
            Wlist[i_view] = mcp_21_W2(Wlist[i_view], W_A, W_B, W_C,W_D, lamda3/niu,lamda2/niu, gama)

        # update Elist
        for i_view in range(n_views):
            E_1 = np.matmul(Wlist[i_view], X[i_view]) - np.matmul(np.matmul(Wlist[i_view], X[i_view]), Z) + C1list[i_view]/niu
            Elist[i_view] = mcp_21_E(Elist[i_view], E_1,lamda1/niu, gama)

        # update Mulits
        for i_view in range(n_views):
            C1list[i_view] = C1list[i_view] + niu * (np.matmul(Wlist[i_view], X[i_view]) - Elist[i_view] - np.matmul(np.matmul(Wlist[i_view], X[i_view]),Z)) 
        C2 = C2 + niu * (Z - J)
        niu = min(niu*fac, niu_max)


    for i_view in range(n_views):
        X[i_view] = X[i_view].transpose()
        testX[i_view] = testX[i_view].transpose()
        Wlist[i_view] = Wlist[i_view].transpose()

    ans = evaluate_all(trainX,Y.transpose(),testX,testY, Wlist,  no_fs_eachmodal = True, no_fs_allmodal=True, fs_eachmodal = True, fs_allmodal=True)
    return ans



X, Y = load_data_ADNI2('ADNI2_MRI_PET', one_hot=True, selet_label=[dis0, dis1], transpose=True)

res = run(trainX, trainY, testX, testY,1,1,1,3)




