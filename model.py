from sklearn.base import BaseEstimator
import numpy as np
import pickle
import scipy
from scipy import linalg
from utils import *
from scipy.spatial.distance import pdist,squareform,cdist
import time
import numpy as np

class EMLE(BaseEstimator):
    def __init__(self, lamda1=1, lamda2 = 1, lamda3 = 1, gamma1 = 3,gamma2 = 3,gamma3 = 3,featureList = [0,310,416]):
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.lamda3 = lamda3
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3

        self.Wlist = None
        self.train_X = None
        self.train_Y = None

        self.featureList = featureList

    def get_params(self, deep=True):
        return {'lamda1': self.lamda1,
                'lamda2': self.lamda2,
                'lamda3': self.lamda3,
                'gamma1': self.gamma1,
                'gamma2': self.gamma2,
                'gamma3': self.gamma3}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def mcp_S(self,S, lamba, gama):
        Sm = np.zeros_like(S)
        for i in range(Sm.shape[0]):
            if S[i] < lamba:
                Sm[i] = 0
            elif S[i] >= lamba and gama > S[i]:
                Sm[i] = (S[i] - lamba)/(1-lamba/gama)
            else:
                Sm[i] = S[i]
        return np.diag(Sm)

    def mcp_21_E(self,E, A,lamba, gama):
        ite_s = 100
        E_c = E.transpose()
        A = A.transpose()
     
        for i_ite in range(ite_s):
            Dnorm = np.linalg.norm(E_c, axis = 1)
            D = 1 /( 2*Dnorm + 1e-12)
            Dnorm = 1 - Dnorm / gama
            Dnorm = Dnorm * (Dnorm >=0)
            D = D * Dnorm
            D = np.diag(D)
            E_old = E_c * 1
            E_c = np.matmul(np.linalg.inv(D + np.eye(D.shape[0]) * lamba), A * lamba)

            if np.linalg.norm(E_c - E_old) <1e-4 :
                break

        return E_c.transpose()



    def mcp_21_W2(self,W, A,B,C,De ,lamba, beta, gama):
        ite_s = 100
        W_c = W.transpose()
        for i_ite in range(ite_s):
            Dnorm = np.linalg.norm(W_c, axis = 1)
            D = 1 /( 2*Dnorm + 1e-12)
            Dnorm = 1 - Dnorm / gama
            Dnorm = Dnorm * (Dnorm >=0)
            D = D * Dnorm
            D = np.diag(D)
            W_old = W_c * 1

            W_c1 =  D + lamba * np.matmul(A, A.transpose()) + beta * np.matmul(C, C.transpose())
            W_c2 =  lamba * np.matmul(A, B.transpose()) + beta*np.matmul(C, De.transpose())

            W_c  = np.matmul(np.linalg.inv(W_c1), W_c2)
            if np.linalg.norm(W_c - W_old) <1e-4:
                break

        return W_c.transpose()

    def cal_mcp(self,a,gama):
        ans =0
        for i in a:
            if abs(i)>=gama:
                ans+=gama/2
            else:
                ans+= (abs(i) - i*i/2/gama)
        return ans

    def fit(self, X, Y):

        lamda1 = self.lamda1
        lamda2 = self.lamda2
        lamda3 = self.lamda3
        gama1 = self.gamma1
        gama2 = self.gamma2
        gama3 = self.gamma3 
        XX = list()
        n_views = len(self.featureList) - 1
        for i_view in range(n_views):
            XX.append(X[:, self.featureList[i_view]:self.featureList[i_view+1]])
        X = XX
 

        n_iters = 200
        niu = 1
        fac = 1.1
        niu_max =1e6
        fi = 1e-4
        Y=Y.transpose()
        HW = Y.shape[0]

        
        for i_view in range(n_views):
            X[i_view] = X[i_view].transpose()
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


        loss_all = list()
        for ite in range(n_iters):

            if ite>10:
                loss0 = np.linalg.norm(np.matmul(Wlist[0], X[0]) - Elist[0] - np.matmul(np.matmul(Wlist[0], X[0]), Z),ord = np.inf)
                loss1 = np.linalg.norm(np.matmul(Wlist[1], X[1]) - Elist[1] - np.matmul(np.matmul(Wlist[1], X[1]), Z),ord = np.inf)
                loss2 = np.linalg.norm(Z-J, ord = np.inf)
                if loss1 <fi and loss2 <fi and loss0<fi:
                    break
                if loss1 >1e10 or loss2 >1e10 or loss0>1e10:
                    break

            # update J
            SVD = Z + C2 / niu
            U, S, VH = np.linalg.svd(SVD)
            S = self.mcp_S(S, 1/niu,gama1)
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
                Wlist[i_view] = self.mcp_21_W2(Wlist[i_view], W_C,W_D, W_A, W_B, lamda3/lamda2,niu/(lamda2*2), gama2)

            # update Elist
            for i_view in range(n_views):
                E_1 = np.matmul(Wlist[i_view], X[i_view]) - np.matmul(np.matmul(Wlist[i_view], X[i_view]), Z) + C1list[i_view]/niu
                Elist[i_view] = self.mcp_21_E(Elist[i_view], E_1,niu /(lamda1*2), gama3)

            # update Mulits
            for i_view in range(n_views):
                C1list[i_view] = C1list[i_view] + niu * (np.matmul(Wlist[i_view], X[i_view]) - Elist[i_view] - np.matmul(np.matmul(Wlist[i_view], X[i_view]),Z)) 
            C2 = C2 + niu * (Z - J)
            niu = min(niu*fac, niu_max)


        self.train_Y = Y.transpose()
        for i_view in range(n_views):
            if i_view ==0:
                self.train_X = X[i_view].transpose()
                self.Wlist = Wlist[i_view].transpose()
            else:
                self.train_X = np.concatenate((self.train_X, X[i_view].transpose()), axis = 1)
                self.Wlist = np.concatenate((self.Wlist, Wlist[i_view].transpose()), axis = 0)

        return self.Wlist