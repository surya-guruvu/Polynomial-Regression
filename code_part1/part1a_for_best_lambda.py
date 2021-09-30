import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


def setup():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--X", default="", type=str, help = "Read content from the file")
    return parser.parse_args()
def solve(args):

    data = np.loadtxt(args.X, dtype=str) 
    n=len(data);

    Xvec= [0 for i in range(n)]
    tvec= [0 for i in range(n)]  

    for i in range(n):
        l= data[i].split(',')
        Xvec[i]=l[0]
        tvec[i]=l[1]

    Xvec=np.array(Xvec,dtype='float64')
    tvec=np.array(tvec,dtype='float64')

    #train set
    Xvec1=Xvec[0:20]
    tvec1=tvec[0:20]

    #Validation set

    Xvec2=Xvec[20:100]
    tvec2=tvec[20:100]

    
    lam= [0,1e-9,1e-8,1e-7,0.000001,0.00001,0.0001,0.001,0.01,0.1,1]
    lamErr_train=[]
    lamErr_test=[]
    for i in lam:
        w1=pInv1(Xvec1,tvec1,6,i)
        P1=designMatrix(6,Xvec1)
        Y1=np.matmul(P1,w1)
        lamErr_train.append(errorRSE(Y1,tvec1))
        P2=designMatrix(6,Xvec2)
        Y2=np.matmul(P2,w1)
        lamErr_test.append(errorRSE(Y2,tvec2))

    print("Max lamb",lam[lamErr_test.index(max(lamErr_test))])
    plt.plot(lam,lamErr_train,label='train')
    plt.plot(lam,lamErr_test,label='test')
    plt.xlabel('lam');
    plt.ylabel('MSE');

    plt.legend(loc='best')
    plt.show()
def basis(m,Xfea):
    l=[0 for i in range(m+1)]
    for j in range(m+1):
        l[j]=(Xfea)**j
    return np.array(l)


def designMatrix(m,X): #Polynomial curve fitting
    P=[]
    n=len(X)

    for i in range(n):
        l=[0 for i in range(m+1)]
        for j in range(m+1):
            l[j]=(X[i])**j
        P.append(l)
    return np.array(P,dtype='float64')

def pInv(X,t,m):
    P=designMatrix(m,X) 
    P4=np.linalg.pinv(P)
    w=P4.dot(t)
    return np.array(w)

def pInv1(X,t,m,lam):
    P=designMatrix(m,X)
    I=lam*(np.eye(m+1))
    P2=(np.linalg.inv(I+P.T.dot(P))).dot(P.T)
    w=P2.dot(t)
    return np.array(w)


def noise_variance(Y,t,N):
    Y1=(Y-t)/N 
    return math.sqrt(sqmagnitude(Y1))



def errorMSE(Y,t):
    Y1=(Y-t)/len(Y)
    return sqmagnitude(Y1)
def errorRSE(Y,t):
    Ymean=(sum(element for element in Y))/(len(Y))
    Y1=[1]*len(Y)
    Y1=np.array(Y1)
    Y1=Ymean*Y1
    R_sq=1-((len(Y)*errorMSE(Y,t))/sqmagnitude(Y-Y1))
    return R_sq
def absError(Y,t):
    Y1=(Y-t)/(len(Y))
    Y1=np.absolute(np.array(Y1))


def sqmagnitude(vector): 
    return (sum(pow(element, 2) for element in vector))

def ith_gradient(Xn,tn,w,m):
    Ph=basis(m,Xn)
    S=-(tn-np.dot(w,Ph))
    return S*Ph
def Sgd(Xvec,tvec,N,m,neta,batch_Size,lam,tolerance):
    
    w=np.array([0]*(m+1))
    P=designMatrix(m,Xvec)
    Er=errorMSE(np.matmul(P,w),tvec)
    tolerance=1e-15;
    while True:
        R=np.random.randint(N, size=(batch_Size))
        grad_error=0
        for j in R:
            grad_error+=ith_gradient(Xvec[j],tvec[j],w,m)
        grad_error+=lam*w
        w=w-((neta)*(grad_error))
        Er1=errorMSE(np.matmul(P,w),tvec)
        if abs(Er1-Er)<=tolerance:
            break;
        Er=Er1

    return w

if __name__ == '__main__':
    args = setup()
    solve(args)
