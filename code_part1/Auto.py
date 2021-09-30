"""
Demo to show importing function
"""
import numpy as np
#import math

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
   


    m=int(args.polynomial)


    #paramter
    if(args.method=="pinv"):
        if(args.lamb==0):
            w=pInv(Xvec,tvec,m) 
        else:
            w=pInv1(Xvec,tvec,m,args.lamb)   

    if(args.method=="gd"):
        B_s=args.batch_size
        lam=args.lamb
        w=Sgd(Xvec,tvec,len(Xvec),m,0.001,B_s,lam) 
        
    print(f"weights={w}")
    

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
    P=designMatrix(m,X) # 20*11 matrix
    #P2=(np.linalg.inv(P.T.dot(P))).dot(P.T)
    P4=np.linalg.pinv(P)
    w=P4.dot(t)
    return np.array(w)

def pInv1(X,t,m,lam):
    P=designMatrix(m,X)
    I=lam*(np.eye(m+1))
    P2=(np.linalg.inv(I+P.T.dot(P))).dot(P.T)
    w=P2.dot(t)
    return np.array(w)

def errorMSE(Y,t):
    Y1=(Y-t)/len(Y)
    return sqmagnitude(Y1)
def errorRSE(Y,t):
    Ymean=(sum(element for element in Y))/(len(Y))
    Y1=[1]*len(Y)
    Y1=np.array(Y1)
    Y1=Ymean*Y1
    R_sq=1-((len(Y)*errorMSE(Y,t))/sqmagnitude(Y-Y1))

def sqmagnitude(vector): 
    return (sum(pow(element, 2) for element in vector))

def ith_gradient(Xn,tn,w,m):
    Ph=basis(m,Xn)
    S=-(tn-np.dot(w,Ph))
    return S*Ph
def Sgd(Xvec,tvec,N,m,neta,batch_Size,lam):
    w=np.array([0]*(m+1))
    P=designMatrix(m,Xvec)
    Er=((errorMSE(np.matmul(P,w),tvec))*N)/2
    iteration=0
    tolerance=1e-14;
    while True:
        R=np.random.randint(N, size=(batch_Size))
        grad_error=0
        for j in R:
            grad_error+=ith_gradient(Xvec[j],tvec[j],w,m)
        grad_error+=lam*w
        iteration+=1
        w=w-((neta)*(grad_error))
        Er1=((errorMSE(np.matmul(P,w),tvec))*N)/2
        if abs(Er1-Er)<=tolerance or iteration>500000:
            break;
        Er=Er1

    return w