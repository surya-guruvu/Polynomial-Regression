import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import argparse



def solve(args):
    df = pd.read_csv(args.X, parse_dates=['id']) 

    df['year'] = [d.year for d in df.id]
    df['month'] = [d.month for d in df.id]
    yearsw = df['year'].unique()
    years=[2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014]
    print(years)
    
    X1=df.loc[df.year==2005, :]['month']
    t1=df.loc[df.year==2005, :]['value']
    
    X1=np.array(X1,dtype='float64')
    t1=np.array(t1,dtype='float64')
    X1=X1/12
    t1=t1
    
    m=5
    w=pInv(X1,t1,m)
    print(w)
    P=designMatrix(m,X1)
    Y=np.matmul(P,w)
    Y=np.matmul(P,w)
    print(errorMSE(Y,t1))
    E=[]
    for i in years:
        X=df.loc[df.year==i, :]['month']
        t=df.loc[df.year==i, :]['value']
                
        X=np.array(X,dtype='float64')
        t=np.array(t,dtype='float64')
        X=X/12
        t=t
        P=designMatrix(m,X)
                    
        Ya=np.matmul(P,w)
        E.append(errorMSE(Ya,t))
    plt.plot(years,E,label='no regularisation')
    
    '''
    m=5
    lam=0.0000007
    w=pInv1(X1,t1,m,lam)
    P=designMatrix(m,X1)
    Y=np.matmul(P,w)
    Y=np.matmul(P,w)
    print(errorMSE(Y,t1))
    E=[]
    for i in years:
        X=df.loc[df.year==i, :]['month']
        t=df.loc[df.year==i, :]['value']
                
        X=np.array(X,dtype='float64')
        t=np.array(t,dtype='float64')
        X=X/12
        t=t
        P=designMatrix(m,X)
                    
        Ya=np.matmul(P,w)
        E.append(errorMSE(Ya,t))
    plt.plot(years,E,label='lam=0.0000007')

    plt.legend(loc='best')
    plt.show()
    '''

    df_test = pd.read_csv(args.Y, parse_dates=['id']) 
    df_test['month'] = [d.month for d in df_test.id]
    print(df_test['month'])
    X2=(np.array(df_test['month']))/12
    P=designMatrix(m,X2)
    Y=np.matmul(P,w)
    df_test['value']=Y
    del df_test['month']
    df_test['id']=df_test['id'].apply(lambda x: x.strftime('%#m/%#d/%y'))
    df_test.to_csv('outputF.csv',index=False)
    print(df_test)
    

    
    #errorplotSSE(Xvec,Xvec1,tvec,tvec1)
    #plt.show()
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
    return magnitude(Y1)




def magnitude(vector): 
    return (sum(pow(element, 2) for element in vector))

def ith_gradient(Xn,tn,w,m):
    Ph=basis(m,Xn)
    S=-(tn-np.dot(w,Ph))
    return S*Ph

def Sgd(Xvec,tvec,N,m,neta,batch_Size,lam,no_iterations):
    
    w=np.array([0]*(m+1))

    for i in range(no_iterations):
        R=np.random.randint(N, size=(batch_Size))
        grad_error=0
        for j in R:
            grad_error+=ith_gradient(Xvec[j],tvec[j],w,m)
        grad_error+=lam*w
        w=w-((neta)*(grad_error))
    return w




