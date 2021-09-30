import argparse
import numpy as np
import math
import matplotlib.pyplot as plt



def setup():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--X", default="", type=str, help = "Read content from the file")
    return parser.parse_args()
def solve2(args):
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
    Xvec1=Xvec[0:30]
    tvec1=tvec[0:30]

    #Validation set

    Xvec2=Xvec[30:100]
    tvec2=tvec[30:100]
    
    train_error=[]
    test_error=[]

    M=[]

    '''
    for i in range(0,21):
        M.append(i)
        P1=designMatrix(i,Xvec2)
        w1=pInv(Xvec1,tvec1,i)
        Y1=np.matmul(P1,w1)
        test_error.append(errorRSE(Y1,tvec2))
        P2=designMatrix(i,Xvec1)
        Y2=np.matmul(P2,w1)
        train_error.append(errorRSE(Y2,tvec1))

    plt.plot(M,train_error,label= 'train')
    plt.plot(M,test_error,color='green',label='test')
    plt.xlabel('m');
    plt.ylabel('RSE');
    print("Degree for min error on Training set::",train_error.index(max(train_error)))
    print("Degree for min error on Testing set::",test_error.index(max(test_error)))

    plt.legend(loc ="best")
    
    
    '''
    m=8
    P1=designMatrix(m,Xvec)
    w1=pInv(Xvec1,tvec1,m)

    Y1=np.matmul(P1,w1)
    print(-Y1+tvec)
    plt.hist(-Y1+tvec)

    print(errorMSE(Y1,tvec))
    
    '''
    
    lamErr_train=[]
    lamErr_test=[]
    #for i in lam:
    w1=pInv1(Xvec1,tvec1,8,1e-7)
    #w1=Sgd(Xvec1,tvec1,len(Xvec1),8,0.0000001,10,0,1000000)
    P1=designMatrix(8,Xvec1)
    Y1=np.matmul(P1,w1)
    #lamErr_train.append(errorMSE(Y1,tvec1))
    print(errorMSE(Y1,tvec1))'''
    '''
    '''
    '''P2=designMatrix(8,Xvec2)
    Y2=np.matmul(P2,w1)
    print(errorMSE(Y2,tvec2))'''
    #lamErr_test.append(errorMSE(Y2,tvec2))

    '''print("Min lamb",lam[lamErr_test.index(min(lamErr_test))])
    plt.plot(lam,lamErr_train,label='train')
    plt.plot(lam,lamErr_test,label='test')
    plt.xlabel('lam');
    plt.ylabel('MSE'); 
    plt.show()'''
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
    solve2(args)