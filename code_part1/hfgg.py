"""
Demo to show importing function
"""
import numpy as np
import math
import matplotlib.pyplot as plt

def solve(args):
    #print("foo imported successfully")
    #print(args)
    #k = 10
    #weights = np.random.random(k+1)
    #print(f"Polynomial={k}")
    #print(f"weights={weights}")

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

    #20 points at the beginning

    Xvec1=Xvec[0:20].transpose()
    tvec1=tvec[0:20].transpose()
    plt.scatter(Xvec1,tvec1,color='red',s=10)    


    #design matrix
    m=int(args.polynomial)

    #P=designMatrix(m,Xvec1)
    #print(P)
    #print(P.transpose())

    #paramter
    if(args.method=="pinv"):

        w=pInv(Xvec1,tvec1,args.lamb,int(args.polynomial))
        '''h3=[]
        h4=[]
        m1=[]
        for i in range(1,21):
            m1.append(i)
            w=pInv(Xvec1,tvec1,args.lamb,i)
            P=designMatrix(i,Xvec1)
            P4=designMatrix(i,Xvec)
            Y=np.matmul(P,w);
            Y4=np.matmul(P4,w)
            h3.append(erroSSE(Y,tvec1))
            h4.append(erroSSE(Y4,tvec))
        plt.plot(m1,h3,color='green')
        plt.plot(m1,h4,color='yellow')'''

        #for different lambdas.
        '''lam= np.linspace(0,1,5000)
        h1=[]
        for i in lam:
            w=pInv(Xvec1,tvec1,i,m)
            print(w)
            P=designMatrix(m,Xvec1)
            Y=np.matmul(P,w)
            h1.append(erroSSE(Y,tvec1))
        plt.plot(lam,h1)   
        h2=[]
        for i in lam:
            w=pInv(Xvec,tvec,i,m)
            print(w)
            P=designMatrix(m,Xvec)
            Y=np.matmul(P,w)
            h2.append(erroSSE(Y,tvec))
        plt.plot(lam,h2)   '''     
    
    P=designMatrix(m,Xvec1)
    #plotting the graph
    x= np.linspace(-2,2,1000)
    m=int(args.polynomial)
    tout=[]
    for i in range(len(x)):
        x1=basis(m,x[i])
        tout.append(np.dot(w,x1))

    plt.plot(x,tout)
        
    
    #finding the output for our twenty points taken as input
    Y=np.matmul(P,w)
    print(erroSSE(Y,tvec1))
    #errorplotSSE(Xvec,Xvec1,tvec,tvec1)
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

def pInv(X,t,lam,m):
    P=designMatrix(m,X)
    P1=P.transpose()
    P2=np.matmul(P1,P)
    I=(np.identity(m+1))*lam
    P3=np.linalg.inv(P2+I)
    P4=np.matmul(P3,P1)
    w=np.matmul(P4,t)

    return np.array(w)

def erroSSE(Y,t):
    Y1=(Y-t)/2
    return magnitude(Y1)




def magnitude(vector): 
    return math.sqrt(sum(pow(element, 2) for element in vector))
