import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def setup():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--X", default="", type=str, help = "Read content from the file")
    return parser.parse_args()

def solve(args):
    df = pd.read_csv(args.X, parse_dates=['id']) 


    # Prepare data
    df['year'] = [d.year for d in df.id]
    df['month'] = [d.month for d in df.id]
    years = df['year'].unique()

    plt.scatter('month', 'value', data=df.loc[df.year==2005, :], label=2005, color='red')
    plt.scatter('month', 'value', data=df.loc[df.year==2004, :], label=2004, color='blue')
    plt.scatter('month', 'value', data=df.loc[df.year==2007, :], label=2007, color='green')
    plt.scatter('month', 'value', data=df.loc[df.year==2006, :], label=2006, color='violet')
    A1=df.loc[df.year==2005, :]['month']
    A1=np.array(A1)
    print(A1)

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

if __name__ == '__main__':
    args=setup()
	solve(args)