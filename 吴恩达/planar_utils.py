import matplotlib.pyplot as plt 
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

def plot_decision_boundary(model,x,y):
    # find min and max and give it some padding
    x_min,x_max=x[0,:].min()-1,x[0,:].max()+1
    y_min,y_max=x[1,:].min()-1,x[1,:].max()+1
    h=0.01
    # generate grid with distance h between points
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

    Z=model(np.c_[xx.ravel(),yy.ravel()]) # model need cols of [x,y]
    Z=Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral)
    plt.xlabel("x1")
    plt.ylabel('x2')
    plt.scatter(x[0,:],x[1,:],c=y.ravel(),cmap=plt.cm.Spectral)
    plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

# help(np.meshgrid)

# xx=np.array([[1,2,3],[1,2,3]])
# yy=np.array([[1,1,1],[2,2,2]])
# Z  = np.array([[0,1,2],[1,2,3]])
# plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral)
# plt.colorbar()
# plt.show()

# print(np.c_[xx.ravel(),yy.ravel()])

# x=np.array([1,2,3,4])
# x=x.reshape((2,2))
# print(x)

# plt.scatter([1,2,3],[4,5,6],c=[1,5,10],cmap='Spectral')
# plt.colorbar()
# plt.show()

def load_planar_dataset(m):
    np.random.seed(1)
   # m=500 # examples 
    N=int(m/2)  # examples per class
    D=2 
    X=np.zeros((m,D))
    Y=np.zeros((m,1))
    a=4 # max ray of flower
    
    for j in range(2):
        ix=range(N*j,N*(j+1))
       # 角度,随机生成0-180度的角度;t=均匀取点+噪声(01正态分布)
        t=np.linspace(j*3.14,(j+1)*3.14,N)+np.random.randn(N)*0.2
       #半径
        r=a*np.sin(4*t)+np.random.randn(N)*0.2
        X[ix]=np.c_[r*np.sin(t),r*np.cos(t)]
        Y[ix]=j

    X=X.T
    Y=Y.T

    return X,Y

# X,Y=load_planar_dataset()
# plt.scatter(X[0,:],X[1,:],c=Y,cmap=plt.cm.Spectral)
# plt.show()

def load_extra_datasets():
    N=200
    noisy_circles=sklearn.datasets.make_circles(N,factor=.5,noise=.3)
    noisy_moons=sklearn.datasets.make_moons(N,noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure=np.random.rand(N,2),np.random.rand(N,2)

    return noisy_circles,noisy_moons,blobs,gaussian_quantiles,no_structure
