import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import   plot_decision_boundary,sigmoid,load_extra_datasets,load_planar_dataset


np.random.seed(1)




X,Y=load_planar_dataset(m=400)

plt.scatter(X[0,:],X[1,:],c=Y.ravel(),s=40,cmap=plt.cm.Spectral)
#plt.show()

# print(X.shape)
# print(Y.shape)
# print(type(X.shape[0]))

# clf=sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T,Y.ravel())
# print(type(clf))
# plot_decision_boundary(lambda x:clf.predict(x),X,Y)
# plt.title("logistic regression")

# accuracy
# LR_predictions=clf.predict(X.T)
# accuracy=np.mean(LR_predictions==Y.ravel())
# print("accuracy:",accuracy)
#拉完了  0.47 

# x:2*m 
# z1=w1x+b1 z:4*m  w1:4*2
# a1=tanh(z1)  4*m
# z2=w2a1+b2  1*m  w2:1*4
# a2=sigmoid(z2) 1*m
# y^=a2
# prediction=1 if a2>0.5 else 0
# cost=-1/m求和(yloga2+(1-y)log(1-a2))


def initial_params(n_h):
    w1=np.random.randn(n_h,2)
    b1=0
    w2=np.random.randn(1,n_h)
    b2=0
    params={"w1":w1,"w2":w2,"b1":b1,"b2":b2}
    return params

#print(type(np.random.randn(4,2))) 
# numpy array

def train(X,Y,params,learning_rate):
    w1=params["w1"]
    w2=params["w2"]
    b1=params["b1"]
    b2=params["b2"]

    for i in range(1000):
        m=X.shape[1] 
        z1=np.dot(w1,X)+b1
        a1=np.tanh(z1)
        z2=np.dot(w2,a1)+b2
        a2=sigmoid(z2)


        cost=(-1/m)*np.sum(Y*np.log(a2)+(1-Y)*np.log(1-a2))

        dz2=a2-Y
        dw2=np.dot(dz2,a1.T)/m
        db2=np.sum(dz2)/m
        da1=np.dot(w2.T,dz2) # 4*m=4*1  1*m
        dz1=da1*(1-np.power(a1,2)) # 
        dw1=np.dot(dz1,X.T)/m
        db1=(1/m)*np.sum(dz1)

        w2=w2-learning_rate*dw2
        w1=w1-learning_rate*dw1
        b2=b2-learning_rate*db2
        b1=b1-learning_rate*db1
    params={"w1":w1,"w2":w2,"b1":b1,"b2":b2}
    Y_pre=(a2>0.5).astype(int)
    accuracy=np.mean(Y==np.ravel(Y_pre))
    return params,accuracy



params=initial_params(n_h=5)
params,acc=train(X,Y,params,0.01)
print(acc)
    


def predict(X,params):
    w1=params["w1"]
    w2=params["w2"]
    b1=params["b1"]
    b2=params["b2"]
    z1=np.dot(w1,X)+b1
    a1=np.tanh(z1)
    z2=np.dot(w2,a1)+b2
    a2=sigmoid(z2)

    Y=(a2>0.5).astype(int)

    return Y

plot_decision_boundary(lambda x:predict(x.T,params),X,Y)


