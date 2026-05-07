from functools import cache
import numpy as np

def sigmoind(Z):
    return 1/(1+np.exp(-Z))

# 任务: 分类二维平面的点
# 数据量m  维度 n=2  x:2*m y:1*m y=0 or 1
# z=wx+b  a=f(z)

def layer_sizes(X,Y):
    n_x=X.shape[0]
    n_h=4
    n_y=Y.shape[0]
    return n_x,n_h,n_y

def initialize_param(n_x,n_h,n_y):
    np.random.seed(2)
    w1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    # n_h*1
    w2=np.random.randn((n_y,n_h))*0.01
    b2=np.zeros((n_y,1))

    params={
        "w1":w1,
        "w2":w2,
        "b1":b1,
        "b2":b2
    }
    return params

def propagation(X,params):
    w1=params["w1"]
    w2=params["w2"]
    b1=params["b1"]
    b2=params["b2"]

    z1=np.dot(w1,X)+b1
    a1=np.tanh(z1)
    # a1 : n_h*m
    # w2:  1*n_h
    z2=np.dot(w2,a1)+b2
    a2=sigmoind(z2)

    cache={
        "z1":z1,
        "z2":z2,
        "a1":a1,
        "a2":a2
    }

    return a2,cache
def compute_cost(a2,Y,params):
    m=Y.shape[1]

    logprobs=np.multiply(np.log(a2),Y)+np.multiply(np.log(1-a2),1-Y)
    cost=-(np.sum(logprobs))/m

    cost=np.squeeze(cost)

    return cost

def backward_propagation(params,cache,X,Y):
    m=X.shape[1]
    w1=params["w1"]
    w2=params["w2"] # 1*n_h
    a1=cache["a1"] # n_h*m
    a2=cache["a2"]
    dz2=a2-Y # 1*m
    dw2=(1/m)*np.dot(dz2,a1.T)
    db2=(1/m)*np.sum(dz2,axis=1,keepdims=True)

    dz1=np.dot(w2.T,dz2)*(1-np.power(a1,2))
    dw1=(1/m)*np.dot(dz1,X.T)
    db1=(1/m)*np.sum(dz1,axis=1,keepdims=True)
     
    grads={
        "dw1":dw1,
        "db1":db1,
        "dw2":dw2,
        "db2":db2
    }
    
    return grads

def update_params(params,grads,learning_rate=1.2):
    w1=params["w1"]
    b1=params["b1"]
    w2=params["w2"]
    b2=params["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    w1 = w1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2
    }

    return parameters

        
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_param(n_x, n_h, n_y)

    for i in range(num_iterations):
        A2, cache = propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_params(parameters, grads, learning_rate=1.2)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters
    



