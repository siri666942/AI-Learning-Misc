import matplotlib.pyplot as plt
import numpy as np

from dnn_app_utils import (
    L_model_backward,
    L_model_forward,
    compute_cost,
    initialize_parameters,
    initialize_parameters_deep,
    linear_activation_backward,
    linear_activation_forward,
    load_data,
    predict,
    print_mislabeled_images,
    update_parameters,
)




def preprocess_data(train_x_orig, test_x_orig):
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    train_x = train_x_flatten / 255.0
    test_x = test_x_flatten / 255.0
    return train_x, test_x

def two_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False):
    '''
    implements a two-layer neural network: linear->relu->linear->sigmoid
    '''
    np.random.seed(1)
    grads={}
    costs=[]
    m=X.shape[1]
    (n_x,n_h,n_y)=layers_dims

    params=initialize_parameters(n_x,n_h,n_y)
    W1=params["W1"]
    W2=params["W2"]
    b1=params["b1"]
    b2=params["b2"]

    for i in range(0,num_iterations):
        A1,cache1=linear_activation_forward(X,W1,b1,"relu")
        A2,cache2=linear_activation_forward(A1,W2,b2,"sigmoid")

        cost=compute_cost(A2,Y)

        dA2=-(np.divide(Y,A2)-np.divide(1-Y,1-A2))

        dA1,dW2,db2=linear_activation_backward(dA2,cache2,"sigmoid")
        dA0,dW1,db1=linear_activation_backward(dA1,cache1,"relu")

        grads["dW1"]=dW1
        grads["dW2"]=dW2
        grads["db1"]=db1
        grads["db2"]=db2

        params=update_parameters(params,grads,learning_rate)

        W1=params["W1"]
        W2=params["W2"]
        b1=params["b1"]
        b2=params["b2"]

        if print_cost and i%100==0:
            print("after iteration{}:{}".format(i,np.squeeze(cost)))
            costs.append(cost)
        

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show()

    return params

train_x_orig, train_y, test_x_orig, test_y, CLASSES=load_data()
train_x,test_x=preprocess_data(train_x_orig,test_x_orig)
n_x=train_x.shape[0]
n_h=7
n_y=1
params=two_layer_model(train_x,train_y,layers_dims=(n_x,n_h,n_y),num_iterations=3000,print_cost=True)

