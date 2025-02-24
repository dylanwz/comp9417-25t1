import numpy as np

'''
Stochastic gradient descent (SGD) is about taking small steps towards an optimal configuration. It
requires two things:
1. a loss function to evaluate how far off our current parameters are; and
2. a step size to tell us how far to step.
'''


'''
Let
    - A,    Nx(P+1) matrix where A_np denotes the value of the pth feature of the nth datapoint;
    - X,    (P+1)x1 vector where X_p denotes the weight of the pth feature;
    - Y,    Nx1 vector where Y_n denotes the true value of the nth datapoint;
    - L,    scalar penalty term;
    - M,    step size of how much we should step in the direction of steepest descent at each
            iteration.
We have P features and N datapoints
'''

# output:   y_pred,   Nx1 vector where y_pred_n is the prediction for the nth input vector 
def predict(A, X):
    return np.matmul(A,X)

# output:   loss val, measure of the loss; how accurate the current weights are in prediction
def loss(A, X, Y, L):
    # In this case, MSE squares the component-wise differences between predicted and actual.
    loss_val = 1/2 * np.linalg.norm(predict(A,X) - Y, 2) + (1/2 * L * np.linalg.norm(X))

    return loss_val

# output:   gradient, magnitude of vector in direction of steepest descent, as scaled by M
def eval_step(A, X, Y, M):
    def loss_der(A, X, Y, L):
        # grad(f) = A^T(Ax - b) + 2 * lambda * x 
        return np.matmul(np.transpose(A), np.matmul(A, X) - Y) + (2 * L * X)
    
    return M * loss_der(A, X, Y)

