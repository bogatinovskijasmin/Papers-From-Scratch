import torch
def pool2d(X, pool_size, mode="max"):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0]-p_h + 1, X.shape[1]  - p_w +1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == "max":
                Y[i, j] = X[i: i+p_h, j:j+p_w].max()
            elif mode=="avg":
                Y[i, j] = X[i: i + p_h, j:j + p_w].mean()
    return Y
def corr2d(X, K):
    height, width = K.shape[0], K.shape[1]
    Y = torch.zeros((X.shape[0]-height+1, X.shape[1]-width+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+height, j:j+width]*K).sum()
    return Y
def corr2d_multi_in(X, K):
    return sum((corr2d(x, k) for x, k in zip(X, K)))
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K])