# Ben Kabongo
# December 2024

# AMF: Explainable recommendation with fusion of aspect information
# https://yneversky.github.io/Papers/Hou2019_Article_ExplainableRecommendationWithF.pdf


import torch
import torch.nn as nn
import torch.nn.functional as F


class Objective(nn.Module):
    """
    Objective function for AMF
            J(U, V, X, Y) = 0.5 * ||R - UV^T||_F^2 + 0.5 * gamma1 * ||P - UX^T||_F^2
                        + 0.5 * gamma2 * ||Q - VY^T||_F^2
                        + 0.5 * lambda1 * (||U||_F^2 + ||V||_F^2 + ||X||_F^2 + ||Y||_F^2)
    """

    def __init__(self, config):
        super().__init__()
        self.gamma1 = config.gamma1
        self.gamma2 = config.gamma2
        self.lambda1 = config.lambda1
        self.frob_norm = lambda A: torch.sum(A ** 2)

    def forward(self, R: torch.Tensor, P: torch.Tensor, Q: torch.Tensor,
                U: torch.Tensor, V: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        
        term1 = 0.5 * self.frob_norm(R - torch.matmul(U, V.T))
        term2 = 0.5 * self.gamma1 * self.frob_norm(P - torch.matmul(U, X.T))
        term3 = 0.5 * self.gamma2 * self.frob_norm(Q - torch.matmul(V, Y.T))
        reg = 0.5 * self.lambda1 * (self.frob_norm(U) + self.frob_norm(V) + self.frob_norm(X) + self.frob_norm(Y))

        loss = term1 + term2 + term3 + reg
        return loss


class AMF(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.objective = Objective(config)

        self.K = config.K
        self.L = config.L

        self.U = nn.Parameter(torch.empty(config.n_users, self.K).uniform_(-0.1, 0.1))
        self.V = nn.Parameter(torch.empty(config.n_items, self.K).uniform_(-0.1, 0.1))
        self.X = nn.Parameter(torch.empty(self.L, self.K).uniform_(-0.1, 0.1))
        self.Y = nn.Parameter(torch.empty(self.L, self.K).uniform_(-0.1, 0.1))

    def forward(self, U_ids, I_ids):
        U = self.U[U_ids] # (batch_size, K)
        V = self.V[I_ids] # (batch_size, K)
        R = torch.sum(U * V, dim=1) # (batch_size,)
        return R
    
    def compute_loss(self, R, P, Q):
        return self.objective(R, P, Q, self.U, self.V, self.X, self.Y)
    


        