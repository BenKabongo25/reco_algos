# Ben Kabongo
# December 2024

# ALFM: Aspect-Aware Latent Factor Model
# https://dl.acm.org/doi/pdf/10.1145/3178876.3186145


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from atm import gibbs_sampling_atm


class JSD(nn.Module):
    """ Jensen-Shannon Divergence
        JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        where M = 0.5 * (P + Q)
    """

    def __init__(self):
        super(JSD, self).__init__()
        self.KL = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, P: torch.Tensor, Q: torch.Tensor):
        assert P.shape == Q.shape, "Shapes of P and Q must match"
        M = 0.5 * (P + Q)
        kl_pm = self.KL(M.log(), P.log()).sum(dim=-1)  # KL(P || M)
        kl_qm = self.KL(M.log(), Q.log()).sum(dim=-1)  # KL(Q || M)
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd


class ALFMLoss(nn.Module):
    """ Loss function for ALFM
        L(R, R_hat, U, I, A, Bu, Bi) = 0.5 * ||R - R_hat||_2^2 + 
            0.5 * lambda_u * ||U||_2^2 + lambda_i * ||I||_2^2 + 
            0.5 * lambda_a * ||A||_1 + 0.5 * lambda_b * (||Bu||_2^2 + ||Bi||_2^2)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lambda_u = config.lambda_u
        self.lambda_i = config.lambda_i
        self.lambda_a = config.lambda_a
        self.lambda_b = config.lambda_b

    def forward(self, R: torch.Tensor, R_hat: torch.Tensor, U: torch.Tensor, I: torch.Tensor,
                A: torch.Tensor, Bu: torch.Tensor=None, Bi: torch.Tensor=None) -> torch.Tensor:
        
        loss = 0.5 * torch.sum((R - R_hat) ** 2)
        loss += 0.5 * self.lambda_u * torch.sum(U ** 2)
        loss += 0.5 * self.lambda_i * torch.sum(I ** 2)
        loss += 0.5 * self.lambda_a * torch.sum(torch.abs(A))
        loss += 0.5 * self.lambda_b * (torch.sum(Bu ** 2) + torch.sum(Bi ** 2))

        return loss


class ALFM(nn.Module):
    """ Aspect-Aware Latent Factor Model """

    def __init__(self, config,
                Theta_u, Psi_i, Pi_u, Lambda_u, Lambda_i):

        super().__init__()
        self.config = config

        self.Theta_u = nn.Parameter(Theta_u, requires_grad=False) # (n_users, n_aspects, n_topics)
        self.Psi_i = nn.Parameter(Psi_i, requires_grad=False) # (n_items, n_aspects, n_topics)
        self.Pi_u = nn.Parameter(Pi_u, requires_grad=False) # (n_users,)
        self.Lambda_u = nn.Parameter(Lambda_u, requires_grad=False) # (n_users, n_aspects)
        self.Lambda_i = nn.Parameter(Lambda_i, requires_grad=False) # (n_items, n_aspects)

        self.user_embedding = nn.Embedding(config.n_users, config.n_factors) # (n_users, n_factors)
        self.item_embedding = nn.Embedding(config.n_items, config.n_factors) # (n_items, n_factors)

        self.A = nn.Parameter(torch.ones(config.n_aspects, config.n_factors)) # (n_aspects, n_factors)
        self.Bu = nn.Parameter(torch.zeros(config.n_users)) # (n_users, 1)
        self.Bi = nn.Parameter(torch.zeros(config.n_items)) # (n_items, 1)
        self.B = nn.Parameter(torch.zeros(1)) # (1,)

        self.JSD = JSD()
        self.loss = ALFMLoss(config)
    
    @classmethod
    def from_gibbs_sampling_atm(cls, config, data, vocabulary, Beta_w, Gamma_u, Gamma_i, Alpha_u, Alpha_i, eta):
        res = gibbs_sampling_atm(config, data, vocabulary, Beta_w, Gamma_u, Gamma_i, Alpha_u, Alpha_i, eta)
        return cls(
            config, 
            torch.tensor(res['Theta_u'], dtype=torch.float32),
            torch.tensor(res['Psi_i'], dtype=torch.float32),
            torch.tensor(res['Pi_u'], dtype=torch.float32),
            torch.tensor(res['Lambda_u'], dtype=torch.float32),
            torch.tensor(res['Lambda_i'], dtype=torch.float32)
        )

    def forward(self, U_ids: torch.Tensor, I_ids: torch.Tensor, R: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        U = self.user_embedding(U_ids)  # (batch_size, n_factors)
        I = self.item_embedding(I_ids)  # (batch_size, n_factors)

        Theta_u = self.Theta_u[U_ids]   # (batch_size, n_aspects, n_topics)
        Psi_i = self.Psi_i[I_ids]       # (batch_size, n_aspects, n_topics)
        Pi_u = self.Pi_u[U_ids]         # (batch_size,)  
        Lambda_u = self.Lambda_u[U_ids] # (batch_size, n_aspects)
        Lambda_i = self.Lambda_i[I_ids] # (batch_size, n_aspects)
        Bu = self.Bu[U_ids]             # (batch_size, 1)
        Bi = self.Bi[I_ids]             # (batch_size, 1)
        B = self.B                      # (1,)

        S_UIA = 1 - self.JSD(Theta_u.view(-1, self.config.n_topics), Psi_i.view(-1, self.config.n_topics))
        S_UIA = S_UIA.view(-1, self.config.n_aspects) # (batch_size, n_aspects)
        P_UIA = Pi_u.unsqueeze(1) * Lambda_u + (1 - Pi_u).unsqueeze(1) * Lambda_i # (batch_size, n_aspects)
        A = self.A # (n_aspects, n_factors)

        A_ratings_hat = ((A.unsqueeze(0) * U.unsqueeze(1)) * (A.unsqueeze(0) * I.unsqueeze(1))).sum(dim=2) 
        A_ratings_hat = S_UIA * A_ratings_hat # (batch_size, n_aspects)
        R_hat = (P_UIA * A_ratings_hat).sum(dim=1) + Bu + Bi + B # (batch_size,)

        loss = None
        if R is not None:
            loss = self.loss(R, R_hat, U, I, self.A, Bu, Bi)

        _out = {
            "R_hat": R_hat,
            "A_ratings_hat": A_ratings_hat,
            "loss": loss
        }
        return _out
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
