# Ben Kabongo
# December 2024

# ALFM: Aspect-Aware Latent Factor Model
# https://dl.acm.org/doi/pdf/10.1145/3178876.3186145


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import dirichlet, bernoulli, multinomial


def generate_atm(config, beta_w, gamma_u, gamma_i, alpha_u, alpha_i, eta):
    """ Aspect-aware Topic Model (ATM) generation process.

    Parameters:
    - n_topics: Number of latent topics in ATM
    - n_users: Number of users
    - n_items: Number of items
    - n_aspects: Number of aspects

    - beta_w: Dirichlet parameters for topic-word distributions
    - gamma_u: Dirichlet parameters for user aspect distributions
    - gamma_i: Dirichlet parameters for item aspect distributions
    - alpha_u: Dirichlet priors for user aspect-topic distributions
    - alpha_i: Dirichlet priors for item aspect-topic distributions
    - eta: Beta priors for Bernoulli parameter

    Returns:
    - Dictionary of sampled parameters and generated data.
    """
    phi = dirichlet.rvs(beta_w, size=config.n_topics)

    lambda_u = dirichlet.rvs(gamma_u, size=config.n_users)
    lambda_i = dirichlet.rvs(gamma_i, size=config.n_items)

    theta_u = np.array([dirichlet.rvs(alpha_u) for _ in range(config.n_users)])  # (n_users, n_aspects, n_topics)
    psi_i = np.array([dirichlet.rvs(alpha_i) for _ in range(config.n_items)])  # (n_items, n_aspects, n_topics)

    pi_u = np.random.beta(eta[0], eta[1], size=config.n_users)

    reviews = []

    for u in range(config.n_users):
        for i in range(config.n_items):
            review = []
            for _ in range(np.random.poisson(5)):
                y = bernoulli.rvs(pi_u[u])
                if y == 0:
                    a_s = multinomial.rvs(1, lambda_u[u]).argmax()
                    z_s = multinomial.rvs(1, theta_u[u, a_s]).argmax()
                else:
                    a_s = multinomial.rvs(1, lambda_i[i]).argmax()
                    z_s = multinomial.rvs(1, psi_i[i, a_s]).argmax()
                
                words = multinomial.rvs(1, phi[z_s], size=np.random.poisson(10))
                review.append((a_s, z_s, words))
            reviews.append((u, i, review))

    return {
        'phi': phi,
        'lambda_u': lambda_u,
        'lambda_i': lambda_i,
        'theta_u': theta_u,
        'psi_i': psi_i,
        'pi_u': pi_u,
        'reviews': reviews
    }


class JSD(nn.Module):
    """Jensen-Shannon Divergence
        JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        where M = 0.5 * (P + Q)
    """

    def __init__(self):
        super(JSD, self).__init__()
        self.KL = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, P: torch.tensor, Q: torch.tensor):
        P, Q = P.view(-1, P.size(-1)), Q.view(-1, Q.size(-1))
        M = (0.5 * (P + Q)).log()
        return 0.5 * (self.KL(M, P.log()) + self.KL(M, Q.log()))


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
                A: torch.Tensor, Bu: torch.Tensor, Bi: torch.Tensor) -> torch.Tensor:
        
        term1 = 0.5 * torch.sum((R - R_hat) ** 2)
        term2 = 0.5 * self.lambda_u * torch.sum(U ** 2)
        term3 = 0.5 * self.lambda_i * torch.sum(I ** 2)
        term4 = 0.5 * self.lambda_a * torch.sum(torch.abs(A))
        term5 = 0.5 * self.lambda_b * (torch.sum(Bu ** 2) + torch.sum(Bi ** 2))

        loss = term1 + term2 + term3 + term4 + term5
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

        self.A = nn.Parameter(torch.empty(config.n_aspects, config.n_factors)) # (n_aspects, n_factors)
        self.Bu = nn.Parameter(torch.empty(config.n_users, config.n_factors)) # (n_users, 1)
        self.Bi = nn.Parameter(torch.empty(config.n_items, config.n_factors)) # (n_items, 1)
        self.B = nn.Parameter(torch.empty(1)) # (1,)

        self.JSD = JSD()

    @classmethod
    def from_parameters(cls, config, beta_w, gamma_u, gamma_i, alpha_u, alpha_i, eta):
        res = generate_atm(config, beta_w, gamma_u, gamma_i, alpha_u, alpha_i, eta)
        return cls(
            config, 
            torch.tensor(res['theta_u'], dtype=torch.float32),
            torch.tensor(res['psi_i'], dtype=torch.float32),
            torch.tensor(res['pi_u'], dtype=torch.float32),
            torch.tensor(res['lambda_u'], dtype=torch.float32),
            torch.tensor(res['lambda_i'], dtype=torch.float32)
        )
    

    def forward(self, U_ids, I_ids):
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

        A_ratings = ((A.unsqueeze(0) * U.unsqueeze(1)) * (A.unsqueeze(0) * I.unsqueeze(1))).sum(dim=2) 
        A_ratings = S_UIA * A_ratings # (batch_size, n_aspects)
        
        R = (P_UIA * A_ratings).sum(dim=1) + Bu + Bi + B # (batch_size,)
        return R, A_ratings

