# Ben Kabongo
# December 2024

# A3NCF: An Adaptive Aspect Attention Model for Rating Prediction
# https://www.ijcai.org/proceedings/2018/0521.pdf


import torch
import torch.nn as nn

from atm import gibbs_sampling_atm


class A3NCF(nn.Module):
    """ A3NCF: An Adaptive Aspect Attention Model for Rating Prediction"""

    def __init__(self, config, Theta_u, Psi_i):

        super().__init__()
        self.config = config

        self.Theta_u = nn.Parameter(Theta_u, requires_grad=False) # (n_users, n_factors)
        self.Psi_i = nn.Parameter(Psi_i, requires_grad=False) # (n_items, n_factors)

        self.user_embedding = nn.Embedding(config.n_users, config.n_factors) # (n_users, n_factors)
        self.item_embedding = nn.Embedding(config.n_items, config.n_factors) # (n_items, n_factors)

        self.fusion_layer = nn.Sequential(
            nn.Linear(config.n_factors * 2, config.n_factors),
            nn.ReLU(),
            nn.Linear(config.n_factors, config.n_factors)
        )

        self.attention_layer = nn.Sequential(
            nn.Linear(config.n_factors * 4, config.n_factors),
            nn.ReLU(),
            nn.Linear(config.n_factors, config.n_factors, bias=False),
            nn.Softmax(dim=1)
        )

        self.rating_prediction = nn.Sequential(
            nn.Linear(config.n_factors, config.n_factors),
            nn.ReLU(),
            nn.Linear(config.n_factors, config.n_factors),
            nn.ReLU(),
            nn.Linear(config.n_factors, 1)
        )
    
    @classmethod
    def from_gibbs_sampling_atm(cls, config, data, vocabulary, Beta_w, Alpha_u, Gamma_i, eta):
        res = gibbs_sampling_atm(config, data, vocabulary, Beta_w, Alpha_u, Gamma_i, eta)
        return cls(
            config, 
            torch.tensor(res['Theta_u'], dtype=torch.float32),
            torch.tensor(res['Psi_i'], dtype=torch.float32)
        )

    def forward(self, U_ids: torch.Tensor, I_ids: torch.Tensor) -> torch.Tensor:
        U = self.user_embedding(U_ids)  # (batch_size, n_factors)
        I = self.item_embedding(I_ids)  # (batch_size, n_factors)
        Theta_u = self.Theta_u[U_ids]   # (batch_size, n_factors)
        Psi_i = self.Psi_i[I_ids]       # (batch_size, n_factors)

        attn = torch.cat((U, I, Theta_u, Psi_i), dim=1)
        attn = self.attention_layer(attn) # (batch_size, n_factors)

        U = U + Theta_u
        I = I + Psi_i
        F = torch.cat((U, I), dim=1)    # (batch_size, n_factors * 2)
        F = self.fusion_layer(F)        # (batch_size, n_factors)
        F = F * attn                    # (batch_size, n_factors)    
    
        R = self.rating_prediction(F)   # (batch_size, 1)
        return R.squeeze(1)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
