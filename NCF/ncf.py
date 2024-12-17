# Ben Kabongo
# December 2024

# NeuMF/NCF

import argparse
import os
import sys
import torch
import torch.nn as nn
from typing import *

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from utils.uir_common import main
from GMF.gmf import GMF
from MLP.mlp import MLP


class NCF(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.gmf = GMF(config)
        self.mlp = MLP(config)
        self.predict = nn.Linear(config.n_factors + config.layers[-1], 1)

    def forward(self, U_ids, I_ids):
        logits_gmf = self.gmf.h(U_ids, I_ids)
        logits_mlp = self.mlp.h(U_ids, I_ids)
        logits = torch.cat([logits_gmf, logits_mlp], dim=-1)
        R = self.predict(logits).view(-1)
        return R
    
    def save(self, save_model_path: str):
        torch.save(self.state_dict(), save_model_path)

    def load(self, save_model_path: str):
        self.load_state_dict(torch.load(save_model_path))
    
NeuMF = NCF


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="TripAdvisor")
    parser.add_argument("--data_path", type=str, default="../datasets/processed/TripAdvisor/data.csv")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--val_path", type=str, default="")
    
    parser.add_argument("--n_factors", type=int, default=32)
    parser.add_argument("--layers", type=int, nargs="+", default=[256, 128, 64, 32])
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--threshold_rating", type=float, default=4.0)
    parser.add_argument("--ranking_metrics_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(ranking_metrics_flag=False)

    parser.add_argument("--save_dir", type=str, default="../exps/TripAdvisor")
    parser.add_argument("--save_model_path", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="NCF")

    parser.add_argument("--user_column", type=str, default="user_id")
    parser.add_argument("--item_column", type=str, default="item_id")
    parser.add_argument("--rating_column", type=str, default="rating")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)

    config = parser.parse_args()
    main(NCF, config)
