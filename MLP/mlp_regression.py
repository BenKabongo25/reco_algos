# Ben Kabongo
# December 2024

# MLP for rating prediction

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import *

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from utils.evaluation import rating_evaluation_pytorch


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


class RatingDataset(Dataset):

    def __init__(self, ratings_df: pd.DataFrame, args):
        super().__init__()
        self.ratings_df = ratings_df
        self.args = args

    def __len__(self) -> int:
        return len(self.ratings_df)
    
    def __getitem__(self, index) -> Any:
        row = self.ratings_df.iloc[index]
        u = row[self.args.user_column]
        i = row[self.args.item_column]
        r = row[self.args.rating_column]
        return u, i, r


class MLPRecommender(nn.Module):

    def __init__(
            self, 
            n_users: int, 
            n_items: int, 
            layers: List[int]=[256, 64, 16],
        ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = layers[0] // 2
        self.layers = layers

        self.user_embed = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embedding_dim)
        self.item_embed = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embedding_dim)

        hiddens = []
        for i in range(len(layers) - 1):
            hiddens.append(nn.Linear(layers[i], layers[i + 1]))
            hiddens.append(nn.ReLU())
        self.hiddens = nn.Sequential(*hiddens)
        self.predict = nn.Linear(layers[-1], 1)

    def forward(self, U_ids: torch.Tensor, I_ids: torch.Tensor) -> torch.Tensor:
        user_embeddings = self.user_embed(U_ids)
        item_embeddings = self.item_embed(I_ids)
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)
        logits = self.hiddens(embeddings)
        R = self.predict(logits)
        return R
    
    def save(self, save_model_path: str):
        torch.save(self.state_dict(), save_model_path)

    def load(self, save_model_path: str):
        self.load_state_dict(torch.load(save_model_path))


def train(model, optimizer, dataloader, loss_fn, args):
    running_loss = .0

    model.train()
    for U_ids, I_ids, R in dataloader:
        optimizer.zero_grad()
        U_ids = torch.LongTensor(U_ids).to(args.device)
        I_ids = torch.LongTensor(I_ids).to(args.device)
        R = torch.tensor(R, dtype=torch.float32).to(args.device)
        R_hat = model(U_ids, I_ids).squeeze()
        loss = loss_fn(R_hat, R)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    running_loss /= len(dataloader)
    return running_loss


def eval(model, dataloader, loss_fn, args):
    users = []
    references = []
    predictions = []
    running_loss = .0

    model.eval()
    with torch.no_grad():
        for U_ids, I_ids, R in dataloader:
            U_ids = torch.LongTensor(U_ids).to(args.device)
            I_ids = torch.LongTensor(I_ids).to(args.device)
            R = torch.tensor(R, dtype=torch.float32).to(args.device)
            R_hat = model(U_ids, I_ids).squeeze()
            loss = loss_fn(R_hat, R)
            running_loss += loss.item()

            users.extend(U_ids.cpu().detach().tolist())
            references.extend(R.cpu().detach().tolist())
            predictions.extend(R_hat.cpu().detach().tolist())

    running_loss /= len(dataloader)
    ratings_scores = rating_evaluation_pytorch(args, predictions, references, users)
    return {"loss": running_loss, **ratings_scores}


def trainer(model, train_dataloader, val_dataloader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    train_infos = {"loss": []}
    val_infos = {}

    best_loss = float("inf")
    best_rmse = float("inf")

    results = {}

    progress_bar = tqdm(range(1, 1 + args.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        train_epoch_loss = train(model, optimizer, train_dataloader, loss_fn, args)
        train_infos["loss"].append(train_epoch_loss)

        if val_dataloader is not None:
            if epoch % args.eval_every == 0:
                val_epoch_infos = eval(model, val_dataloader, loss_fn, args)
                for metric in val_epoch_infos:
                    if metric not in val_infos:
                        val_infos[metric] = []
                    val_infos[metric].append(val_epoch_infos[metric])
                
                val_epoch_rmse = val_epoch_infos["rmse"]
                if best_rmse > val_epoch_rmse:
                    best_rmse = val_epoch_rmse
                    model.save(args.save_model_path)

        else:
            if best_loss > train_epoch_loss:
                best_loss = train_epoch_loss
                model.save(args.save_model_path)

        progress_bar.set_description(
            f"[{epoch} / {args.n_epochs}] Loss: {train_epoch_loss:.4f}" +
            f" Best: loss={best_loss:.4f} rmse={best_rmse:.4f}"
        )

        results = {"train": train_infos, "val": val_infos}
        with open(args.save_res_path, "w") as res_file:
            json.dump(results, res_file)

    return results


def main(args):
    set_seed(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    if args.verbose:
        print(
            f"Dataset: {args.dataset_name}\n" +
            f"Device: {device}\n\n" +
            f"Arguments:\n{args}\n\n"
        )

    columns = [args.user_column, args.item_column, args.rating_column]
    if args.data_path != "":
        data_df = pd.read_csv(args.data_path)[columns]
        data_df[args.rating_column] = data_df[args.rating_column].astype(float)
        if args.verbose:
            print("Data shape: ", data_df.shape)
            print(data_df.head())
        
        train_df = data_df.sample(frac=args.train_size, seed=args.seed)
        test_val_df = data_df.drop(train_df.index)
        test_size = args.test_size / (args.test_size + args.val_size)
        test_df = test_val_df.sample(frac=test_size, seed=args.seed)
        val_df = None
        if args.val_size > 0:
            val_df = test_val_df.drop(test_df.index)

    else:
        train_df = pd.read_csv(args.train_path)[columns]
        train_df[args.rating_column] = train_df[args.rating_column].astype(float)
        test_df = pd.read_csv(args.test_path)[columns]
        test_df[args.rating_column] = test_df[args.rating_column].astype(float)
        data_df = pd.concat([train_df, test_df])

        val_df = None
        if args.val_path != "":
            val_df = pd.read_csv(args.val_path)[columns]
            val_df[args.rating_column] = val_df[args.rating_column].astype(float)
            data_df = pd.concat([data_df, val_df])

    users_vocab = {u: i for i, u in enumerate(data_df[args.user_column].unique())}
    items_vocab = {i: j for j, i in enumerate(data_df[args.item_column].unique())}

    train_df[args.user_column] = train_df[args.user_column].apply(lambda u: users_vocab[u])
    train_df[args.item_column] = train_df[args.item_column].apply(lambda i: items_vocab[i])
    test_df[args.user_column] = test_df[args.user_column].apply(lambda u: users_vocab[u])
    test_df[args.item_column] = test_df[args.item_column].apply(lambda i: items_vocab[i])
    if val_df is not None:
        val_df[args.user_column] = val_df[args.user_column].apply(lambda u: users_vocab[u])
        val_df[args.item_column] = val_df[args.item_column].apply(lambda i: items_vocab[i])

    train_dataset = RatingDataset(train_df, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = RatingDataset(test_df, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    val_dataloader = None
    if val_df is not None:
        val_dataset = RatingDataset(val_df, args)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MLPRecommender(n_users=len(users_vocab), n_items=len(items_vocab), layers=args.layers)
    model.to(args.device)
    if args.save_model_path != "":
        model.load(args.save_model_path)
    else:
        args.save_model_path = os.path.join(args.save_dir, args.exp_name, "model.pth")
    
    args.save_res_path = os.path.join(args.save_dir, args.exp_name, "results.json")
    results = trainer(model, train_dataloader, val_dataloader, args)
    with open(args.save_res_path, "w") as res_file:
        json.dump(results, res_file)

    test_infos = eval(model, test_dataloader, nn.MSELoss(), args)
    results["test"] = test_infos
    with open(args.save_res_path, "w") as res_file:
        json.dump(results, res_file)


    plt.figure(figsize=(10, 6))
    plt.plot(results["train"]["loss"], label="train")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, args.exp_name, "loss.png"))

    for metric in results["val"]:
        plt.figure(figsize=(10, 6))
        plt.plot(results["val"][metric], label="val")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, args.exp_name, metric.lower() + ".png"))

  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="RateBeer")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--val_path", type=str, default="")
    
    parser.add_argument("--layers", type=int, nargs="+", default=[256, 64, 16])
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--threshold_rating", type=float, default=4.0)
    parser.add_argument("--ranking_metrics_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(ranking_metrics_flag=False)

    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--save_model_path", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="MLP")

    parser.add_argument("--user_column", type=str, default="user_id")
    parser.add_argument("--item_column", type=str, default="item_id")
    parser.add_argument("--rating_column", type=str, default="rating")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)

    args = parser.parse_args()
    main(args)
