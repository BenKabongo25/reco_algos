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

from .functions import set_seed
from .evaluation import rating_evaluation_pytorch


class RatingDataset(Dataset):

    def __init__(self, ratings_df: pd.DataFrame, args):
        super().__init__()
        self.ratings_df = ratings_df
        self.args = args

    def __len__(self) -> int:
        return len(self.ratings_df)
    
    def __getitem__(self, index) -> Any:
        row = self.ratings_df.iloc[index]
        u = int(row[self.args.user_column])
        i = int(row[self.args.item_column])
        r = float(row[self.args.rating_column])
        return u, i, r


def train(model, optimizer, dataloader, loss_fn, args):
    running_loss = .0

    model.train()
    for U_ids, I_ids, R in dataloader:
        optimizer.zero_grad()
        U_ids = torch.LongTensor(U_ids).to(args.device)
        I_ids = torch.LongTensor(I_ids).to(args.device)
        R = torch.tensor(R.clone().detach(), dtype=torch.float32).to(args.device)
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


def main(model_class, args):
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
        
        train_df = data_df.sample(frac=args.train_size, random_state=args.seed)
        test_val_df = data_df.drop(train_df.index)
        test_size = args.test_size / (args.test_size + args.val_size)
        test_df = test_val_df.sample(frac=test_size, random_state=args.seed)
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
    args.n_users = len(users_vocab)
    args.n_items = len(items_vocab)

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

    model = model_class(args)
    model.to(args.device)
    if args.save_model_path != "":
        model.load(args.save_model_path)
    else:
        args.save_model_path = os.path.join(args.save_dir, args.exp_name, "model.pth")
    
    args.save_res_path = os.path.join(args.save_dir, args.exp_name, "results.json")
    results = trainer(model, train_dataloader, val_dataloader, args)
    with open(args.save_res_path, "w") as res_file:
        json.dump(results, res_file)
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.exp_name), exist_ok=True)

    test_infos = eval(model, test_dataloader, nn.MSELoss(), args)
    results["test"] = test_infos
    with open(args.save_res_path, "w") as res_file:
        json.dump(results, res_file)

    for metric in results["val"]:
        plt.figure(figsize=(10, 6))
        plt.plot(results["val"][metric], label="val")
        if metric == "loss":
            plt.plot(results["train"][metric], label="train")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, args.exp_name, metric.lower() + ".png"))

