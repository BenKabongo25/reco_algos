# Ben Kabongo
# December 2024

import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Any, Dict, List

from data import AspectRatingsDataset, process_data
from module import ANR, RatingsLoss



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rating_evaluation_pytorch(config: Any, 
                              predictions: List[float], references: List[float]) -> Dict[str, float]:
    actual_ratings = torch.tensor(references, dtype=torch.float32).to(config.device)
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32).to(config.device)
    rmse = torch.sqrt(F.mse_loss(predictions_tensor, actual_ratings))
    mae = F.l1_loss(predictions_tensor, actual_ratings)
    return {'rmse': rmse.item(), 'mae': mae.item()}


def train(model: ANR, config, optimizer, loss_fn, dataloader):
    model.train()
    losses = {"total": 0.0, "overall_rating": 0.0, "aspects_ratings": 0.0}

    for batch in tqdm(dataloader, f"Training ANR", colour="blue", total=len(dataloader)):
        U_ids = torch.LongTensor(batch["user_id"]).to(config.device) # (batch_size,)
        I_ids = torch.LongTensor(batch["item_id"]).to(config.device) # (batch_size,)
        R = torch.tensor(batch["overall_rating"], dtype=torch.float32).to(config.device) # (batch_size,)
        A_ratings = None
        if getattr(config, "aspects", None) is not None:
            A_ratings = torch.tensor(batch["aspects_ratings"], dtype=torch.float32).to(config.device) # (batch_size, n_aspects)

        output = model(U_ids, I_ids)
        R_hat = output["overall_rating"]
        A_ratings_hat = output["aspects_ratings"]
        all_loss = loss_fn(R, R_hat, A_ratings, A_ratings_hat)
        loss = all_loss["total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for loss in all_loss:
            losses[loss] += all_loss[loss].item()

    for loss in losses:
        losses[loss] /= len(dataloader)
    return losses
    

def eval(model: ANR, config, dataloader):
    model.eval()

    references = {"overall_rating": []}
    predictions = {"overall_rating": []}
    aspects_flag = getattr(config, "aspects", None) is not None
    if aspects_flag:
        for aspect in config.aspects:
            references[aspect] = []
            predictions[aspect] = []

    for batch_idx, batch in tqdm(enumerate(dataloader), "Evaluation", colour="cyan", total=len(dataloader)):
        U_ids = torch.LongTensor(batch["user_id"]).to(config.device) # (batch_size,)
        I_ids = torch.LongTensor(batch["item_id"]).to(config.device) # (batch_size,)
        R = torch.tensor(batch["overall_rating"], dtype=torch.float32).to(config.device) # (batch_size,)
        A_ratings = None
        if aspects_flag:
            A_ratings = torch.tensor(batch["aspects_ratings"], dtype=torch.float32).to(config.device)

        output = model(U_ids, I_ids)
        R_hat = output["overall_rating"]
        A_ratings_hat = output["aspects_ratings"]

        references["overall_rating"].extend(R.cpu().detach().tolist())
        predictions["overall_rating"].extend(R_hat.cpu().detach().tolist())
        if aspects_flag:
            for aspect in config.aspects:
                references[aspect].extend(A_ratings[:, aspect].cpu().detach().tolist())
                predictions[aspect].extend(A_ratings_hat[:, aspect].cpu().detach().tolist())

        if config.verbose and batch_idx == 0:
            n_samples = min(10, len(U_ids))
            log = ""
            for i in range(n_samples):
                log += "\n" + " ".join([
                    f"User ID: {U_ids[i]}",
                    f"Item ID: {I_ids[i]}",
                    f"Overall Rating: Actual={R[i]:.4f} Predicted={R_hat[i]:4f}"
                ])
                if aspects_flag:
                    for aspect in config.aspects:
                        log += f"\nAspect {aspect}: Actual={A_ratings[i, aspect]:.4f} Predicted={A_ratings_hat[i, aspect]:.4f}"
            config.logger.info(log)

    scores = {}
    for element in references:
        scores[element] = rating_evaluation_pytorch(config, predictions[element], references[element])
    return scores


def write_loss(writer, losses, phase, epoch):
    for loss in losses:
        writer.add_scalar(f"{loss}/{phase}", losses[loss], epoch)


def write_eval(writer, scores, phase, epoch):
    for element in scores:
        for metric in scores[element]:
            writer.add_scalar(f"{element}/{metric}/{phase}", scores[element][metric], epoch)


def trainer(model: ANR, config, train_dataloader, eval_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = RatingsLoss(config).to(config.device)

    train_infos = {"loss": []}
    eval_infos = {}

    best_rating = float("inf")
    progress_bar = tqdm(range(1, 1 + config.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        losses = train(model, config, optimizer, loss_fn, train_dataloader)        
        write_loss(config.writer, losses, "train", epoch)

        for loss in losses.keys():
            if loss not in train_infos.keys():
                train_infos[loss] = []
            train_infos[loss].append(losses[loss])
        train_loss = losses["total"]

        desc = (
            f"[{epoch} / {config.n_epochs}] Loss: {train_loss:.4f} " +
            f"Best: {config.rating_metric}={best_rating:.4f}"
        )

        if epoch % config.eval_every == 0:
            with torch.no_grad():
                scores = eval(model, config, dataloader=eval_dataloader)
            write_eval(config.writer, scores, "eval", epoch)
            
            for metric_set in scores.keys():
                if metric_set not in eval_infos.keys():
                    eval_infos[metric_set] = {}
                for metric in scores[metric_set].keys():
                    if metric not in eval_infos[metric_set].keys():
                        eval_infos[metric_set][metric] = []
                    eval_infos[metric_set][metric].append(scores[metric_set][metric])

            eval_rating = scores["overall_rating"][config.rating_metric]
            if eval_rating < best_rating:
                model.save_model(config.save_model_path)
                best_rating = eval_rating

            desc = (
                f"[{epoch} / {config.n_epochs}] " +
                f"Loss: train={train_loss:.4f} " +
                f"Rating ({config.rating_metric}): test={eval_rating:.4f} best={best_rating:.4f}"
            )

        config.logger.info(desc)
        progress_bar.set_description(desc)

        results = {"train": train_infos, "eval": eval_infos}
        with open(config.res_file_path, "w") as res_file:
            json.dump(results, res_file)

    return train_infos, eval_infos


def load_word_embeddings(path):
    pretrained_word_embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            pretrained_word_embeddings[word] = vector
    return pretrained_word_embeddings


def run(config):
    set_seed(config.seed)

    os.makedirs(config.save_dir, exist_ok=True)
    config.log_file_path = os.path.join(config.save_dir, "log.txt")
    config.res_file_path = os.path.join(config.save_dir, "res.json")
    config.save_model_path = os.path.join(config.save_dir, "model.pth")

    logger = logging.getLogger("ANR" + config.dataset_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(config.log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    config.logger = logger

    writer = SummaryWriter(config.save_dir)
    config.writer = writer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    columns = ["user_id", "item_id", "rating", "review"]
    aspects_flag = getattr(config, "aspects", None) is not None
    if aspects_flag:
        columns += config.aspects

    if config.data_path != "":
        data_df = pd.read_csv(config.data_path).dropna()[columns]
        data_df["rating"] = data_df["rating"].astype(float)

        if aspects_flag:
            for aspect in config.aspects:
                data_df[aspect] = data_df[aspect].astype(float)

        if config.verbose:
            config.logger.info(f"Data shape: {data_df.shape}")
            config.logger.info(str(data_df.head()))
        
        train_df = data_df.sample(frac=config.train_size, random_state=config.seed)
        test_val_df = data_df.drop(train_df.index)
        test_size = config.test_size / (config.test_size + config.val_size)
        test_df = test_val_df.sample(frac=test_size, random_state=config.seed)
        val_df = test_val_df.drop(test_df.index)

    else:
        train_df = pd.read_csv(config.train_path)[columns]
        train_df["rating"] = train_df["rating"].astype(float)
        test_df = pd.read_csv(config.test_path)[columns]
        test_df["rating"] = test_df["rating"].astype(float)
        val_df = pd.read_csv(config.val_path)[columns]
        val_df["rating"] = val_df["rating"].astype(float)

        if aspects_flag:
            for aspect in config.aspects:
                train_df[aspect] = train_df[aspect].astype(float)
                test_df[aspect] = test_df[aspect].astype(float)
                val_df[aspect] = val_df[aspect].astype(float)

        data_df = pd.concat([train_df, test_df, val_df])

    users_vocab = {user_id: i for i, user_id in enumerate(data_df["user_id"].unique())}
    items_vocab = {item_id: i for i, item_id in enumerate(data_df["item_id"].unique())}
    config.n_users = len(users_vocab)
    config.n_items = len(items_vocab)

    train_df["user_id"] = train_df["user_id"].map(users_vocab)
    train_df["item_id"] = train_df["item_id"].map(items_vocab)
    test_df["user_id"] = test_df["user_id"].map(users_vocab)
    test_df["item_id"] = test_df["item_id"].map(items_vocab)
    val_df["user_id"] = val_df["user_id"].map(users_vocab)
    val_df["item_id"] = val_df["item_id"].map(items_vocab)

    if config.model_params_path == "":
        words_embeddings = load_word_embeddings(config.words_embedding_path)
        params = process_data(config, train_df, list(users_vocab.values()), list(items_vocab.values()), words_embeddings)
        config.model_params_path = os.path.join(config.save_dir, "params.pkl")
        pickle.dump(params, open(config.model_params_path, "wb"))
        config.logger.info(f"Model params saved at {config.model_params_path}")
    else:
        params = pickle.load(open(config.model_params_path, "rb"))

    U_documents = torch.tensor(params["users_document"], dtype=torch.long)
    I_documents = torch.tensor(params["items_document"], dtype=torch.long)
    vocab_words_embeddings = torch.tensor(params["vocab_words_embeddings"], dtype=torch.float32)
    model = ANR(config, U_documents, I_documents, vocab_words_embeddings)

    model.to(config.device)
    if config.load_model:
        model.load_model(config.save_model_path)

    train_dataset = AspectRatingsDataset(config, train_df)
    eval_dataset = AspectRatingsDataset(config, val_df)
    test_dataset = AspectRatingsDataset(config, test_df)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)    

    if config.verbose:
        log = "\n" + (
            f"Model name: ANR\n" +
            f"Dataset: {config.dataset_name}\n" +
            f"#Aspects: {config.n_aspects}\n" +
            f"#Users: {config.n_users}\n" +
            f"#Items: {config.n_items}\n" +
            f"Device: {device}\n\n" +
            f"Args:\n{config}\n\n" +
            f"Model: {model}\n\n" +
            f"Data:\n{train_df.head(5)}\n\n"
        )
        config.logger.info(log)

    config.logger.info("Training...")
    train_infos, eval_infos = trainer(model, config, train_dataloader, eval_dataloader)

    config.logger.info("Testing...")
    model.load_model(config.save_model_path)
    with torch.no_grad():
        test_infos = eval(model, config, dataloader=test_dataloader)

    results = {"test": test_infos, "train": train_infos, "eval": eval_infos}
    config.logger.info(str(results))
    with open(config.res_file_path, "w") as res_file:
        json.dump(results, res_file)

    config.logger.info("Done!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="")

    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--words_embedding_path", type=str, default="")

    parser.add_argument("--n_aspects", type=int, default=6)
    parser.add_argument("--d_words", type=int, default=300)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction)
    parser.set_defaults(bias=True)
    parser.add_argument("--context_windows_size", type=int, default=3)
    parser.add_argument("--hidden_size1", type=int, default=128)
    parser.add_argument("--hidden_size2", type=int, default=64)
    parser.add_argument("--doc_len", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--vocab_size", type=int, default=10_000)

    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--model_params_path", type=str, default="")
    parser.add_argument("--load_model", action=argparse.BooleanOptionalAction)
    parser.set_defaults(load_model=False)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--alpha", type=float, default=1/7)
    parser.add_argument("--beta", type=float, default=6/7)

    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--rating_metric", type=str, default="rmse")
    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    parser.add_argument("--verbose_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)

    config = parser.parse_args()
    run(config)
