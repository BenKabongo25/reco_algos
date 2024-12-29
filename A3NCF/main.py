# Ben Kabongo
# December 2024

import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Any, Dict, List


from topic_model import process_data, gibbs_sampling_topic_model
from data import RatingsDataset
from module import A3NCF



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


def train(model: A3NCF, config, optimizer, dataloader, loss_fn):
    model.train()
    train_loss = .0

    for batch in tqdm(dataloader, f"Training A3NCF", colour="blue", total=len(dataloader)):
        U_ids = torch.LongTensor(batch["user_id"]).to(config.device) # (batch_size,)
        I_ids = torch.LongTensor(batch["item_id"]).to(config.device) # (batch_size,)
        R = torch.tensor(batch["overall_rating"], dtype=torch.float32).to(config.device) # (batch_size,)
        R_hat = model(U_ids, I_ids)
        loss = loss_fn(R_hat, R)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(dataloader)
    

def eval(model: A3NCF, config, dataloader):
    model.eval()

    references = {"overall_rating": []}
    predictions = {"overall_rating": []}

    for batch_idx, batch in tqdm(enumerate(dataloader), "Evaluation", colour="cyan", total=len(dataloader)):
        U_ids = torch.LongTensor(batch["user_id"]).to(config.device) # (batch_size,)
        I_ids = torch.LongTensor(batch["item_id"]).to(config.device) # (batch_size,)
        R = torch.tensor(batch["overall_rating"], dtype=torch.float32).to(config.device) # (batch_size,)
        R_hat = model(U_ids, I_ids)

        references["overall_rating"].extend(R.cpu().detach().tolist())
        predictions["overall_rating"].extend(R_hat.cpu().detach().tolist())

        if config.verbose and batch_idx == 0:
            n_samples = min(10, len(U_ids))
            log = ""
            for i in range(n_samples):
                log += "\n" + " ".join([
                    f"User ID: {U_ids[i]}",
                    f"Item ID: {I_ids[i]}",
                    f"Overall Rating: Actual={R[i]:.4f} Predicted={R_hat[i]:4f}"
                ])
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


def trainer(model: A3NCF, config, train_dataloader, eval_dataloader):
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.MSELoss()

    train_infos = {"loss": []}
    eval_infos = {}

    best_rating = float("inf")
    progress_bar = tqdm(range(1, 1 + config.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        train_loss = train(model, config, optimizer, train_dataloader, loss_fn)
        train_infos["loss"].append(train_loss)
        config.writer.add_scalar("loss/train", train_loss, epoch)

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


def run(config):
    set_seed(config.seed)

    os.makedirs(config.save_dir, exist_ok=True)
    config.log_file_path = os.path.join(config.save_dir, "log.txt")
    config.res_file_path = os.path.join(config.save_dir, "res.json")
    config.save_model_path = os.path.join(config.save_dir, "model.pth")

    logger = logging.getLogger("A3NCF" + config.dataset_name)
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
    if config.data_path != "":
        data_df = pd.read_csv(config.data_path).dropna()[columns]
        data_df["rating"] = data_df["rating"].astype(float)
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
        data, words_vocab = process_data(config, train_df)
        Beta_w = np.ones(config.vocabulary_size)
        Alpha_u = np.ones(config.n_factors)
        Gamma_i = np.ones(config.n_factors)
        eta = (1, 1)
        model = A3NCF.from_gibbs_sampling_topic_model(
            config, data, words_vocab, Beta_w, Alpha_u, Gamma_i, eta
        )
    else:
        import pickle
        params = pickle.load(open(config.model_params_path, "rb"))
        Theta_u = torch.tensor(params["Theta_u"], dtype=torch.float32)
        Psi_i = torch.tensor(params["Psi_i"], dtype=torch.float32)
        model = A3NCF(config, Theta_u, Psi_i)

    model.to(config.device)
    if config.load_model:
        model.load_model(config.save_model_path)

    train_dataset = RatingsDataset(config, train_df)
    eval_dataset = RatingsDataset(config, val_df)
    test_dataset = RatingsDataset(config, test_df)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)    

    if config.verbose:
        log = "\n" + (
            f"Model name: A3NCF\n" +
            f"Dataset: {config.dataset_name}\n" +
            f"#Factors: {config.n_factors}\n" +
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

    parser.add_argument("--n_factors", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--vocabulary_size", type=int, default=10_000)
    parser.add_argument("--gibbs_sampling_iterations", type=int, default=1000)

    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--model_params_path", type=str, default="")
    parser.add_argument("--load_model", action=argparse.BooleanOptionalAction)
    parser.set_defaults(load_model=False)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)

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
