# Ben Kabongo
# January 2025

import argparse
import json
import logging
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import *
from module import *
from utils import *


def train(model, config, optimizer, loss_fn, dataloader):
    model.train()
    losses = {"total": 0.0, "rating": 0.0, "review": 0.0, "regularization": 0.0}

    for batch in tqdm(dataloader, desc="Training", colour="cyan"):
        U_ids = batch["user_id"]
        I_ids = batch["item_id"]
        T_ids = batch["review_ids"]
        R = batch["rating"]

        U_ids = torch.LongTensor(U_ids).to(config.device) # (batch_size,)
        I_ids = torch.LongTensor(I_ids).to(config.device) # (batch_size,)
        T_ids = torch.stack(T_ids, dim=1).to(dtype=torch.long, device=config.device) # (batch_size, seq_len + 2)
        R = torch.tensor(R.clone().detach(), dtype=torch.float32).to(config.device) # (batch_size,)

        optimizer.zero_grad()
        R_hat, logits = model(U_ids, I_ids, T_ids[:, :-1])
        loss_ouputs = loss_fn(R, R_hat, T_ids[:, 1:], logits)
        loss = loss_ouputs["total"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()

        for loss in loss_ouputs.keys():
            losses[loss] += loss_ouputs[loss].item()

    for loss in losses.keys():
        losses[loss] /= len(dataloader)
    return losses


def eval(model, config, loss_fn, dataloader):
    model.eval()
    losses = {"total": 0.0, "rating": 0.0, "review": 0.0, "regularization": 0.0}

    for batch in tqdm(dataloader, desc="Training", colour="cyan"):
        U_ids = batch["user_id"]
        I_ids = batch["item_id"]
        T_ids = batch["review_ids"]
        R = batch["rating"]

        U_ids = torch.LongTensor(U_ids).to(config.device) # (batch_size,)
        I_ids = torch.LongTensor(I_ids).to(config.device) # (batch_size,)
        T_ids = torch.stack(T_ids, dim=1).to(dtype=torch.long, device=config.device) # (batch_size, seq_len + 2)
        R = torch.tensor(R.clone().detach(), dtype=torch.float32).to(config.device) # (batch_size,)

        R_hat, logits = model(U_ids, I_ids, T_ids[:, :-1])
        loss_ouputs = loss_fn(R, R_hat, T_ids[:, 1:], logits, model.parameters())
        loss = loss_ouputs["total"]

        for loss in loss_ouputs.keys():
            losses[loss] += loss_ouputs[loss].item()

    for loss in losses.keys():
        losses[loss] /= len(dataloader)
    return losses


def generate_and_eval(model, config, dataloader, word_dict, 
                      rating_metrics=RATING_METRICS, review_metrics=REVIEW_METRICS):
    model.eval()
    
    users = []
    items = []
    reference_ratings = []
    prediction_ratings = []
    reference_reviews = []
    prediction_reviews = []

    with torch.no_grad():    
        for batch_idx, batch in tqdm(enumerate(dataloader), desc="Evaluation", colour="cyan", total=len(dataloader)):
            U_ids = batch["user_id"]
            I_ids = batch["item_id"]
            reviews = batch["review"]
            R = batch["rating"]

            U_ids = torch.LongTensor(U_ids).to(config.device) # (batch_size,)
            I_ids = torch.LongTensor(I_ids).to(config.device) # (batch_size,)
            R = torch.tensor(R.clone().detach(), dtype=torch.float32).to(config.device) # (batch_size,)
            R_hat, reviews_hat = model.generate(U_ids, I_ids, word_dict, config.review_length)  # (batch_size, seq_len)
            
            users.extend(U_ids.cpu().detach().tolist())
            items.extend(I_ids.cpu().detach().tolist())
            reference_ratings.extend(R.cpu().detach().tolist())
            prediction_ratings.extend(R_hat)
            reference_reviews.extend(reviews)
            prediction_reviews.extend(reviews_hat)

            if config.verbose and batch_idx == 0:
                for i in range(len(reviews)):
                    log = f"User ID: {U_ids[i]} "
                    log += f"Item ID: {I_ids[i]} "
                    log += f"Actual Rating: {R[i]} "
                    log += f"Predicted Rating: {R_hat[i]} "
                    log += f"Review: {reviews[i]}\n"
                    log += f"Generated: {reviews_hat[i]}\n"
                    log += f"-" * 80
                    config.logger.info(log)

    rating_scores = rating_evaluation(config, prediction_ratings, reference_ratings, rating_metrics)
    reviews_scores = review_evaluation(config, prediction_reviews, reference_reviews, review_metrics)
    if config.verbose:
        log = ""
        for metric, score in rating_scores.items():
            log += f"{metric}: {score:.4f} "
        log += "\n"
        for metric, score in reviews_scores.items():
            log += f"{metric}: {score:.4f} "
        config.logger.info(log)

    output_df = pd.DataFrame({
        "user_id": users,
        "item_id": items,
        "reference_review": reference_reviews,
        "prediction_review": prediction_reviews,
        "reference_rating": reference_ratings,
        "prediction_rating": prediction_ratings
    })
    return rating_scores, reviews_scores, output_df


def trainer(model, config, train_dataloader, eval_dataloader, word_dict):
    loss_fn = NRTLoss(config)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=config.lr)

    train_infos = {}
    eval_infos = {}

    best_bleu = -float("inf")
    best_rmse = float("inf")
    progress_bar = tqdm(range(1, 1 + config.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        train_epoch_infos = train(model ,config, optimizer, loss_fn, train_dataloader)

        for k_1 in train_epoch_infos.keys():
            if k_1 not in train_infos.keys():
                train_infos[k_1] = []
            train_infos[k_1].append(train_epoch_infos[k_1])

        train_loss = train_epoch_infos["total"]
        desc = (
            f"[{epoch} / {config.n_epochs}] " +
            f"train: loss={train_loss:.4f} " +
            f"best: rmse={best_rmse:.4f} bleu={best_bleu:.4f}"
        )

        if epoch % config.eval_every == 0:
            rating_scores, reviews_scores, _ = generate_and_eval(
                model, config, eval_dataloader, word_dict,
                rating_metrics=[RMSE], review_metrics=[BLEU]
            )
            eval_epoch_infos = {**rating_scores, **reviews_scores}
            
            for k_1 in eval_epoch_infos.keys():
                if k_1 not in eval_infos.keys():
                    eval_infos[k_1] = []
                eval_infos[k_1].append(eval_epoch_infos[k_1])    

            test_bleu = eval_epoch_infos["bleu"]
            test_rmse = eval_epoch_infos["rmse"]

            if test_bleu > best_bleu:
                best_bleu = test_bleu
                model.save(config.save_model_path)
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                #model.save(config.save_model_path)

            desc = (
                f"[{epoch} / {config.n_epochs}] " +
                f"train: loss={train_loss:.4f} " +
                f"eval: rmse={test_rmse:.4f} bleu={test_bleu:.4f} " +
                f"best: rmse={best_rmse:.4f} bleu={best_bleu:.4f}"
            )

        progress_bar.set_description(desc)
        config.logger.info(desc)

        results = {"train": train_infos, "eval": eval_infos}
        with open(config.res_file_path, "w") as res_file:
            json.dump(results, res_file)

    return train_infos, eval_infos


def main(config):
    set_seed(config.seed)

    os.makedirs(config.save_dir, exist_ok=True)
    config.log_file_path = os.path.join(config.save_dir, "log.txt")
    config.res_file_path = os.path.join(config.save_dir, "res.json")
    config.save_model_path = os.path.join(config.save_dir, "model.pth")

    logger = logging.getLogger("NRT" + config.dataset_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(config.log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    config.logger = logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    columns = ["user_id", "item_id", "rating", "review"]
    data_df = pd.read_csv(config.data_path)[columns]
    word_dict, user_dict, item_dict = build_dictionnaries(data_df)
    word_dict.keep_most_frequent(config.vocab_size)

    train_df = data_df.sample(frac=config.train_size, random_state=config.seed)
    test_eval_df = data_df.drop(train_df.index)
    eval_size = config.eval_size / (config.eval_size + config.test_size)
    eval_df = test_eval_df.sample(frac=eval_size, random_state=config.seed)
    test_df = test_eval_df.drop(eval_df.index)

    train_dataset = RatingReviewDataset(config, train_df, word_dict, user_dict, item_dict)
    eval_dataset = RatingReviewDataset(config, eval_df, word_dict, user_dict, item_dict)
    test_dataset = RatingReviewDataset(config, test_df, word_dict, user_dict, item_dict)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    config.n_users = len(user_dict)
    config.n_items = len(item_dict)
    config.n_tokens = len(word_dict)
    config.bos_idx = word_dict.word2idx['<bos>']
    config.pad_idx = word_dict.word2idx['<pad>']

    model = NRT(
        n_users=config.n_users,
        n_items=config.n_items,
        n_tokens=config.n_tokens,
        d_model=config.d_model,
        hidden_size=config.hidden_size,
        n_layers_mlp=config.n_layers_mlp,
        max_rating=config.max_rating,
        min_rating=config.min_rating
    )
    if config.load_model:
        config.logger.info("Load model")
        model.load(config.save_model_path)
    model.to(config.device)

    if config.verbose:
        log = "NRT\n\n"
        for k, v in vars(config).items():
            log += f"{k}: {v}\n"
        log += f"{'-' * 80}\n\n"
        log += f"{train_df.head(5)}\n"
        config.logger.info("\n" + log)

    config.logger.info("Start training")
    train_infos, eval_infos = trainer(model, config, train_dataloader, eval_dataloader, word_dict)
    config.logger.info("Training done")

    config.logger.info("Start testing")
    model.load(config.save_model_path)
    model.to(config.device)
    rating_scores, reviews_scores, output_df = generate_and_eval(model, config, test_dataloader, word_dict)
    config.logger.info("Testing done")
    test_infos = {
        "rating": rating_scores,
        "review": reviews_scores
    }

    config.logger.info("Save results")
    output_df.to_csv(os.path.join(config.save_dir, "output.csv"), index=False)
    results = {"test": test_infos, "train": train_infos, "eval": eval_infos}
    config.logger.info(f"Results: {results}")
    with open(config.res_file_path, "w") as res_file:
        json.dump(results, res_file)
    config.logger.info("Results saved")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NRT')

    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument("--train_size", type=float, default=0.6)
    parser.add_argument("--eval_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--load_model', action=argparse.BooleanOptionalAction)
    parser.set_defaults(load_model=False)

    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--min_rating', type=float, default=1.)
    parser.add_argument('--max_rating', type=float, default=5.)

    parser.add_argument('--d_model', type=int, default=512,
                        help='size of user/item embeddings')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='number of hidden units')
    parser.add_argument('--n_layers_mlp', type=int, default=4,
                        help='number of layers in MLP')
    parser.add_argument('--review_length', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='keep the most frequent words in the vocabulary')
    
    parser.add_argument('--lambda_rating', type=float, default=1.0,
                        help='weight of the rating loss')
    parser.add_argument('--lambda_review', type=float, default=1.0,
                        help='weight of the review loss')
    parser.add_argument('--lambda_reg', type=float, default=1e-3,
                        help='weight of the regularization loss')

    parser.add_argument('--n_epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=5.0,
                        help='gradient clipping')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)

    parser.add_argument("--lower_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(lower_flag=True)
    parser.add_argument("--delete_stopwords_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_stopwords_flag=False)
    parser.add_argument("--delete_punctuation_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_punctuation_flag=False)
    parser.add_argument("--delete_non_ascii_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_non_ascii_flag=True)
    parser.add_argument("--replace_maj_word_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(replace_maj_word_flag=False)
    parser.add_argument("--stem_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(stem_flag=False)
    parser.add_argument("--lemmatize_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(lemmatize_flag=False)

    config = parser.parse_args()
    main(config)
