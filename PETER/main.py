# Ben Kabongo
# December 2024

# PEPLER for review generation
# https://dl.acm.org/doi/pdf/10.1145/3580488


import argparse
import json
import logging
import math
import pandas as pd
import os
import sys
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from module import PETER
from data import RatingReviewFeatureDataset, build_dictionnaries

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from utils.evaluation import rating_evaluation_pytorch, review_evaluation
from utils.text import postprocess_text
from utils.functions import set_seed


def empty_cache():
    with torch.no_grad():
        torch.cuda.empty_cache()


def train(model, config, optimizer, rating_criterion, text_criterion, dataloader):
    model.train()

    total_loss = 0.
    text_loss = 0.
    context_loss = 0.
    rating_loss = 0.

    progress_bar = tqdm(enumerate(dataloader, 1), desc="Training", colour="cyan", total=len(dataloader))
    for batch_idx, batch in progress_bar:
        empty_cache()
        user = torch.LongTensor(batch['user_id']).to(config.device)
        item = torch.LongTensor(batch['item_id']).to(config.device)
        rating = torch.tensor(batch['rating'].clone().detach(), dtype=torch.float32).to(config.device)
        seq = torch.LongTensor(batch['review_ids']).transpose(1, 0).to(config.device)

        if config.use_features:
            features = torch.LongTensor(batch['features']).transpose(1, 0).to(config.device)
            text = torch.cat([features, seq[:-1]], 0) 
        else:
            text = seq[:-1]

        log_word_prob, log_context_dis, rating_p, _ = model(user, item, text)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
        context_dis = log_context_dis.unsqueeze(0).repeat((config.review_length - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
        c_loss = text_criterion(context_dis.view(-1, config.n_tokens), seq[1:-1].reshape((-1,)))
        r_loss = rating_criterion(rating_p, rating)
        t_loss = text_criterion(log_word_prob.view(-1, config.n_tokens), seq[1:].reshape((-1,)))
        loss = config.lambda_text * t_loss + config.lambda_context * c_loss + config.lambda_rating * r_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()

        total_loss += loss.item()
        text_loss += t_loss.item()
        context_loss += c_loss.item()
        rating_loss += r_loss.item()
        
    total_loss /= len(dataloader)
    text_loss /= len(dataloader)
    context_loss /= len(dataloader)
    rating_loss /= len(dataloader)
    return {'total_loss': total_loss, 'text_loss': text_loss, 'context_loss': context_loss, 'rating_loss': rating_loss}


def evaluate(model, config, rating_criterion, text_criterion, dataloader):
    model.eval()

    total_loss = 0.
    text_loss = 0.
    context_loss = 0.
    rating_loss = 0.

    progress_bar = tqdm(enumerate(dataloader, 1), desc="Evaluation", colour="cyan", total=len(dataloader))
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            empty_cache()
            user = torch.LongTensor(batch['user_id']).to(config.device)
            item = torch.LongTensor(batch['item_id']).to(config.device)
            rating = torch.tensor(batch['rating'].clone().detach(), dtype=torch.float32).to(config.device)
            seq = torch.LongTensor(batch['review_ids']).transpose(1, 0).to(config.device)

            if config.use_features:
                features = torch.LongTensor(batch['features']).transpose(1, 0).to(config.device)
                text = torch.cat([features, seq[:-1]], 0) 
            else:
                text = seq[:-1]

            log_word_prob, log_context_dis, rating_p, _ = model(user, item, text)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
            context_dis = log_context_dis.unsqueeze(0).repeat((config.review_length + 1 - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
            c_loss = text_criterion(context_dis.view(-1, config.n_tokens), seq[1:-1].reshape((-1,)))
            r_loss = rating_criterion(rating_p, rating)
            t_loss = text_criterion(log_word_prob.view(-1, config.n_tokens), seq[1:].reshape((-1,)))
            loss = config.lambda_text * t_loss + config.lambda_context * c_loss + config.lambda_rating * r_loss

            total_loss += loss.item()
            text_loss += t_loss.item()
            context_loss += c_loss.item()
            rating_loss += r_loss.item()
        
    total_loss /= len(dataloader)
    text_loss /= len(dataloader)
    context_loss /= len(dataloader)
    rating_loss /= len(dataloader)
    return {'total_loss': total_loss, 'text_loss': text_loss, 'context_loss': context_loss, 'rating_loss': rating_loss}


def rating_and_evaluate(model, config, dataloader):
    model.eval()
    users = []
    items = []
    reference_ratings = []
    predict_ratings = []

    progress_bar = tqdm(enumerate(dataloader, 1), desc="Rating eval", colour="green", total=len(dataloader))
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            empty_cache()
            user = torch.LongTensor(batch['user_id']).to(config.device)
            item = torch.LongTensor(batch['item_id']).to(config.device)
            rating = torch.tensor(batch['rating'].clone().detach(), dtype=torch.float32).to(config.device)
            seq = torch.LongTensor(batch['review_ids']).transpose(1, 0).to(config.device)

            if config.use_features:
                features = torch.LongTensor(batch['features']).transpose(1, 0).to(config.device)
                text = torch.cat([features, seq[:-1]], 0) 
            else:
                text = seq[:-1]

            _, _, rating_p, _ = model(user, item, text)

            reference_ratings.extend(rating.cpu().detach().numpy().tolist())
            predict_ratings.extend(rating_p.cpu().detach().numpy().tolist())
            users.extend(user.cpu().detach().numpy().tolist())
            items.extend(item.cpu().detach().numpy().tolist())

    ratings_scores = rating_evaluation_pytorch(config, predictions=predict_ratings, references=reference_ratings)
    output_df = pd.DataFrame({
        'user_id': users, 
        'item_id': items, 
        'reference': reference_ratings,
        'prediction': predict_ratings
    })
    return {'rating': ratings_scores, 'output': output_df}


def ids2tokens(ids, word_dict, eos):
    tokens = []
    for i in ids:
        if i == eos:
            break
        tokens.append(word_dict.idx2word[i])
    text = " ".join(tokens)
    text = postprocess_text(text)
    return text


def predict(log_context_dis, topk):
    word_prob = log_context_dis.exp()  # (batch_size, ntoken)
    if topk == 1:
        context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
    else:
        context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
    return context 


def generate_and_evaluate(model, config, word_dict, dataloader):
    model.eval()

    users = []
    items = []
    reference_texts = []
    reference_ratings = []
    predict_texts = []
    predict_ratings = []
    predict_contexts = []

    progress_bar = tqdm(enumerate(dataloader, 1), desc="Generate", colour="green", total=len(dataloader))
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            empty_cache()
            user = torch.LongTensor(batch['user_id']).to(config.device)
            item = torch.LongTensor(batch['item_id']).to(config.device)
            rating = torch.tensor(batch['rating'].clone().detach(), dtype=torch.float32).to(config.device)
            seq = torch.LongTensor(batch['review_ids']).transpose(1, 0).to(config.device)
            review = batch['review']

            bos = seq[:, 0].unsqueeze(0) # (1, batch_size)
            if config.use_features:
                features = torch.LongTensor(batch['features']).transpose(1, 0).to(config.device)
                text = torch.cat([features, bos], 0)  # (src_len - 1, batch_size)
            else:
                text = bos  # (src_len - 1, batch_size)

            start_idx = text.size(0)
            for idx in range(config.review_length):
                # produce a word at each step
                if idx == 0:
                    log_word_prob, log_context_dis, rating_p, _ = model(user, item, text, False)  # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                    predict_ratings.extend(rating_p.cpu().detach().numpy().tolist())
                    context = predict(log_context_dis, topk=1)  # (batch_size, 1)
                    predict_contexts.extend(context.tolist())
                else:
                    log_word_prob, _, _, _ = model(user, item, text, False, False, False)  # (batch_size, ntoken)
                word_prob = log_word_prob.exp()  # (batch_size, ntoken)
                word_idx = torch.argmax(word_prob, dim=1)  # (batch_size,), pick the one with the largest probability
                text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
            ids = text[start_idx:].t().tolist()  # (batch_size, seq_len)

            users.extend(user.cpu().detach().numpy().tolist())
            items.extend(item.cpu().detach().numpy().tolist())
            reference_texts.extend(review)
            reference_ratings.extend(rating.cpu().detach().numpy().tolist())

            review_p = [ids2tokens(rids, word_dict, config.eos_idx) for rids in ids]
            predict_texts.extend(review_p)

    ratings_scores = rating_evaluation_pytorch(config, predictions=predict_ratings, references=reference_ratings)
    review_scores = review_evaluation(config, predictions=predict_texts, references=reference_texts)
    output_df = pd.DataFrame({
        'user_id': users, 
        'item_id': items, 
        'reference_review': reference_texts,
        'prediction_review': predict_texts,
        'reference_rating': reference_ratings,
        'prediction_rating': predict_ratings,
    })
    return {'text': review_scores, 'rating': ratings_scores, 'output': output_df}


def trainer(model, config, optimizer, rating_criterion, text_criterion, train_dataloader, val_dataloader):
    best_val_loss = float('inf')
    endure_count = 0

    train_infos = {"text_loss": [], "rating_loss": [], "context_loss": [], "total_loss": []}
    val_infos = {"text_loss": [], "rating_loss": [], "context_loss": [], "total_loss": []}

    for epoch in range(1, config.n_epochs + 1):
        config.logger.info('epoch {}'.format(epoch))

        train_epoch_infos = train(model, config, optimizer, rating_criterion, text_criterion, train_dataloader)
        val_epoch_infos = evaluate(model, config, rating_criterion, text_criterion, val_dataloader)

        train_infos['text_loss'].append(train_epoch_infos['text_loss'])
        train_infos['rating_loss'].append(train_epoch_infos['rating_loss'])
        train_infos['context_loss'].append(train_epoch_infos['context_loss'])
        train_infos['total_loss'].append(train_epoch_infos['total_loss'])
        val_infos['text_loss'].append(val_epoch_infos['text_loss'])
        val_infos['rating_loss'].append(val_epoch_infos['rating_loss'])
        val_infos['context_loss'].append(val_epoch_infos['context_loss'])
        val_infos['total_loss'].append(val_epoch_infos['total_loss'])

        val_loss = val_epoch_infos['total_loss']

        config.logger.info('text ppl {:4.4f} | rating loss {:4.4f} | valid loss {:4.4f} on validation'.format(
            math.exp(val_epoch_infos['text_loss']), val_epoch_infos['rating_loss'], val_loss))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(config.model_path, 'wb') as f:
                torch.save(model, f)
        else:
            endure_count += 1
            config.logger.info('Endured {} time(s)'.format(endure_count))
            if endure_count == config.endure_times:
                config.logger.info('Exiting from early stop')
                break

    return {'train': train_infos, 'val': val_infos}


def main(config):
    set_seed(config.seed)

    os.makedirs(config.save_dir, exist_ok=True)
    config.log_file_path = os.path.join(config.save_dir, "log.txt")
    config.res_file_path = os.path.join(config.save_dir, "res.json")
    config.save_model_path = os.path.join(config.save_dir, "model.pth")

    logger = logging.getLogger("PETER" + config.dataset_name)
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

    config.n_users = len(user_dict)
    config.n_items = len(item_dict)
    config.n_tokens = len(word_dict)
    config.bos_idx = word_dict.word2idx['<bos>']
    config.pad_idx = word_dict.word2idx['<pad>']
    config.eos_idx = word_dict.word2idx['<eos>']

    config.input_length = 2 # [u, i]
    if config.use_features:
        config.input_length = 2 + config.n_features  # [u, i, f]
    config.output_length = config.review_length + 1  # [r1, r2, ..., rL, <eos>]

    train_df = data_df.sample(frac=config.train_size, random_state=config.seed)
    test_eval_df = data_df.drop(train_df.index)
    eval_size = config.eval_size / (config.eval_size + config.test_size)
    eval_df = test_eval_df.sample(frac=eval_size, random_state=config.seed)
    test_df = test_eval_df.drop(eval_df.index)

    train_df['user_id'] = train_df['user_id'].apply(lambda u: user_dict[u])
    train_df['item_id'] = train_df['item_id'].apply(lambda i: item_dict[i])
    train_df['rating'] = train_df['rating'].astype(float)
    test_df['user_id'] = test_df['user_id'].apply(lambda u: user_dict[u])
    test_df['item_id'] = test_df['item_id'].apply(lambda i: item_dict[i])
    test_df['rating'] = test_df['rating'].astype(float)
    eval_df['user_id'] = eval_df['user_id'].apply(lambda u: user_dict[u])
    eval_df['item_id'] = eval_df['item_id'].apply(lambda i: item_dict[i])
    eval_df['rating'] = eval_df['rating'].astype(float)

    train_dataset = RatingReviewFeatureDataset(config, train_df, word_dict, user_dict, item_dict)
    eval_dataset = RatingReviewFeatureDataset(config, eval_df, word_dict, user_dict, item_dict)
    test_dataset = RatingReviewFeatureDataset(config, test_df, word_dict, user_dict, item_dict)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = PETER(
        peter_mask=config.peter_mask,
        src_len=config.iput_length,
        tgt_len=config.output_length,
        pad_idx=config.pad_idx,
        nuser=config.n_users, 
        nitem=config.n_items, 
        ntoken=config.n_tokens, 
        emsize=config.embedding_dim, 
        nhead=config.n_heads, 
        nhid=config.n_hiddens,
        nlayers=config.n_layers,
        dropout=config.dropout,
    )
    if config.load_model:
        config.logger.info("Load model")
        model.load(config.save_model_path)
    model.to(config.device)

    if config.verbose:
        log = "PETER\n\n"
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
    rating_scores, reviews_scores, output_df = generate_and_evaluate(model, config, test_dataloader, word_dict)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PErsonalized Transformer for Explainable Recommendation (PETER)')

    parser.add_argument('--data_path', type=str, default=None,
                        help='path for loading data')
    parser.add_argument('--train_size', type=float, default=0.8,
                        help='train size')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='validation size')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='test size')
    parser.add_argument('--train_path', type=str, default="",
                        help='path for training data')
    parser.add_argument('--val_path', type=str, default="",
                        help='path for validation data')
    parser.add_argument('--test_path', type=str, default="",
                        help='path for test data')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='report interval')
    parser.add_argument('--save_dir', type=str, default='./PEPLER/',
                        help='directory to save the final model')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--load_model', action=argparse.BooleanOptionalAction)
    parser.set_defaults(load_model=False)
    
    parser.add_argument('--threshold_rating', type=float, default=4.0,
                        help='threshold for rating')
    parser.add_argument('--ranking_metrics_flag', action=argparse.BooleanOptionalAction, default=True,
                        help='whether to compute ranking metrics')
    parser.add_argument('--lang', type=str, default='en',
                        help='language')

    parser.add_argument('--endure_times', type=int, default=5,
                        help='the maximum endure times of loss increasing on validation')
    parser.add_argument('--rating_reg', type=float, default=0.01,
                        help='regularization on recommendation task')
    parser.add_argument('--text_reg', type=float, default=1.0,
                        help='regularization on text generation task')
    parser.add_argument('--use_mf', action=argparse.BooleanOptionalAction, default=False,
                        help='otherwise MLP')
    parser.add_argument('--review_length', type=int, default=128,
                        help='number of words to generate for each sample')
    
    parser.add_argument("--truncate_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(truncate_flag=True)
    parser.add_argument("--lower_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(lower_flag=True)
    parser.add_argument("--delete_balise_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_balise_flag=True)
    parser.add_argument("--delete_non_ascii_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_non_ascii_flag=True)
    
    config = parser.parse_args()
    main(config)
