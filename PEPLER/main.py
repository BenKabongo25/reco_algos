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
from transformers import GPT2Tokenizer

from module import PEPLER
from data import RatingReviewDataset

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from utils.evaluation import rating_evaluation_pytorch, review_evaluation
from utils.text import postprocess_text



def train(model, config, optimizer, rating_criterion, dataloader):
    model.train()
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0

    progress_bar = tqdm(enumerate(dataloader, 1), desc="Training", colour="green", total=len(dataloader))
    for batch_idx, batch in progress_bar:
        user = batch['user_id'].to(config.device)
        item = batch['item_id'].to(config.device)
        rating = batch['rating'].to(config.device)
        seq = batch['tokens'].to(config.device)
        mask = batch['mask'].to(config.device)
        
        optimizer.zero_grad()
        outputs, rating_p = model(user, item, seq, mask)
        t_loss = outputs.loss
        r_loss = rating_criterion(rating_p, rating)
        loss = config.text_reg * t_loss + config.rating_reg * r_loss
        loss.backward()
        optimizer.step()

        batch_size = user.size(0)
        text_loss += batch_size * t_loss.item()
        rating_loss += batch_size * r_loss.item()
        total_sample += batch_size

        if batch_idx % config.log_interval == 0 or batch_idx == len(dataloader):
            cur_t_loss = text_loss / total_sample
            cur_r_loss = rating_loss / total_sample
            description = 'text ppl {:4.4f} | rating loss {:4.4f} | {:5d}/{:5d} batches'.format(
                math.exp(cur_t_loss), cur_r_loss, batch_idx, len(dataloader))
            logging.info(description)
            progress_bar.set_description(description)
            text_loss = 0.
            rating_loss = 0.
            total_sample = 0

    text_loss /= total_sample
    rating_loss /= total_sample
    return {'text_loss': text_loss, 'rating_loss': rating_loss}


def evaluate(model, config, rating_criterion, dataloader):
    model.eval()
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0

    progress_bar = tqdm(enumerate(dataloader, 1), desc="Eval", colour="green", total=len(dataloader))
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            user = batch['user_id'].to(config.device)
            item = batch['item_id'].to(config.device)
            rating = batch['rating'].to(config.device)
            seq = batch['tokens'].to(config.device)
            mask = batch['mask'].to(config.device)

            outputs, rating_p = model(user, item, seq, mask)
            t_loss = outputs.loss
            r_loss = rating_criterion(rating_p, rating)

            batch_size = user.size(0)
            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            total_sample += batch_size
    
    text_loss /= total_sample
    rating_loss /= total_sample
    return {'text_loss': text_loss, 'rating_loss': rating_loss}


def ids2tokens(ids, tokenizer, eos):
    text = tokenizer.decode(ids)
    text = postprocess_text(text)
    tokens = []
    for token in text.split():
        if token == eos:
            break
        tokens.append(token)
    return tokens


def generate_and_evaluate(model, config, tokenizer, dataloader):
    model.eval()

    reference_texts = []
    reference_ratings = []
    predict_texts = []
    predict_ratings = []

    progress_bar = tqdm(enumerate(dataloader, 1), desc="Generate", colour="green", total=len(dataloader))
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            user = batch['user_id'].to(config.device)
            item = batch['item_id'].to(config.device)
            rating = batch['rating'].to(config.device)
            seq = batch['tokens'].to(config.device)
            review = batch['review']

            reference_texts.extend(review)
            reference_ratings.extend(rating.cpu().detach().numpy().tolist())

            text = seq[:, :1].to(config.device)  # bos, (batch_size, 1)
            for idx in range(seq.size(1)):
                # produce a word at each step
                if idx == 0:
                    outputs, rating_p = model(user, item, text, None)
                    predict_ratings.extend(rating_p.cpu().detach().numpy().tolist())
                else:
                    outputs, _ = model(user, item, text, None, False)
                last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
            review_p = ids2tokens(ids, tokenizer, config.eos)
            predict_texts.extend(review_p)

    ratings_scores = rating_evaluation_pytorch(config, predictions=predict_ratings, references=reference_ratings)
    review_scores = review_evaluation(config, predictions=predict_texts, references=reference_texts)

    return {'text': review_scores, 'rating': ratings_scores}


def trainer(model, config, optimizer, rating_criterion, train_dataloader, val_dataloader):
    best_val_loss = float('inf')
    endure_count = 0

    train_infos = {"text_loss": [], "rating_loss": []}
    val_infos = {"text_loss": [], "rating_loss": []}

    for epoch in range(1, config.n_epochs + 1):
        logging.info('epoch {}'.format(epoch))

        train_epoch_infos = train(model, config, optimizer, rating_criterion, train_dataloader)
        val_epoch_infos = evaluate(model, config, rating_criterion, val_dataloader)

        train_infos['text_loss'].append(train_epoch_infos['text_loss'])
        train_infos['rating_loss'].append(train_epoch_infos['rating_loss'])
        val_infos['text_loss'].append(val_epoch_infos['text_loss'])
        val_infos['rating_loss'].append(val_epoch_infos['rating_loss'])

        val_loss = val_epoch_infos['text_loss'] + val_epoch_infos['rating_loss']

        logging.info('text ppl {:4.4f} | rating loss {:4.4f} | valid loss {:4.4f} on validation'.format(
            math.exp(val_epoch_infos['text_loss']), val_epoch_infos['rating_loss'], val_loss))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(config.model_path, 'wb') as f:
                torch.save(model, f)
        else:
            endure_count += 1
            logging.info('Endured {} time(s)'.format(endure_count))
            if endure_count == config.endure_times:
                logging.info('Cannot endure it anymore | Exiting from early stop')
                break

    return {'train': train_infos, 'val': val_infos}


def main(config):
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config.save_dir, exist_ok=True)
    config.model_path = os.path.join(config.save_dir, 'model.pt')
    config.results_path = os.path.join(config.save_dir, 'results.json')

    logging.basicConfig(filename=os.path.join(config.save_dir, 'log.txt'), level=logging.INFO)

    # Load data
    logging.info('Loading data')
    config.bos = '<bos>'
    config.eos = '<eos>'
    config.pad = '<pad>'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=config.bos, eos_token=config.eos, pad_token=config.pad)

    columns = ['user_id', 'item_id', 'rating', 'review']
    if config.data_path != "":
        data_df = pd.read_csv(config.data_path)[columns]
        train_df = data_df.sample(frac=config.train_size, random_state=config.seed)
        test_val_df = data_df.drop(train_df.index)
        test_size = config.test_size / (config.test_size + config.val_size)
        test_df = test_val_df.sample(frac=test_size, random_state=config.seed)
        val_df = None if config.val_size == 0 else test_val_df.drop(test_df.index)

    else:
        train_df = pd.read_csv(config.train_path)[columns]
        test_df = pd.read_csv(config.test_path)[columns]
        data_df = pd.concat([train_df, test_df])
        val_df = None
        if config.val_path != "":
            val_df = pd.read_csv(config.val_path)[columns]
            data_df = pd.concat([data_df, val_df])

    train_size, test_size, val_size = len(train_df), len(test_df), len(val_df) if val_df is not None else 0
    logging.info("Data shape: {}".format(data_df.shape))
    logging.info("Train size: {}, Test size: {}, Validation size: {}".format(train_size, test_size, val_size))
    logging.info({}.format(data_df.head()))

    users_vocab = {u: i for i, u in enumerate(data_df['user_id'].unique())}
    items_vocab = {i: j for j, i in enumerate(data_df['item_id'].unique())}
    config.n_users = len(users_vocab)
    config.n_items = len(items_vocab)

    train_df['user_id'] = train_df['user_id'].apply(lambda u: users_vocab[u])
    train_df['item_id'] = train_df['item_id'].apply(lambda i: items_vocab[i])
    train_df['rating'] = train_df['rating'].astype(float)
    test_df['user_id'] = test_df['user_id'].apply(lambda u: users_vocab[u])
    test_df['item_id'] = test_df['item_id'].apply(lambda i: items_vocab[i])
    test_df['rating'] = test_df['rating'].astype(float)
    if val_df is not None:
        val_df['user_id'] = val_df['user_id'].apply(lambda u: users_vocab[u])
        val_df['item_id'] = val_df['item_id'].apply(lambda i: items_vocab[i])
        val_df['rating'] = val_df['rating'].astype(float)

    train_dataset = RatingReviewDataset(config, train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset = RatingReviewDataset(config, test_df, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    val_dataloader = None
    if val_df is not None:
        val_dataset = RatingReviewDataset(config, val_df, tokenizer)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Build the model
    config.n_tokens = len(tokenizer)
    model = PEPLER.from_pretrained('gpt2', config.n_users, config.n_items, config.use_mf)
    model.resize_token_embeddings(config.n_tokens)  # three tokens added, update embedding table
    model.to(config.device)

    rating_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Train the model
    logging.info('Start training')
    results = trainer(model, config, optimizer, rating_criterion, train_dataloader, val_dataloader)

    # Test
    with open(config.model_path, 'rb') as f:
        model = torch.load(f).to(config.device)
    logging.info('Start testing')
    test_results = generate_and_evaluate(model, config, tokenizer, test_dataloader)
    results['test'] = test_results

    # Save results
    logging.info('Saving results')
    with open(config.results_path, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PErsonalized Prompt Learning for Explainable Recommendation (PEPLER)')

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
    
    config = parser.parse_args()
    main(config)
