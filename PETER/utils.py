import os
import math
import torch
import heapq
import random
import pickle
import datetime



class WordDictionary:
    def __init__(self):
        self.idx2word = ['<bos>', '<eos>', '<pad>', '<unk>']
        self.__predefine_num = len(self.idx2word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.__word2count = {}

    def add_sentence(self, sentence):
        for w in sentence.split():
            self.add_word(w)

    def add_word(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
            self.__word2count[w] = 1
        else:
            self.__word2count[w] += 1

    def __len__(self):
        return len(self.idx2word)

    def keep_most_frequent(self, max_vocab_size=20000):
        if len(self.__word2count) > max_vocab_size:
            frequent_words = heapq.nlargest(max_vocab_size, self.__word2count, key=self.__word2count.get)
            self.idx2word = self.idx2word[:self.__predefine_num] + frequent_words
            self.word2idx = {w: i for i, w in enumerate(self.idx2word)}


class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


def sentence_format(sentence, max_len, pad, bos, eos):
    length = len(sentence)
    if length >= max_len:
        return [bos] + sentence[:max_len] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (max_len - length)


class Batchify:
    def __init__(self, data, word2idx, seq_len=15, batch_size=128, shuffle=False):
        bos = word2idx['<bos>']
        eos = word2idx['<eos>']
        pad = word2idx['<pad>']
        u, i, r, t, f = [], [], [], [], []
        for x in data:
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'])
            t.append(sentence_format(x['text'], seq_len, pad, bos, eos))
            f.append([x['feature']])

        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.seq = torch.tensor(t, dtype=torch.int64).contiguous()
        self.feature = torch.tensor(f, dtype=torch.int64).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        feature = self.feature[index]  # (batch_size, 1)
        return user, item, rating, seq, feature


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def ids2tokens(ids, word2idx, idx2word):
    eos = word2idx['<eos>']
    tokens = []
    for i in ids:
        if i == eos:
            break
        tokens.append(idx2word[i])
    return tokens



########################################################################################################################
########################################################################################################################
########################################################################################################################

    

import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer

import evaluate
import numpy as np
import torch
import torch.nn.functional as F

from typing import Any, Dict, List


class DataLoaderFromDataFrameForReviewGeneration:
    
    def __init__(self, data_df, vocab_size, train_size=0.8, eval_size=0.1, test_size=0.1, seed=42):
        self.word_dict = WordDictionary()
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.data_df = data_df
        self.train_size = train_size
        self.eval_size = eval_size
        self.test_size = test_size
        self.seed = seed
        self.initialize()
        self.word_dict.keep_most_frequent(vocab_size)
        self.__unk = self.word_dict.word2idx['<unk>']
        self.feature_set = set()
        self.train, self.valid, self.test = self.load_data()

    def initialize(self):
        for i in range(len(self.data_df)):
            row = self.data_df.iloc[i]
            user_id = row['user_id']
            item_id = row['item_id']
            rating = row['rating']
            review = row['review']
            self.user_dict.add_entity(user_id)
            self.item_dict.add_entity(item_id)
            self.word_dict.add_sentence(review)
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self):
        train_df = self.data_df.sample(frac=self.train_size, random_state=self.seed)
        test_eval_df = self.data_df.drop(train_df.index)
        eval_size = self.eval_size / (self.eval_size + self.test_size)
        eval_df = test_eval_df.sample(frac=eval_size, random_state=self.seed)
        test_df = test_eval_df.drop(eval_df.index)

        data = {'train': [], 'valid': [], 'test': []}
        dfs = {'train': train_df, 'valid': eval_df, 'test': test_df}
        for key in dfs:
            for i in range(len(dfs[key])):
                row = dfs[key].iloc[i]
                user_id = row['user_id']
                item_id = row['item_id']
                rating = row['rating']
                review = row['review']
                data[key].append({'user': self.user_dict.entity2idx[user_id],
                                  'item': self.item_dict.entity2idx[item_id],
                                  'rating': rating,
                                  'text': self.seq2ids(review),
                                  'feature': self.__unk})
                self.feature_set.add('<unk>')

        return data['train'], data['valid'], data['test']

    def seq2ids(self, seq):
        return [self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()]


def delete_punctuation(text: str) -> str:
    punctuation = r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\n\t]"
    text = re.sub(punctuation, " ", text)
    text = re.sub('( )+', ' ', text)
    return text


def delete_stopwords(text: str) -> str:
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return ' '.join([w for w in text.split() if w not in stop_words])


def delete_non_ascii(text: str) -> str:
    return ''.join([w for w in text if ord(w) < 128])


def replace_maj_word(text: str) -> str:
    token = '<MAJ>'
    return ' '.join([w if not w.isupper() else token for w in delete_punctuation(text).split()])


def delete_digit(text: str) -> str:
    return re.sub('[0-9]+', '', text)


def first_line(text: str) -> str:
    return re.split(r'[.!?]', text)[0]


def last_line(text: str) -> str:
    if text.endswith('\n'): text = text[:-2]
    return re.split(r'[.!?]', text)[-1]


def delete_balise(text: str) -> str:
    return re.sub("<.*?>", "", text)


def stem(text: str) -> str:
    stemmer = EnglishStemmer()
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = " ".join(stemmed_tokens)
    return stemmed_text


def lemmatize(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_text = " ".join(lemmatized_tokens)
    return lemmatized_text


def preprocess_text(text: str, args: Any, max_length: int=-1) -> str:
    text = str(text).strip()
    if getattr(args, "replace_maj_word_flag", False): text = replace_maj_word(text)
    if getattr(args, "lower_flag", False): text = text.lower()
    if getattr(args, "delete_punctuation_flag", False): text = delete_punctuation(text)
    if getattr(args, "delete_balise_flag", False): text = delete_balise(text)
    if getattr(args, "delete_stopwords_flag", False): text = delete_stopwords(text)
    if getattr(args, "delete_non_ascii_flag", False): text = delete_non_ascii(text)
    if getattr(args, "delete_digit_flag", False): text = delete_digit(text)
    if getattr(args, "first_line_flag", False): text = first_line(text)
    if getattr(args, "last_line_flag", False): text = last_line(text)
    if getattr(args, "stem_flag", False): text = stem(text)
    if getattr(args, "lemmatize_flag", False): text = lemmatize(text)
    if max_length > 0 and args.truncate_flag:
        text = str(text).strip().split()
        if len(text) > max_length:
            text = text[:max_length]
        text = " ".join(text)
    return text


def postprocess_text(text: str, special_tokens=[]) -> str:
    for token in special_tokens:
        text = text.replace(token, "")
    text = re.sub(r" \'(s|m|ve|d|ll|re)", r"'\1", text)
    text = re.sub(r" \(", "(", text)
    text = re.sub(r" \)", ")", text)
    text = re.sub(r" ,", ",", text)
    text = re.sub(r" \.", ".", text)
    text = re.sub(r" !", "!", text)
    text = re.sub(r" \?", "?", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text



RMSE = "rmse"
MAE = "mae"
RATING_METRICS = [RMSE, MAE]

PRECISION = "precision"
RECALL = "recall"
F1 = "f1"
NDCG = "ndcg"
MAP = "map"
RANKING_METRICS = [PRECISION, RECALL, F1, NDCG, MAP]

BLEU = "bleu"
METEOR = "meteor"
ROUGE = "rouge"
BERTSCORE = "bertscore"
REVIEW_METRICS = [BLEU, METEOR, ROUGE, BERTSCORE]


def rating_evaluation(
        config: Any, predictions: List[float], references: List[float], 
        users: List[float]=None, metrics: List[str]=[RMSE, MAE],
    ) -> Dict[str, float]:
    
    results = {}
    
    actual_ratings = torch.tensor(references, dtype=torch.float32).to(config.device)
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32).to(config.device)

    if RMSE in metrics:
        rmse = torch.sqrt(F.mse_loss(predictions_tensor, actual_ratings))
        results[RMSE] = rmse.item()
        
    if MAE in metrics:
        mae = F.l1_loss(predictions_tensor, actual_ratings)
        results[MAE] = mae.item()
    mae = F.l1_loss(predictions_tensor, actual_ratings)

    if not set(metrics).intersection(RANKING_METRICS):
        return results

    threshold = torch.tensor(config.threshold_rating).to(config.device)
    actual_binary = (actual_ratings >= threshold).float()
    predicted_binary = (predictions_tensor >= threshold).float()
    true_positives = (predicted_binary * actual_binary).sum()

    precision = true_positives / predicted_binary.sum() if predicted_binary.sum() > 0 else torch.tensor(1.0).to(config.device)
    recall = true_positives / actual_binary.sum() if actual_binary.sum() > 0 else torch.tensor(1.0).to(config.device)
    
    if PRECISION in metrics:
        results[PRECISION] = precision.item()
    
    if RECALL in metrics:
        results[RECALL] = recall.item()
    
    if F1 in metrics:
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0).to(config.device)
        results[F1] = f1.item()

    if not set(metrics).intersection([NDCG, MAP]):
        return results
    
    users = list(set(users))
    
    if NDCG in metrics:
        ndcg_scores = []

    if MAP in metrics:
        average_precisions = []
    
    for user in users:
        user_predictions = [pred for idx, pred in enumerate(predictions) if users[idx] == user]
        user_references = [ref for idx, ref in enumerate(references) if users[idx] == user]

        user_predictions_tensor = torch.tensor(user_predictions, dtype=torch.float32).to(config.device)
        
        sorted_indices = torch.configort(user_predictions_tensor, descending=True)
        relevances = torch.tensor(user_references, dtype=torch.float32).to(config.device)[sorted_indices]

        if NDCG in metrics:
            ndcg = ndcg_at_k(relevances, config.k, config.device)
            ndcg_scores.append(ndcg.item())

        if MAP in metrics:
            ap = average_precision(relevances, config.threshold_rating, config.device)
            average_precisions.append(ap.item())

    if NDCG in metrics:
        ndcg = torch.tensor(ndcg_scores).mean().item()
        results[NDCG] = ndcg

    if MAP in metrics:
        map = torch.tensor(average_precisions).mean().item()
        results[MAP] = map

    return results


def review_evaluation(
        config, predictions: List[str], references: List[str],
        metrics: List[str]=[BLEU, METEOR, ROUGE, BERTSCORE]
    ) -> Dict[str, Any]:

    results = {}
    references_list = [[ref] for ref in references]

    if BLEU in metrics:
        bleu_metric = evaluate.load("bleu")
        bleu_results = bleu_metric.compute(predictions=predictions, references=references_list)
        results[BLEU] = bleu_results["bleu"]

    if METEOR in metrics:
        meteor_metric = evaluate.load("meteor")
        meteor_results = meteor_metric.compute(predictions=predictions, references=references)
        results[METEOR] = meteor_results["meteor"]

    if ROUGE in metrics:
        rouge_metric = evaluate.load("rouge")
        rouge_results = rouge_metric.compute(predictions=predictions, references=references)
        results[ROUGE + ".1"] = rouge_results["rouge1"]
        results[ROUGE + ".2"] = rouge_results["rouge2"]
        results[ROUGE + ".L"] = rouge_results["rougeL"]
        results[ROUGE + ".Lsum"] = rouge_results["rougeLsum"]

    if BERTSCORE in metrics:
        bertscore_metric = evaluate.load("bertscore")
        bertscore_results = bertscore_metric.compute(
            predictions=predictions, references=references, lang=config.lang, device=config.device
        )
        results[BERTSCORE + ".precision"] = np.mean(bertscore_results["precision"])
        results[BERTSCORE + ".recall"] = np.mean(bertscore_results["recall"])
        results[BERTSCORE + ".f1"] = np.mean(bertscore_results["f1"])

    return results


def dcg_at_k(relevances, k, device):
    relevances = relevances[:k]
    positions = torch.arange(1, len(relevances) + 1, dtype=torch.float32).to(device)
    return torch.sum(relevances / torch.log2(positions + 1))


def ndcg_at_k(relevances, k, device):
    dcg = dcg_at_k(relevances, k, device)
    ideal_relevances = torch.sort(relevances, descending=True).values
    idcg = dcg_at_k(ideal_relevances, k, device)
    return dcg / idcg if idcg > 0 else torch.tensor(0.0).to(device)


def average_precision(relevances, threshold_rating, device):
    precisions = []
    num_relevant = 0
    for k in range(1, len(relevances) + 1):
        if relevances[k - 1] >= threshold_rating:
            num_relevant += 1
            precisions.append(num_relevant / k)

    if num_relevant == 0:
        return torch.tensor(0.0).to(device)
    
    return torch.tensor(precisions).mean().to(device)
