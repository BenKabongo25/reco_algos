# Ben Kabongo
# January 2025

import heapq
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from typing import *

from utils import preprocess_text


class ReviewDataset(Dataset):

    def __init__(self, config, data_df, word_dict, user_dict, item_dict):
        super().__init__()
        self.config = config
        self.data_df = data_df
        self.word_dict = word_dict
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.reviews = []
        self.reviews_ids = []
        self._reviews2ids()

    def _reviews2ids(self):
        for i in tqdm(range(len(self)), desc="Tokenization", colour="green"):
            review = self.data_df.iloc[i]["review"]
            review = preprocess_text(review, self.config, self.config.review_length)
            self.reviews.append(review)

            ids = self.word_dict.tokens2ids(review.split())
            ids = sentence_format(
                ids, 
                self.config.review_length, 
                self.word_dict.word2idx['<pad>'], 
                self.word_dict.word2idx['<bos>'], 
                self.word_dict.word2idx['<eos>']
            )
            self.reviews_ids.append(ids)

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        user_id = self.user_dict.entity2idx[row["user_id"]]
        item_id = self.item_dict.entity2idx[row["item_id"]]
        review = self.reviews[index]
        review_ids = self.reviews_ids[index]

        return {
            "user_id": user_id,
            "item_id": item_id,
            "review": review,
            "review_ids": review_ids
        }
    

def collate_fn(batch):
    collated_batch = {}
    for key in batch[0]:
        collated_batch[key] = [d[key] for d in batch]
        if isinstance(collated_batch[key][0], torch.Tensor):
            collated_batch[key] = torch.cat(collated_batch[key], 0)
    return collated_batch


def sentence_format(sentence, max_text_length, pad, bos, eos):
    length = len(sentence)
    if length >= max_text_length:
        return [bos] + sentence[:max_text_length] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (max_text_length - length)


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

    def ids2tokens(self, ids):
        eos = self.word2idx['<eos>']
        tokens = []
        for i in ids:
            if i == eos:
                break
            tokens.append(self.idx2word[i])
        return tokens
    
    def tokens2ids(self, tokens):
        return [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]


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


def sentence_format(sentence, max_text_length, pad, bos, eos):
    length = len(sentence)
    if length >= max_text_length:
        return [bos] + sentence[:max_text_length] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (max_text_length - length)


def build_dictionnaries(data_df):
    word_dict = WordDictionary()
    user_dict = EntityDictionary()
    item_dict = EntityDictionary()
    for _, row in data_df.iterrows():
        user_dict.add_entity(row['user_id'])
        item_dict.add_entity(row['item_id'])
        word_dict.add_sentence(row['review'])
    return word_dict, user_dict, item_dict
