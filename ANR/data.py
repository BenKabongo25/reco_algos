# Ben Kabongo
# December 2024

import nltk
import numpy as np
import pandas as pd
import re
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Any


class AspectRatingsDataset(Dataset):
	
	def __init__(self, config: Any, data_df: pd.DataFrame):
		super().__init__()
		self.data_df = data_df
		self.config = config
	
	def __len__(self) -> int:
		return len(self.data_df)

	def __getitem__(self, index):
		row = self.data_df.iloc[index]
		user_id = row["user_id"]
		item_id = row["item_id"]
		overall_rating = row["rating"]

		_out = {
			"user_id": user_id,
			"item_id": item_id,
			"overall_rating": overall_rating,
		}

		if self.config.aspects_flag:
			aspects_ratings = [row[aspect] for aspect in self.config.aspects]
			_out["aspects_ratings"] = aspects_ratings
			
		return _out


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


def process_text(text: str) -> str:
	text = text.lower()
	text = delete_punctuation(text)
	text = delete_stopwords(text)
	text = delete_non_ascii(text)
	return text


def process_data(config, data_df, users, items, words_embeddings):
	data_df = data_df.reset_index(drop=True)
	
	reviews = []
	vocabulary = {}

	for index in tqdm(range(len(data_df)), desc="Create vocabulary", total=len(data_df), colour="cyan"):
		row = data_df.iloc[index]
		review = process_text(row["review"])
		words = nltk.word_tokenize(review)
		review_words = []
		for word in words:
			if word in words_embeddings.keys():
				if word not in vocabulary:
					vocabulary[word] = 0
				vocabulary[word] += 1
				review_words.append(word)
		reviews.append(review_words)

	if config.vocab_size == -1:
		config.vocab_size = len(vocabulary)
	else:
		vocabulary = {k: v for k, v in sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)[:config.vocab_size]}
		for index in tqdm(range(len(data_df)), desc="Delete unknown", total=len(data_df), colour="cyan"):
			review = reviews[index]
			reviews[index] = [word for word in review if word in vocabulary]
	
	vocabulary = list(vocabulary.keys())
	for index in tqdm(range(len(data_df)), desc="To ids", total=len(data_df), colour="cyan"):
		review = reviews[index]
		reviews[index] = [vocabulary.index(word) for word in review]
	
	vocab_words_embeddings = [words_embeddings[word] for word in vocabulary]
	vocabulary.append("<PAD>")
	pad_index = len(vocabulary) - 1
	vocab_words_embeddings.append([0] * config.d_words)
	vocab_words_embeddings = np.array(vocab_words_embeddings)
	
	users_document = {}
	for uid in tqdm(range(len(users)), desc="User document", total=len(users), colour="cyan"):
		user_reviews = data_df[data_df["user_id"] == users[uid]]
		user_document = []
		for index in user_reviews.index:
			review = reviews[index]
			user_document.extend(review)
		if len(user_document) < config.doc_len:
			user_document.extend([pad_index] * (config.doc_len - len(user_document)))
		users_document[uid] = user_document[:config.doc_len]
	users_document = np.array([users_document[uid] for uid in range(len(users))])

	items_document = {}
	for iid in tqdm(range(len(items)), desc="Item document", total=len(items), colour="cyan"):
		item_reviews = data_df[data_df["item_id"] == items[iid]]
		item_document = []
		for index in item_reviews.index:
			review = reviews[index]
			item_document.extend(review)
		if len(item_document) < config.doc_len:
			item_document.extend([pad_index] * (config.doc_len - len(item_document)))
		items_document[iid] = item_document[:config.doc_len]
	items_document = np.array([items_document[iid] for iid in range(len(items))])

	_out = {
		"users_document": users_document,
		"items_document": items_document,
		"vocab_words_embeddings": vocab_words_embeddings,
		"vocabulary": vocabulary,
	}
	return _out
