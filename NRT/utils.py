# Ben Kabongo
# January 2025

import evaluate
import nltk
import numpy as np
import random
import re
import torch
import torch.nn.functional as F

from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from typing import *


def rating_evaluation(config: Any, predictions: List[float], references: List[float]) -> Dict[str, float]:
    actual_ratings = torch.tensor(references, dtype=torch.float32).to(config.device)
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32).to(config.device)
    rmse = torch.sqrt(F.mse_loss(predictions_tensor, actual_ratings))
    mae = F.l1_loss(predictions_tensor, actual_ratings)
    return {'rmse': rmse.item(), 'mae': mae.item()}


def review_evaluation(config: Any, predictions: List[str], references: List[str]) -> Dict[str, Any]:
    references_list = [[ref] for ref in references]

    bleu_metric = evaluate.load("bleu")
    bleu_results = bleu_metric.compute(predictions=predictions, references=references_list)
    bleu_results["precision"] = np.mean(bleu_results["precisions"])

    bertscore_metric = evaluate.load("bertscore")
    bertscore_results = bertscore_metric.compute(
        predictions=predictions, references=references, lang=config.lang,
        device=config.device
    )
    bertscore_results["precision"] = np.mean(bertscore_results["precision"])
    bertscore_results["recall"] = np.mean(bertscore_results["recall"])
    bertscore_results["f1"] = np.mean(bertscore_results["f1"])

    meteor_metric = evaluate.load("meteor")
    meteor_results = meteor_metric.compute(predictions=predictions, references=references)

    rouge_metric = evaluate.load("rouge")
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)

    return {
        "n_examples": len(predictions),
        "meteor": float(meteor_results["meteor"]),
        "bleu": float(bleu_results["bleu"]),
        "rouge1": float(rouge_results["rouge1"]),
        "rouge2": float(rouge_results["rouge2"]),
        "rougeL": float(rouge_results["rougeL"]),
        "rougeLsum": float(rouge_results["rougeLsum"]),
        "bertscore.precision": float(bertscore_results["precision"]),
        "bertscore.recall": float(bertscore_results["recall"]),
        "bertscore.f1": float(bertscore_results["f1"]),
    }


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


def preprocess_text(text: str, config: Any, max_length: int=-1) -> str:
    if config.lower_flag: text = text.lower()
    if config.delete_punctuation_flag: text = delete_punctuation(text)
    if config.delete_stopwords_flag: text = delete_stopwords(text)
    if config.delete_non_ascii_flag: text = delete_non_ascii(text)
    if config.stem_flag: text = stem(text)
    if config.lemmatize_flag: text = lemmatize(text)
    if max_length > 0:
        text = str(text).strip().split()
        if len(text) > max_length:
            text = text[:max_length]
        text = " ".join(text)
    return text


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)