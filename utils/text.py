# Ben Kabongo
# May 2024

import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from typing import Any


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


def postprocess_text(text: str) -> str:
    # https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    text = re.sub('\'s', ' \'s', text)
    text = re.sub('\'m', ' \'m', text)
    text = re.sub('\'ve', ' \'ve', text)
    text = re.sub('n\'t', ' n\'t', text)
    text = re.sub('\'re', ' \'re', text)
    text = re.sub('\'d', ' \'d', text)
    text = re.sub('\'ll', ' \'ll', text)
    text = re.sub('\(', ' ( ', text)
    text = re.sub('\)', ' ) ', text)
    text = re.sub(',+', ' , ', text)
    text = re.sub(':+', ' , ', text)
    text = re.sub(';+', ' . ', text)
    text = re.sub('\.+', ' . ', text)
    text = re.sub('!+', ' ! ', text)
    text = re.sub('\?+', ' ? ', text)
    text = re.sub(' +', ' ', text).strip()
    return text
