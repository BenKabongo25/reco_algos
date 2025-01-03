# Ben Kabongo
# May 2024

# Vocabulary

import json
import pandas as pd
from tqdm import tqdm
from typing import *


class Vocabulary:

    def __init__(self, default_add: bool=True):
        self._elements2ids = {}
        self._ids2elements = {}
        self.n_elements = 0
        self.default_add = default_add

    def add_element(self, element: Union[int, float, str]):
        if element not in self._elements2ids:
            self._elements2ids[element] = self.n_elements
            self._ids2elements[self.n_elements] = element
            self.n_elements += 1

    def add_elements(self, elements: List[Union[int, float, str]]):
        for element in tqdm(elements, "Vocabulary creation", colour="green"):
            self.add_element(element)

    def __len__(self):
        return self.n_elements
    
    def id2element(self, id: int) -> Union[int, float, str]:
        return self._ids2elements[id]
    
    def element2id(self, element: Union[int, float, str]) -> int:
        if element not in self._elements2ids:
            if self.default_add:
                self.add_element(element)
            else:
                return None
        return self._elements2ids[element]
    
    def ids2elements(self, ids: List[int]) -> List[Union[int, float, str]]:
        return [self._ids2elements[id] for id in ids]
    
    def elements2ids(self, elements: List[Union[int, float, str]]) -> List[int]:
        return [self.element2id(element) for element in elements]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"elements2ids": self._elements2ids, "ids2elements": self._ids2elements, "n_elements": self.n_elements}, f)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
            self._elements2ids = data["elements2ids"]
            self._ids2elements = data["ids2elements"]
            self.n_elements = data["n_elements"]


class WordVocabulary(Vocabulary):
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, default_add: bool=True, oov: bool=True):
        super().__init__(default_add)
        self.add_element("<PAD>")
        self.add_element("<EOS>")
        self.add_element("<SOS>")
        if oov:
            self.add_element("<OOV>")
        self.oov = oov

    


def create_vocab_from_df(metadata_df: pd.DataFrame, element_column: str) -> Vocabulary:
    elements = metadata_df[element_column].unique()
    vocab = Vocabulary()
    vocab.add_elements(elements)
    return vocab


def to_vocab_id(element, vocabulary: Vocabulary) -> int:
    return vocabulary.element2id(element)