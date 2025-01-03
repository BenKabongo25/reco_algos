# Ben Kabongo
# January 2025

# NRT: Neural Rating Regression with Abstractive Tips Generation for Recommendation
# Paper: https://arxiv.org/pdf/1708.00154
# Source: https://github.com/lileipisces/NRT


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NRTEncoder(nn.Module):

    def __init__(self, n_users, n_items, d_model, hidden_size, n_layers=4, max_rating=5, min_rating=1):
        super(NRTEncoder, self).__init__()
        self.max_rating = int(max_rating)
        self.min_rating = int(min_rating)
        self.n_ratings = self.max_rating - self.min_rating + 1

        self.user_embeddings = nn.Embedding(n_users, d_model)
        self.item_embeddings = nn.Embedding(n_items, d_model)

        layers = [nn.Linear(d_model * 2, hidden_size), nn.Sigmoid()]
        for _ in range(n_layers):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Sigmoid()])
        layers.append(nn.Linear(hidden_size, 1))
        self.layers = nn.Sequential(*layers)

        self.encoder = nn.Sequential(
            nn.Linear(d_model * 2 + self.n_ratings, hidden_size),
            nn.Tanh()
        )

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

    def forward(self, U_ids: torch.Tensor, I_ids: torch.Tensor) -> Tuple[torch.Tensor]:
        U_embeddings = self.user_embeddings(U_ids)  # (batch_size, d_model)
        I_embeddings = self.item_embeddings(I_ids)  # (batch_size, d_model)
        embeddings = torch.cat([U_embeddings, I_embeddings], dim=1)  # (batch_size, d_model * 2)
        rating = self.layers(embeddings).squeeze(1)  # (batch_size,)

        rating_int = torch.clamp(rating, min=self.min_rating, max=self.max_rating).type(torch.int64)  # (batch_size,)
        rating_one_hot = F.one_hot(rating_int - self.min_rating, num_classes=self.n_ratings)  # (batch_size, n_ratings)

        encoder_input = torch.cat([U_embeddings, I_embeddings, rating_one_hot], dim=1)  # (batch_size, d_model * 2 + num_rating)
        encoder_state = self.encoder(encoder_input).unsqueeze(0) # (1, batch_size, hidden_size)

        return rating, encoder_state


class GRUDecoder(nn.Module):

    def __init__(self, n_tokens, d_model, hidden_size):
        super(GRUDecoder, self).__init__()
        self.word_embeddings = nn.Embedding(n_tokens, d_model)
        self.gru = nn.GRU(d_model, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, n_tokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def forward(self, seq, hidden):  # seq: (batch_size, seq_len), hidden: (nlayers, batch_size, hidden_size)
        seq_emb = self.word_embeddings(seq)  # (batch_size, seq_len, d_model)
        output, hidden = self.gru(seq_emb, hidden)  # (batch_size, seq_len, hidden_size) vs. (nlayers, batch_size, hidden_size)
        decoded = self.linear(output)  # (batch_size, seq_len, n_tokens)
        return F.log_softmax(decoded, dim=-1), hidden


class NRT(nn.Module):

    def __init__(self, n_users, n_items, n_tokens, d_model, hidden_size, n_layers_mlp=4, max_rating=5, min_rating=1):
        super(NRT, self).__init__()
        self.encoder = NRTEncoder(n_users, n_items, d_model, hidden_size, n_layers_mlp, max_rating, min_rating)
        self.decoder = GRUDecoder(n_tokens, d_model, hidden_size)

    def forward(self, user, item, seq):  # (batch_size,) vs. (batch_size, seq_len)
        rating, hidden = self.encoder(user, item)
        log_word_prob, _ = self.decoder(seq, hidden)
        return rating, log_word_prob  # (batch_size,) vs. (batch_size, seq_len, n_tokens)
    
    def generate(self, U_ids, I_ids, review_length, bos_token):
        inputs = torch.tensor([bos_token]).unsqueeze(0).to(U_ids.device)  # (1, 1)
        inputs = inputs.repeat(U_ids.size(0), 1) # (batch_size, 1)
        hidden = None
        ids = inputs
        for idx in range(review_length):
            # produce a word at each step
            if idx == 0:
                rating, hidden = self.encoder(U_ids, I_ids)
                logits, hidden = self.decoder(inputs, hidden)
            else:
                logits, hidden, hidden_c = self.decoder(inputs, hidden, hidden_c)
            word_prob = logits.squeeze().exp()  # (batch_size, n_tokens)
            inputs = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
            ids = torch.cat([ids, inputs], 1)  # (batch_size, len++)
        ids = ids[:, 1:].tolist()  # remove bos

        reviews_hat = []
        for i in range(len(ids)):
            review = word_dict.ids2tokens(ids[i])
            review = [word for word in review if word not in ['<bos>', '<eos>', '<pad>', '<unk>']]
            reviews_hat.append(" ".join(review))
        return ids
    
    def save(self, save_model_path: str):
        torch.save(self.state_dict(), save_model_path)

    def load(self, save_model_path: str):
        self.load_state_dict(torch.load(save_model_path))