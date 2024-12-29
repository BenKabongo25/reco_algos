# Ben Kabongo
# December 2024

import nltk
import numpy as np
import pickle
import os
import re

from nltk.stem.snowball import EnglishStemmer
from scipy.stats import dirichlet, bernoulli, beta
from tqdm import tqdm


def gibbs_sampling_topic_model(config, data, vocabulary, Beta_w, Alpha_u, Gamma_i, eta):
    """ Gibbs Sampling for topic model parameter estimation.

    Parameters:
    - config: Configuration object with n_factors, n_users, n_items, n_aspects, gibbs_sampling_iterations, etc.
    - data_df: Data with user-item reviews.
    - vocabulary: Vocabulary of words.
    - Beta_w: Dirichlet parameters for topic-word distributions.
    - Alpha_u: Dirichlet parameters for user-topic distributions.
    - Gamma_i: Dirichlet parameters for item-topic distributions.
    - eta: Beta priors for Bernoulli parameter.

    Returns:
    - Updated topic model parameters.
    """
    
    Phi = dirichlet.rvs(Beta_w, size=config.n_factors)  # (n_factors, vocab_size) Topic-word distributions
    Theta_u = dirichlet.rvs(Alpha_u, size=config.n_users) # (n_users, n_factors) User topic distributions
    Psi_i = dirichlet.rvs(Gamma_i, size=config.n_items) # (n_items, n_factors) Item topic distributions
    Pi_u = beta.rvs(eta[0], eta[1], size=config.n_users)  # (n_users,) Bernoulli parameter for users

    # Gibbs Sampling
    old_params = [Phi.copy(), Theta_u.copy(), Psi_i.copy(), Pi_u.copy()]
    for _ in tqdm(range(config.gibbs_sampling_iterations), desc="Gibbs Sampling", 
                  total=config.gibbs_sampling_iterations, colour="cyan"):
        
        sampled_reviews = []
        
        for (u, i, review) in data:
            sampled_review = []
            
            for sentence in review:
                y = bernoulli.rvs(Pi_u[u])
                    
                if y == 0: # Based on user's preferences
                    z_s = np.random.choice(np.arange(config.n_factors), p=Theta_u[u])
                else: # Based on item's characteristics
                    z_s = np.random.choice(np.arange(config.n_factors), p=Psi_i[i])

                sampled_sentence = []
                for word in sentence:
                    if word not in vocabulary:
                        continue
                    #w = np.random.choice(np.arange(config.vocabulary_size), p=Phi[z_s])
                    w = vocabulary[word]
                    sampled_sentence.append(w)
                
                if len(sampled_sentence) == 0:
                    continue
                sampled_review.append((y, z_s, sampled_sentence))

            if len(sampled_review) > 0:
                sampled_reviews.append((u, i, sampled_review))
                
        # Update distributions
        Phi_counts = np.zeros_like(Phi)
        Theta_u_counts = np.zeros_like(Theta_u)
        Psi_i_counts = np.zeros_like(Psi_i)
        Pi_u0_counts = np.zeros_like(Pi_u)
        Pi_u1_counts = np.zeros_like(Pi_u)

        for u, i, review in sampled_reviews:
            for y, z_s, sentence in review:
                Theta_u_counts[u, z_s] += 1
                Psi_i_counts[i, z_s] += 1
                if y == 0:
                    Pi_u0_counts[u] += 1
                else:
                    Pi_u1_counts[u] += 1
                for w in sentence:
                    Phi_counts[z_s, w] += 1

        # Normalize
        Phi = (Phi_counts + Beta_w) / (Phi_counts.sum(axis=1, keepdims=True) + Beta_w.sum())
        Theta_u = (Theta_u_counts + Alpha_u) / (Theta_u_counts.sum(axis=1, keepdims=True) + Alpha_u.sum())
        Psi_i = (Psi_i_counts + Gamma_i) / (Psi_i_counts.sum(axis=1, keepdims=True) + Gamma_i.sum())
        Pi_u = (Pi_u0_counts + eta[0]) / (Pi_u0_counts + Pi_u1_counts + eta[0] + eta[1])

        new_params = [Phi, Theta_u, Psi_i, Pi_u]
        if has_converged(old_params, new_params):
            print("Gibbs Sampling converged.")
            break
        old_params = new_params

    params = {'Phi': Phi, 'Theta_u': Theta_u, 'Psi_i': Psi_i, 'Pi_u': Pi_u}
    params_path = os.path.join(config.save_dir, 'params.pkl')
    pickle.dump(params, open(params_path, 'wb'))
    return params


def has_converged(old_params, new_params, tolerance=1e-4):
    changes = [np.abs(new - old).max() for old, new in zip(old_params, new_params)]
    return all(change < tolerance for change in changes)


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


def process_text(text: str) -> str:
    text = text.lower()
    text = delete_punctuation(text)
    text = delete_stopwords(text)
    text = delete_non_ascii(text)
    return text


def process_data(config, data_df):
    """
    Process data for ATM.
    """
    stemmer = EnglishStemmer()

    data = []
    vocabulary = {}

    
    for index in tqdm(range(len(data_df)), desc="Processing Data", total=len(data_df), colour="cyan"):
        row = data_df.iloc[index]
        user_id = row["user_id"]
        item_id = row["item_id"]
        review = process_text(row["review"])
        sentences = nltk.sent_tokenize(review)
        
        stemmed_sentences = []
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = 0
                vocabulary[token] += 1
            stemmed_sentence = [stemmer.stem(token) for token in tokens]
            stemmed_sentences.append(stemmed_sentence)

        data.append((user_id, item_id, stemmed_sentences))

    if config.vocabulary_size == -1:
        config.vocabulary_size = len(vocabulary)
    else:
        vocabulary = {k: v for k, v in sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)[:config.vocabulary_size]}
    vocabulary = {k: i for i, k in enumerate(vocabulary.keys())}

    return data, vocabulary
