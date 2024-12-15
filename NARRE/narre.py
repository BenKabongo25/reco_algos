# Ben Kabongo
# December 2024

# NARRE: Neural Attention Rating Regression with Review-level Explanations
# Doc: https://cornac.readthedocs.io/en/stable/api_ref/models.html#module-cornac.models.narre.recom_narre


import argparse
import cornac
import json
import numpy as np

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from utils.cornac_common import main


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    # NARRE Cornac args
    args.add_argument("--embedding_size", type=int, default=100)
    args.add_argument("--words_embedding_path", type=str, default="")
    args.add_argument("--id_embedding_size", type=int, default=32)
    args.add_argument("--n_factors", type=int, default=32)
    args.add_argument("--attention_size", type=int, default=16)
    args.add_argument("--kernel_sizes", type=int, nargs="+", default=[3])
    args.add_argument("--n_filters", type=int, default=64)
    args.add_argument("--dropout_rate", type=float, default=0.5)
    args.add_argument("--max_text_length", type=int, default=128)
    args.add_argument("--max_num_review", type=int, default=64)
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument("--max_iter", type=int, default=50)
    args.add_argument("--optimizer", type=str, default="adam") # ‘adam’ or ‘rmsprop’.
    args.add_argument("--learning_rate", type=float, default=0.001)
    args.add_argument("--model_selection", type=str, default="best") # ‘last’ or ‘best’.
    args.add_argument("--user_based", action=argparse.BooleanOptionalAction)
    args.set_defaults(user_based=False)
    args.add_argument("--trainable", action=argparse.BooleanOptionalAction)
    args.set_defaults(trainable=True)
    args.add_argument("--seed", type=int, default=None)

    # Data args
    args.add_argument("--data_path", type=str, default="")

    args.add_argument("--train_path", type=str, default="")
    args.add_argument("--val_path", type=str, default="")
    args.add_argument("--test_path", type=str, default="")    
    
    args.add_argument("--train_size", type=float, default=0.8)
    args.add_argument("--test_size", type=float, default=0.1)
    args.add_argument("--val_size", type=float, default=0.1)

    args.add_argument("--fmt", type=str, default="UIRT")
    args.add_argument("--user_column", type=str, default="user_id")
    args.add_argument("--item_column", type=str, default="item_id")
    args.add_argument("--rating_column", type=str, default="rating")
    args.add_argument("--timestamp_column", type=str, default="timestamp")
    args.add_argument("--timestamp", action=argparse.BooleanOptionalAction)
    args.set_defaults(timestamp=False)

    args.add_argument("--rating_threshold", type=float, default=4.0)
    args.add_argument("--exclude_unknowns", action=argparse.BooleanOptionalAction)
    args.set_defaults(exclude_unknowns=False)

    args.add_argument("--review_modality", action=argparse.BooleanOptionalAction)
    args.set_defaults(review_modality=True)
    args.add_argument("--review_column", type=str, default="review")

    # Metrics
    args.add_argument("--rmse", action=argparse.BooleanOptionalAction)
    args.set_defaults(rmse=True)
    args.add_argument("--mae", action=argparse.BooleanOptionalAction)
    args.set_defaults(mae=True)
    args.add_argument("--rating_user_based", action=argparse.BooleanOptionalAction)
    args.set_defaults(rating_user_based=False)

    args.add_argument("--ranking_k", type=int, default=10)
    args.add_argument("--precision", action=argparse.BooleanOptionalAction)
    args.set_defaults(precision=False)
    args.add_argument("--recall", action=argparse.BooleanOptionalAction)
    args.set_defaults(recall=False)
    args.add_argument("--f1", action=argparse.BooleanOptionalAction)
    args.set_defaults(f1=False)
    args.add_argument("--auc", action=argparse.BooleanOptionalAction)
    args.set_defaults(auc=False)
    args.add_argument("--ndcg", action=argparse.BooleanOptionalAction)
    args.set_defaults(ndcg=False)
    args.add_argument("--hit", action=argparse.BooleanOptionalAction)
    args.set_defaults(hit=False)
    args.add_argument("--map", action=argparse.BooleanOptionalAction)
    args.set_defaults(map=False)
    args.add_argument("--mrr", action=argparse.BooleanOptionalAction)
    args.set_defaults(mrr=False)

    # Experiment
    args.add_argument("--exp_name", type=str, default="MF")
    args.add_argument("--show_validation", action=argparse.BooleanOptionalAction)
    args.set_defaults(show_validation=False)
    args.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    args.set_defaults(verbose=True)
    args.add_argument("--save_dir", type=str, default="")

    config = args.parse_args()

    # Model
    pretrained_word_embeddings = {}
    with open(config.words_embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            pretrained_word_embeddings[word] = vector

    model = cornac.models.NARRE(
        embedding_size=config.embedding_size,
        id_embedding_size=config.id_embedding_size,
        n_factors=config.n_factors,
        attention_size=config.attention_size,
        kernel_sizes=config.kernel_sizes,
        n_filters=config.n_filters,
        dropout_rate=config.dropout_rate,
        max_text_length=config.max_text_length,
        max_num_review=config.max_num_review,
        batch_size=config.batch_size,
        max_iter=config.max_iter,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        model_selection=config.model_selection,
        trainable=config.trainable,
        init_params={"pretrained_word_embeddings": pretrained_word_embeddings},
        seed=config.seed
    )

    main(config, models=[model])
